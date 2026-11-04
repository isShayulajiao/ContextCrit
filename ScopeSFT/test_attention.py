"""CPU numerical and retained-activation checks for exact blocked attention."""
import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from ScopeSFT.attention import chunked_causal_attention, register_chunked_attention


class ChunkedAttentionTests(unittest.TestCase):
    def fixture(self, grouped=True):
        torch.manual_seed(21)
        heads = 2 if grouped else 4
        return [torch.randn(2, h, 9, 3, dtype=torch.float64, requires_grad=True)
                for h in (4, heads, heads)]

    def dense(self, query, key, value, padding, scaling=0.37):
        length = query.shape[2]
        allowed = torch.ones(length, length, dtype=torch.bool).tril()[None, None]
        if padding is not None:
            allowed = allowed & padding[:, None, None, :].bool()
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed, scale=scaling,
            enable_gqa=query.shape[1] != key.shape[1],
        )

    def test_every_key_causal_padding_and_gqa_match_dense_outputs_and_gradients(self):
        masks = [None, torch.tensor([[1] * 6 + [0] * 3, [0, 0] + [1] * 7])]
        for grouped in (False, True):
            for padding in masks:
                for block_size in (1, 4, 20):
                    with self.subTest(grouped=grouped, padding=padding is not None, block_size=block_size):
                        inputs = self.fixture(grouped)
                        dense = self.dense(*inputs, padding)
                        blocked = chunked_causal_attention(
                            *inputs, padding, query_chunk_size=block_size, scaling=0.37)
                        torch.testing.assert_close(blocked, dense, rtol=1e-12, atol=1e-12)
                        target = torch.randn_like(dense)
                        dense_grad = torch.autograd.grad((dense * target).sum(), inputs)
                        blocked_grad = torch.autograd.grad((blocked * target).sum(), inputs)
                        for actual, expected in zip(blocked_grad, dense_grad):
                            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)

    def test_checkpoint_reuses_dropout_mask_in_backward_without_changing_rng(self):
        source = self.fixture()
        inputs = [x.detach().clone().requires_grad_() for x in source]
        torch.manual_seed(52)
        actual = chunked_causal_attention(*inputs, query_chunk_size=4, dropout=0.3)
        actual_grad = torch.autograd.grad(actual.square().sum(), inputs)
        actual_random_after = torch.rand(4)

        # The uncheckpointed reference executes exactly the same blocks and RNG
        # draws; a single dense dropout call need not consume RNG identically.
        def direct(function, *arguments, **kwargs):
            return function(*arguments)

        inputs = [x.detach().clone().requires_grad_() for x in source]
        torch.manual_seed(52)
        with patch("ScopeSFT.attention.checkpoint", direct):
            expected = chunked_causal_attention(*inputs, query_chunk_size=4, dropout=0.3)
            expected_grad = torch.autograd.grad(expected.square().sum(), inputs)
        expected_random_after = torch.rand(4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_item, expected_item in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual_item, expected_item, rtol=0, atol=0)
        self.assertTrue(torch.equal(actual_random_after, expected_random_after))

    def test_fp16_head_dim_256_math_fallback_is_finite_and_matches_dense(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch.manual_seed(61)
        inputs = [torch.randn(1, heads, 11, 256, dtype=torch.float16, requires_grad=True)
                  for heads in (4, 1, 1)]
        padding = torch.tensor([[1] * 9 + [0] * 2])
        # Turing may not have a fused SDPA kernel for the official 256-wide
        # heads.  This explicitly verifies the bounded math fallback.
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            expected = self.dense(*inputs, padding, scaling=256 ** -0.5)
            actual = chunked_causal_attention(
                *inputs, padding, query_chunk_size=3, scaling=256 ** -0.5)
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
            expected_gradients = torch.autograd.grad(expected.float().square().sum(), inputs)
            gradients = torch.autograd.grad(actual.float().square().sum(), inputs)
        for actual_gradient, expected_gradient in zip(gradients, expected_gradients):
            self.assertTrue(torch.isfinite(actual_gradient).all())
            # Summing separate key/value block gradients changes FP16 rounding
            # order; compare both elementwise error and the total relative norm.
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-3, atol=5e-3)
            relative_error = ((actual_gradient - expected_gradient).float().norm()
                              / expected_gradient.float().norm())
            self.assertLess(relative_error.item(), 1e-3)

    def test_forward_does_not_retain_any_score_or_causal_mask_blocks(self):
        inputs = self.fixture()
        padding = torch.ones(2, 9, dtype=torch.bool)
        saved = []

        def pack(tensor):
            saved.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            result = chunked_causal_attention(*inputs, padding, query_chunk_size=4)
        # Non-reentrant checkpoint stores only Q/K/V views, padding, and its
        # empty dummy.  In particular it stores neither [block,L] boolean masks
        # nor [B,H,block,L] scores, even summed across all chunks.
        self.assertTrue(saved)
        for shape in saved:
            self.assertTrue(shape == (0,) or shape == (2, 9)
                            or (len(shape) == 4 and shape[-1] == 3), shape)
        result.sum().backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in inputs))

    def test_hf_mask_hook_returns_only_padding_and_rejects_cache_or_custom_mask(self):
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, causal_mask_function

        name = register_chunked_attention(4)
        hook = ALL_MASK_ATTENTION_FUNCTIONS[name]
        padding = torch.ones(2, 9, dtype=torch.bool)
        arguments = dict(batch_size=2, cache_position=torch.arange(9), kv_length=9,
                         attention_mask=padding, mask_function=causal_mask_function)
        self.assertIs(hook(**arguments), padding)
        for changes in ({"kv_offset": 1}, {"cache_position": torch.arange(1, 10)},
                        {"mask_function": lambda *args: True}, {"use_vmap": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                hook(**(arguments | changes))

    def test_registered_backend_matches_tiny_qwen3_full_forward_and_gradient(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(6)
        config = Qwen3Config(vocab_size=23, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8,
                             max_position_embeddings=64, attention_dropout=0.0)
        config._attn_implementation = "eager"
        dense = Qwen3ForCausalLM(config).eval()
        blocked = copy.deepcopy(dense)
        blocked.set_attn_implementation(register_chunked_attention(3))
        tokens = torch.tensor([[1, 2, 3, 4, 5, 0], [2, 3, 4, 5, 6, 7]])
        padding = (tokens != 0).long()
        dense_result = dense(input_ids=tokens, attention_mask=padding, use_cache=False).logits
        blocked_result = blocked(input_ids=tokens, attention_mask=padding, use_cache=False).logits
        torch.testing.assert_close(blocked_result, dense_result, rtol=1e-5, atol=1e-6)
        weights = torch.randn_like(dense_result)
        dense_gradient = torch.autograd.grad((dense_result * weights).sum(), dense.model.layers[0].self_attn.q_proj.weight)[0]
        blocked_gradient = torch.autograd.grad((blocked_result * weights).sum(), blocked.model.layers[0].self_attn.q_proj.weight)[0]
        torch.testing.assert_close(blocked_gradient, dense_gradient, rtol=1e-4, atol=1e-6)

    def test_qwen35_hybrid_with_nested_gradient_checkpointing_matches_dense(self):
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

        torch.manual_seed(16)
        config = Qwen3_5TextConfig(
            vocab_size=23, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            linear_num_key_heads=1, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
            max_position_embeddings=64, attention_dropout=0.0,
            rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5,
                             "mrope_section": [1, 1, 0]},
        )
        config._attn_implementation = "eager"
        dense = Qwen3_5ForCausalLM(config).train()
        blocked = copy.deepcopy(dense)
        blocked.set_attn_implementation(register_chunked_attention(3))
        blocked.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        tokens = torch.tensor([[1, 2, 3, 4, 5, 0], [2, 3, 4, 5, 6, 7]])
        padding = (tokens != 0).long()
        dense_result = dense(input_ids=tokens, attention_mask=padding, use_cache=False).logits
        blocked_result = blocked(input_ids=tokens, attention_mask=padding, use_cache=False).logits
        torch.testing.assert_close(blocked_result, dense_result, rtol=1e-5, atol=1e-6)
        weights = torch.randn_like(dense_result)
        dense_parameters = tuple(dense.parameters())
        blocked_parameters = tuple(blocked.parameters())
        dense_gradients = torch.autograd.grad((dense_result * weights).sum(), dense_parameters)
        blocked_gradients = torch.autograd.grad((blocked_result * weights).sum(), blocked_parameters)
        for actual, expected in zip(blocked_gradients, dense_gradients):
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=2e-6)

    def test_invalid_masks_and_shapes_fail_before_attention(self):
        inputs = self.fixture()
        for kwargs in ({"query_chunk_size": 0}, {"query_chunk_size": True}, {"dropout": -1},
                       {"scaling": float("inf")}, {"attention_mask": torch.ones(2, 1, 9, 9)},
                       {"attention_mask": torch.full((2, 9), 0.5)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                chunked_causal_attention(*inputs, **kwargs)
        with self.assertRaises(ValueError):
            chunked_causal_attention(inputs[0][:, :, :1], inputs[1], inputs[2])


if __name__ == "__main__":
    unittest.main()

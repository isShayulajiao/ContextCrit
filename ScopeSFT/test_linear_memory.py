"""Exact recurrence, backward continuity and workspace checks, CPU only."""
from unittest import mock
import unittest

import torch

from ScopeSFT.linear_memory import chunked_native_delta_rule


def fixture(length=145, *, dtype=torch.float32, dimension=5, initial=True):
    torch.manual_seed(319)
    shapes = ((1, length, 2, dimension), (1, length, 2, dimension),
              (1, length, 2, 3), (1, length, 2), (1, length, 2))
    result = []
    for index, shape in enumerate(shapes):
        value = torch.randn(shape) * 0.1
        if index == 3:
            value = -value.abs() - 0.03
        if index == 4:
            value = value.sigmoid()
        result.append(value.to(dtype).requires_grad_())
    state = torch.randn(1, 2, dimension, 3).to(dtype) * 0.1 if initial else None
    if state is not None:
        state.requires_grad_()
    return result, state


def copies(inputs, state, *, dtype=None):
    result = [t.detach().clone().to(dtype or t.dtype).requires_grad_() for t in inputs]
    copied_state = None if state is None else state.detach().clone().to(dtype or state.dtype).requires_grad_()
    return result, copied_state


def native_fp32(inputs, state, *, normalize=True, chunk_size=64):
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    with torch.autocast(device_type=inputs[0].device.type, enabled=False):
        return torch_chunk_gated_delta_rule(
            *[t.float() for t in inputs], chunk_size=chunk_size,
            initial_state=None if state is None else state.float(),
            output_final_state=True, use_qk_l2norm_in_kernel=normalize,
        )


class LinearMemoryTests(unittest.TestCase):
    def test_segment_outputs_and_every_input_gradient_match_native(self):
        for length, segment_size, normalize, initial in (
            (65, 64, True, True), (145, 64, True, False),
            (261, 128, True, True), (145, 128, False, True),
            (64, 512, True, True),
        ):
            with self.subTest(length=length, segment=segment_size, normalized=normalize, initial=initial):
                inputs, state = fixture(length, initial=initial)
                reference_inputs, reference_state = copies(inputs, state)
                expected, expected_state = native_fp32(reference_inputs, reference_state, normalize=normalize)
                actual, actual_state = chunked_native_delta_rule(
                    *inputs, initial_state=state, output_final_state=True,
                    use_qk_l2norm_in_kernel=normalize, sequence_chunk_size=segment_size,
                )
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(actual_state, expected_state, rtol=1e-5, atol=1e-6)
                torch.manual_seed(43)
                weights = torch.randn_like(actual)
                state_weights = torch.randn_like(actual_state)
                actual_loss = (actual * weights).sum() + (actual_state * state_weights).sum()
                expected_loss = (expected * weights).sum() + (expected_state * state_weights).sum()
                actual_args = tuple(inputs) + (() if state is None else (state,))
                expected_args = tuple(reference_inputs) + (() if reference_state is None else (reference_state,))
                actual_grads = torch.autograd.grad(actual_loss, actual_args)
                expected_grads = torch.autograd.grad(expected_loss, expected_args)
                for index, (gradient, reference_gradient) in enumerate(zip(actual_grads, expected_grads)):
                    with self.subTest(input=index):
                        self.assertTrue(torch.isfinite(gradient).all())
                        torch.testing.assert_close(gradient, reference_gradient, rtol=3e-4, atol=8e-6)

    def test_late_loss_reaches_early_tokens_and_initial_state(self):
        inputs, state = fixture(199)
        output, final_state = chunked_native_delta_rule(
            *inputs, initial_state=state, output_final_state=True,
            use_qk_l2norm_in_kernel=True, sequence_chunk_size=64,
        )
        # Only the final segment's activation is supervised. Earlier tokens and
        # the external initial state must still receive true BPTT gradients.
        loss = output[:, -1].sum()
        gradient_value, gradient_key, gradient_state = torch.autograd.grad(loss, (inputs[2], inputs[1], state))
        self.assertGreater(gradient_value[:, :64].abs().sum().item(), 0)
        self.assertGreater(gradient_key[:, :64].abs().sum().item(), 0)
        self.assertGreater(gradient_state.abs().sum().item(), 0)
        self.assertIsNotNone(final_state.grad_fn)

    def test_hybrid_lora_nested_checkpoint_and_late_scope_ct_match_native(self):
        from functools import partial
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
        from ScopeSFT.test_training_forward import adapted_model, dense_loss, trainable_gradients

        reference = adapted_model("qwen35", patched=False)
        bounded = adapted_model("qwen35", patched=True, chunked=True, checkpointed=True)
        bounded.load_state_dict(reference.state_dict())
        layers = []
        for module in bounded.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.chunk_gated_delta_rule = partial(chunked_native_delta_rule, sequence_chunk_size=64)
                layers.append(module)
        self.assertEqual(len(layers), 1)
        torch.manual_seed(82)
        ids = torch.randint(1, 23, (2, 145))
        labels = torch.full_like(ids, -100)
        labels[0, -5:] = ids[0, -5:]
        labels[1, -3:] = ids[1, -3:]
        batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                     critical_positions=torch.tensor([144, -1]),
                     positive_ids=torch.tensor([ids[0, -1].item(), -1]),
                     negative_ids=torch.tensor([(ids[0, -1].item() + 1) % 23, -1]))
        expected, _ = dense_loss(reference, batch)
        actual = bounded(**batch, scope_compute_loss=True, use_cache=False).loss
        expected.backward()
        actual.backward()
        torch.testing.assert_close(actual, expected)
        reference_gradients, gradients = trainable_gradients(reference), trainable_gradients(bounded)
        self.assertEqual(reference_gradients.keys(), gradients.keys())
        for name in reference_gradients:
            with self.subTest(parameter=name):
                torch.testing.assert_close(gradients[name], reference_gradients[name], rtol=7e-4, atol=2e-6)

    def test_fp16_zero_and_low_amplitude_qk_have_finite_gradients(self):
        for amplitude in (0.0, 0.001):
            with self.subTest(amplitude=amplitude):
                inputs, state = fixture(129, dtype=torch.float16, dimension=128)
                with torch.no_grad():
                    inputs[0].fill_(amplitude)
                    inputs[1].fill_(amplitude)
                ref_inputs, ref_state = copies(inputs, state, dtype=torch.float32)
                expected, expected_state = native_fp32(ref_inputs, ref_state)
                with torch.autocast("cpu", dtype=torch.float16):
                    actual, actual_state = chunked_native_delta_rule(
                        *inputs, initial_state=state, output_final_state=True,
                        use_qk_l2norm_in_kernel=True, sequence_chunk_size=64,
                    )
                    loss = actual.float().square().mean() + actual_state.square().mean()
                expected_loss = expected.square().mean() + expected_state.square().mean()
                gradients = torch.autograd.grad(loss, tuple(inputs) + (state,))
                ref_gradients = torch.autograd.grad(expected_loss, tuple(ref_inputs) + (ref_state,))
                self.assertEqual(actual.dtype, torch.float16)
                self.assertEqual(actual_state.dtype, torch.float32)
                torch.testing.assert_close(actual.float(), expected, rtol=2e-3, atol=2e-4)
                for actual_gradient, reference_gradient in zip(gradients, ref_gradients):
                    self.assertTrue(torch.isfinite(actual_gradient).all())
                    torch.testing.assert_close(actual_gradient.float(), reference_gradient, rtol=8e-3, atol=3e-5)

    def test_native_calls_are_bounded_and_checkpoint_keeps_original_dtype_views(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as native
        inputs, state = fixture(1041, dtype=torch.float16, dimension=7)
        shapes, calls = [], []
        original = native.torch_chunk_gated_delta_rule

        def tracked(*arguments, **kwargs):
            calls.append((arguments[0].shape, arguments[0].dtype, torch.is_autocast_enabled("cpu")))
            return original(*arguments, **kwargs)

        def saved(tensor):
            shapes.append((tuple(tensor.shape), tensor.dtype))
            return tensor

        with mock.patch.object(native, "torch_chunk_gated_delta_rule", tracked):
            with torch.autograd.graph.saved_tensors_hooks(saved, lambda tensor: tensor):
                outputs, final_state = chunked_native_delta_rule(
                    *inputs, initial_state=state, output_final_state=True,
                    use_qk_l2norm_in_kernel=True, sequence_chunk_size=128,
                )
            forward_calls = len(calls)
            (outputs.float().square().mean() + final_state.square().mean()).backward()
        self.assertEqual(forward_calls, 9)
        self.assertGreater(len(calls), forward_calls)
        self.assertTrue(all(shape[1] <= 128 and dtype == torch.float32 and not amp
                            for shape, dtype, amp in calls))
        # Inputs to checkpoints stay FP16, and no native triangular/FP32 QKV
        # activation is retained during the full-sequence forward.
        self.assertTrue(any(dtype == torch.float16 and len(shape) == 4 for shape, dtype in shapes))
        self.assertFalse(any(dtype == torch.float32 and len(shape) == 4 and shape[1] > 2
                             for shape, dtype in shapes))
        self.assertFalse(any(len(shape) == 5 for shape, _ in shapes))
        self.assertTrue(all(torch.isfinite(t.grad).all() for t in inputs))

    def test_no_grad_and_optional_final_state(self):
        inputs, state = fixture(145)
        with torch.no_grad():
            actual, returned = chunked_native_delta_rule(
                *inputs, initial_state=state, sequence_chunk_size=64,
                use_qk_l2norm_in_kernel=True,
            )
            expected, _ = native_fp32(inputs, state)
        self.assertIsNone(returned)
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(actual, expected)

    def test_invalid_segment_sizes_and_shapes_fail_explicitly(self):
        inputs, state = fixture(65)
        for value in (0, -1, 32, 65, True, 64.0):
            with self.subTest(size=value), self.assertRaises(ValueError):
                chunked_native_delta_rule(*inputs, sequence_chunk_size=value)
        for value in (0, True, -1):
            with self.subTest(native_size=value), self.assertRaises(ValueError):
                chunked_native_delta_rule(*inputs, chunk_size=value)
        with self.assertRaises(ValueError):
            chunked_native_delta_rule(*inputs, initial_state=state[:, :, :-1])
        invalid = list(inputs)
        invalid[-1] = invalid[-1][:, :-1]
        with self.assertRaises(ValueError):
            chunked_native_delta_rule(*invalid)


if __name__ == "__main__":
    unittest.main()

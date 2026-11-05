"""Saved-activation offload correctness; actual CUDA copies require CUDA."""
from copy import deepcopy
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ScopeSFT.offload import saved_activation_context, resolve_activation_offload


class NestedCheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(8, 12)
        self.second = nn.Linear(12, 8)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        def layer(value):
            # Match the trainer's nested layer/inner-block checkpoint pattern.
            hidden = checkpoint(lambda z: self.dropout(self.first(z)).sin(), value,
                                use_reentrant=False, preserve_rng_state=True)
            return self.second(hidden).cos()
        return checkpoint(layer, x, use_reentrant=False, preserve_rng_state=True)


class ActivationOffloadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_default_policy_uses_actual_cuda_length_and_explicit_overrides(self):
        for tokens in (0, 3394, 16383, 16384, 36000, 65536):
            for cpu in (False, True):
                with self.subTest(tokens=tokens, cpu=cpu):
                    self.assertEqual(resolve_activation_offload("auto", cpu=cpu, max_tokens=tokens),
                                     not cpu and tokens >= 16384)
                    self.assertFalse(resolve_activation_offload("none", cpu=cpu, max_tokens=tokens))
                    self.assertTrue(resolve_activation_offload("cpu", cpu=cpu, max_tokens=tokens))
        for requested, cpu, tokens in (("invalid", False, 36000), ("auto", False, -1),
                                       ("auto", False, True), ("auto", "no", 36000)):
            with self.assertRaises(ValueError):
                resolve_activation_offload(requested, cpu=cpu, max_tokens=tokens)

    def test_cpu_nested_checkpoint_preserves_rng_loss_and_all_gradients(self):
        torch.manual_seed(19)
        expected_model = NestedCheckpointModel().train()
        actual_model = deepcopy(expected_model)
        expected_input = torch.randn(3, 9, 8, requires_grad=True)
        actual_input = expected_input.detach().clone().requires_grad_()
        torch.manual_seed(31)
        expected = expected_model(expected_input).square().mean()
        expected.backward()
        torch.manual_seed(31)
        with saved_activation_context(True, actual_model) as context:
            actual = actual_model(actual_input).square().mean()
        # Hooks remain available to backward after the context exits.
        actual.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_input.grad, expected_input.grad)
        for actual_parameter, expected_parameter in zip(actual_model.parameters(), expected_model.parameters()):
            self.assertIsNotNone(actual_parameter.grad)
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)
        self.assertGreater(context.stats["saved_tensors"], 0)
        self.assertEqual(context.stats["offloaded_tensors"], 0)
        self.assertEqual(context.stats["offloaded_bytes"], 0)

    def test_parameter_storage_aliases_and_tied_views_are_not_copied(self):
        model = nn.Linear(5, 7, bias=False)
        value = torch.randn(4, 5, requires_grad=True)
        with saved_activation_context(True, model) as context:
            result = nn.functional.linear(value, model.weight).square().sum()
        # Linear backward saves weight.T, a Tensor view rather than Parameter.
        self.assertGreater(context.stats["retained_parameter_tensors"], 0)
        self.assertEqual(context._pack(model.weight.T).value.untyped_storage().data_ptr(),
                         model.weight.untyped_storage().data_ptr())
        result.backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all())
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_disabled_context_does_not_shadow_existing_saved_tensor_hooks(self):
        saved = []
        value = torch.randn(5, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(
                lambda tensor: saved.append(tensor.shape) or tensor.detach(), lambda tensor: tensor):
            with saved_activation_context(False):
                result = value.square().sum()
        self.assertTrue(saved)
        result.backward()
        torch.testing.assert_close(value.grad, value.detach() * 2)

    def test_retained_graph_supports_repeated_gradient_requests(self):
        value = torch.randn(2, 3, requires_grad=True)
        with saved_activation_context(True):
            result = value.sin().square().sum()
        first = torch.autograd.grad(result, value, retain_graph=True)[0]
        second = torch.autograd.grad(result, value)[0]
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first, 2 * value.detach().sin() * value.detach().cos())

    def test_hybrid_peft_scope_loss_and_gradients_match_without_offload(self):
        from ScopeSFT.test_training_forward import adapted_model, training_batch, trainable_gradients

        reference = adapted_model("qwen35", patched=True, chunked=True, checkpointed=True)
        actual_model = adapted_model("qwen35", patched=True, chunked=True, checkpointed=True)
        batch = training_batch()
        expected = reference(**batch, scope_compute_loss=True, use_cache=False)
        expected.loss.backward()
        with saved_activation_context(True, actual_model):
            actual = actual_model(**batch, scope_compute_loss=True, use_cache=False)
        actual.loss.backward()
        torch.testing.assert_close(actual.loss, expected.loss)
        actual_gradients = trainable_gradients(actual_model)
        expected_gradients = trainable_gradients(reference)
        self.assertEqual(actual_gradients.keys(), expected_gradients.keys())
        for name in actual_gradients:
            torch.testing.assert_close(actual_gradients[name], expected_gradients[name])

    def test_cpu_fp16_autocast_with_nested_checkpoint_matches_reference(self):
        torch.manual_seed(57)
        reference = NestedCheckpointModel().train()
        actual_model = deepcopy(reference)
        reference_value = torch.randn(3, 9, 8, requires_grad=True)
        actual_value = reference_value.detach().clone().requires_grad_()
        torch.manual_seed(71)
        with torch.autocast("cpu", dtype=torch.float16):
            expected = reference(reference_value).float().square().mean()
        expected.backward()
        torch.manual_seed(71)
        with torch.autocast("cpu", dtype=torch.float16), saved_activation_context(True, actual_model):
            actual = actual_model(actual_value).float().square().mean()
        actual.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_value.grad, reference_value.grad)
        for actual_parameter, expected_parameter in zip(actual_model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)

    def test_saved_activation_hooks_preserve_gradients_across_linear_state_segments(self):
        from ScopeSFT.linear_memory import chunked_native_delta_rule
        from ScopeSFT.test_linear_memory import fixture, copies, native_fp32

        values, initial_state = fixture(145)
        expected_values, expected_initial = copies(values, initial_state)
        expected, expected_state = native_fp32(expected_values, expected_initial)
        with saved_activation_context(True):
            actual, state = chunked_native_delta_rule(
                *values, initial_state=initial_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True, sequence_chunk_size=64)
        # A late-only loss must still reach the first segment and initial state.
        actual_gradients = torch.autograd.grad(actual[:, -1].square().sum() + state.square().sum(),
                                               values + [initial_state])
        expected_gradients = torch.autograd.grad(expected[:, -1].square().sum() + expected_state.square().sum(),
                                                 expected_values + [expected_initial])
        self.assertGreater(actual_gradients[2][:, :64].abs().sum().item(), 0)
        for gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(gradient, expected_gradient, rtol=7e-4, atol=2e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "Actual saved-activation CUDA copies require a GPU")
    def test_cuda_roundtrip_and_nested_checkpoint_gradient_parity(self):
        torch.manual_seed(43)
        reference = NestedCheckpointModel().cuda().train()
        actual_model = deepcopy(reference)
        reference_value = torch.randn(3, 9, 8, device="cuda", requires_grad=True)
        actual_value = reference_value.detach().clone().requires_grad_()
        torch.manual_seed(17)
        expected = reference(reference_value).square().mean()
        expected.backward()
        torch.manual_seed(17)
        with saved_activation_context(True, actual_model) as context:
            actual = actual_model(actual_value).square().mean()
            packed = context._pack(actual_value)
            self.assertTrue(packed.offloaded)
            self.assertEqual(packed.value.device.type, "cpu")
            self.assertTrue(packed.value.is_pinned())
            torch.testing.assert_close(context._unpack(packed), actual_value)
            parameter = context._pack(actual_model.first.weight.T)
            self.assertFalse(parameter.offloaded)
            self.assertEqual(parameter.value.device.type, "cuda")
        actual.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_value.grad, reference_value.grad)
        for actual_parameter, expected_parameter in zip(actual_model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)
        self.assertGreater(context.stats["offloaded_bytes"], 0)
        self.assertGreater(context.stats["offloaded_tensors"], 0)


if __name__ == "__main__":
    unittest.main()

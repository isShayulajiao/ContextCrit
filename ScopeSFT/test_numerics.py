"""FP16 backward regressions for the native Qwen3.5 fallback; CPU only."""
from copy import deepcopy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule,
)

from ScopeSFT.model_support import lora_target_modules
from ScopeSFT.numerics import install_fp32_native_fallback, remove_fp32_native_fallback
from ScopeSFT.test_model_support import tiny_text_config
from ScopeSFT.training_forward import install_training_forward


@pytest.mark.parametrize("name", ["chunk_gated_delta_rule", "recurrent_gated_delta_rule"])
def test_small_and_zero_fp16_qk_match_fp32_native_forward_and_gradient(name):
    torch.manual_seed(71)
    model = Qwen3_5ForCausalLM(tiny_text_config())
    layer = model.model.layers[0].linear_attn
    native = getattr(layer, name)
    counts = install_fp32_native_fallback(model)
    assert counts["chunk_fallback_layers"] == counts["recurrent_fallback_layers"] == 1
    assert counts["native_only"] is True
    query = torch.randn(1, 4, 2, 8).half() * .001
    key = torch.randn_like(query) * .001
    query[:, 0] = key[:, 0] = 0
    values = torch.randn_like(query) * .1
    g = torch.full((1, 4, 2), -.3, dtype=torch.float16)
    beta = torch.full_like(g, .5)
    half_inputs = [item.detach().requires_grad_() for item in (query, key, values, g, beta)]
    reference_inputs = [item.detach().float().requires_grad_() for item in half_inputs]
    arguments = dict(initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
    expected, expected_state = native(*reference_inputs, **arguments)
    with torch.autocast("cpu", dtype=torch.float16):
        actual, state = getattr(layer, name)(*half_inputs, **arguments)
    assert actual.dtype == torch.float16
    assert state.dtype == torch.float32
    torch.testing.assert_close(actual.float(), expected, rtol=2e-3, atol=2e-5)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    actual_gradients = torch.autograd.grad(actual.float().square().sum() + state.square().sum(), half_inputs)
    reference_gradients = torch.autograd.grad(expected.square().sum() + expected_state.square().sum(), reference_inputs)
    for actual_gradient, expected_gradient in zip(actual_gradients, reference_gradients):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient.float(), expected_gradient, rtol=3e-3, atol=2e-4)


def test_install_is_idempotent_restorable_and_does_not_replace_optimized_callables():
    model = Qwen3_5ForCausalLM(tiny_text_config())
    layer = model.model.layers[0].linear_attn
    optimized = lambda *args, **kwargs: None
    layer.chunk_gated_delta_rule = optimized
    first = install_fp32_native_fallback(model)
    second = install_fp32_native_fallback(model)
    assert first == second
    assert first["chunk_fallback_layers"] == 0
    assert first["recurrent_fallback_layers"] == 1
    assert layer.chunk_gated_delta_rule is optimized
    remove_fp32_native_fallback(model)
    assert layer.chunk_gated_delta_rule is optimized
    assert layer.recurrent_gated_delta_rule is torch_recurrent_gated_delta_rule
    layer.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    install_fp32_native_fallback(model)
    remove_fp32_native_fallback(model)
    assert layer.chunk_gated_delta_rule is torch_chunk_gated_delta_rule


def test_tiny_hybrid_peft_fp16_autocast_has_finite_adapter_gradients_matching_fp32():
    torch.manual_seed(7)
    base = Qwen3_5ForCausalLM(tiny_text_config()).half()
    install_fp32_native_fallback(base)
    install_training_forward(base, ct_weight=.2, chunk_size=3)
    model = get_peft_model(base, LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0., task_type="CAUSAL_LM",
        target_modules=lora_target_modules(base),
    )).train()
    reference = deepcopy(model).float()
    # deepcopy preserves function closures, so install a fresh training forward
    # for the reference decoder rather than reusing the original model closure.
    from ScopeSFT.training_forward import remove_training_forward
    remove_training_forward(reference.get_base_model())
    install_training_forward(reference.get_base_model(), ct_weight=.2, chunk_size=3)
    for instance in (model, reference):
        instance.enable_input_require_grads()
        instance.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ids = torch.tensor([[1, 5, 9, 7, 6, 8, 3, 2], [1, 5, 9, 7, 2, 0, 0, 0]])
    labels = ids.clone()
    labels[:, :3] = -100
    labels[1, 5:] = -100
    inputs = dict(input_ids=ids, attention_mask=(ids != 0).long(), labels=labels,
                  critical_positions=torch.tensor([3, 3]), positive_ids=torch.tensor([7, 7]),
                  negative_ids=torch.tensor([10, 10]), scope_compute_loss=True, use_cache=False)
    expected = reference(**inputs).loss
    expected.backward()
    with torch.autocast("cpu", dtype=torch.float16):
        actual = model(**inputs).loss
    actual.backward()
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
    expected_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        reference_gradient = expected_parameters[name].grad
        assert reference_gradient is not None, name
        torch.testing.assert_close(parameter.grad, reference_gradient, rtol=.04, atol=2e-3, msg=name)

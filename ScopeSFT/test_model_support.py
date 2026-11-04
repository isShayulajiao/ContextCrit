"""Tiny random native-model checks; no Hub access or pretrained weights."""
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
)

from ScopeSFT.model_support import (
    QWEN35_LINEAR_TARGETS,
    QWEN_LORA_TARGETS,
    load_text_model,
    lora_target_modules,
    text_config,
    text_decoder,
    validate_context_length,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous_threads)


def tiny_text_config():
    return Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, max_position_embeddings=262144,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]},
        pad_token_id=0, eos_token_id=2, tie_word_embeddings=True,
        attention_dropout=0.0,
    )


def tiny_outer_model():
    return Qwen3_5ForConditionalGeneration(Qwen3_5Config(
        text_config=tiny_text_config().to_dict(), tie_word_embeddings=True,
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                       "num_heads": 2, "patch_size": 2, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "out_hidden_size": 32,
                       "num_position_embeddings": 16},
    ))


def inputs():
    return {"input_ids": torch.tensor([[1, 5, 9, 7, 6, 8, 3, 2]]),
            "attention_mask": torch.ones((1, 8), dtype=torch.long), "use_cache": False}


def test_outer_checkpoint_loads_exact_text_weights_without_vision(tmp_path, monkeypatch):
    outer = tiny_outer_model().eval()
    with torch.no_grad():
        expected = outer(**inputs()).logits
    outer.save_pretrained(tmp_path)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    def no_vision(*args, **kwargs):
        raise AssertionError("The text training loader constructed a vision tower")

    monkeypatch.setattr(Qwen3_5VisionModel, "__init__", no_vision)
    model, info = load_text_model(
        tmp_path, dtype=torch.float32, attention="eager", local_files_only=True, max_length=65536,
    )
    assert isinstance(model, Qwen3_5ForCausalLM)
    assert info["source_model_type"] == "qwen3_5"
    assert info["training_model_type"] == "qwen3_5_text"
    assert info["vision_tower_loaded"] is False
    assert not any("visual" in name for name, _ in model.named_parameters())
    assert text_decoder(model) is model.model
    with torch.no_grad():
        torch.testing.assert_close(model(**inputs()).logits, expected)


def test_text_config_preserves_outer_head_tying_and_is_independent():
    outer = tiny_outer_model().config
    outer.text_config.tie_word_embeddings = False
    config = text_config(outer)
    assert config.tie_word_embeddings is True
    config.max_position_embeddings = 10
    assert outer.text_config.max_position_embeddings == 262144


def test_context_length_is_checked_before_weight_loading(tmp_path, monkeypatch):
    tiny_outer_model().config.save_pretrained(tmp_path)

    def no_weights(*args, **kwargs):
        raise AssertionError("Context validation happened after weight loading")

    monkeypatch.setattr(Qwen3_5ForCausalLM, "from_pretrained", no_weights)
    with pytest.raises(ValueError, match="exceeds"):
        load_text_model(tmp_path, max_length=262145, local_files_only=True)
    assert validate_context_length(tiny_text_config(), 65536) == 262144
    for invalid in (0, 1, True, 2.5):
        with pytest.raises(ValueError, match="integer"):
            validate_context_length(tiny_text_config(), invalid)


def test_native_logits_selection_and_peft_backward_cover_both_attention_types():
    base = Qwen3_5ForCausalLM(tiny_text_config())
    targets = lora_target_modules(base)
    assert set(targets) == set(QWEN_LORA_TARGETS + QWEN35_LINEAR_TARGETS)
    model = get_peft_model(base, LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0, task_type="CAUSAL_LM", target_modules=targets,
    ))
    model.eval()
    with torch.no_grad():
        all_logits = model(**inputs()).logits
        selected_logits = model(**inputs(), logits_to_keep=3).logits
        indexed_logits = model(**inputs(), logits_to_keep=torch.tensor([0, 4, 6])).logits
    assert selected_logits.shape == (1, 3, 64)
    torch.testing.assert_close(selected_logits, all_logits[:, -3:])
    torch.testing.assert_close(indexed_logits, all_logits[:, [0, 4, 6]])
    model.train()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model(**inputs(), logits_to_keep=3).logits.square().mean().backward()
    for projection in QWEN_LORA_TARGETS + QWEN35_LINEAR_TARGETS:
        grads = [parameter.grad for name, parameter in model.named_parameters()
                 if f".{projection}.lora_B." in name]
        assert grads, projection
        assert all(value is not None and torch.isfinite(value).all() for value in grads), projection
        assert any(bool(value.abs().sum() > 0) for value in grads), projection
    assert text_decoder(model) is base.model


@pytest.mark.parametrize("save_original_format", [True, False])
def test_adapter_and_merged_export_roundtrip_from_outer_checkpoint(tmp_path, save_original_format):
    source = tmp_path / "source"
    tiny_outer_model().save_pretrained(source)
    base, _ = load_text_model(source, attention="eager", local_files_only=True)
    model = get_peft_model(base, LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=lora_target_modules(base),
    ))
    # Nonzero adapter updates exercise merge; a zero adapter could conceal errors.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(mean=0.0, std=0.01)
    model.eval()
    with torch.no_grad():
        expected = model(**inputs()).logits
    adapter = tmp_path / "adapter"
    model.save_pretrained(adapter)
    reloaded_base, _ = load_text_model(source, attention="eager", local_files_only=True)
    reloaded = PeftModel.from_pretrained(reloaded_base, adapter, local_files_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(**inputs()).logits, expected)
    merged = model.merge_and_unload(safe_merge=True).eval()
    exported = tmp_path / "exported"
    merged.save_pretrained(exported, save_original_format=save_original_format)
    config = json.loads((exported / "config.json").read_text())
    assert config["model_type"] == "qwen3_5_text"
    assert "vision_config" not in config
    state = load_file(exported / "model.safetensors")
    assert not any("visual" in key for key in state)
    prefix = "model.language_model." if save_original_format else "model."
    assert prefix + "layers.0.linear_attn.in_proj_qkv.weight" in state
    final = AutoModelForCausalLM.from_pretrained(exported, local_files_only=True,
                                               attn_implementation="eager").eval()
    assert isinstance(final, Qwen3_5ForCausalLM)
    with torch.no_grad():
        torch.testing.assert_close(final(**inputs()).logits, expected, atol=2e-6, rtol=2e-5)


def test_missing_text_weight_is_rejected(tmp_path):
    Qwen3_5ForCausalLM(tiny_text_config()).save_pretrained(tmp_path, save_original_format=False)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    state.pop("model.layers.1.self_attn.q_proj.weight")
    save_file(state, path, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="not loaded completely"):
        load_text_model(tmp_path, attention="eager", local_files_only=True)


def test_native_qwen3_remains_supported(tmp_path):
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=16, max_position_embeddings=128,
                         pad_token_id=0, eos_token_id=2, tie_word_embeddings=True)
    original = Qwen3ForCausalLM(config).eval()
    original.save_pretrained(tmp_path)
    loaded, info = load_text_model(tmp_path, attention="eager", local_files_only=True)
    assert info["training_model_type"] == "qwen3"
    assert set(lora_target_modules(loaded)) == set(QWEN_LORA_TARGETS)
    with torch.no_grad():
        torch.testing.assert_close(loaded(**inputs()).logits, original(**inputs()).logits)


def test_gpt2_is_only_available_by_explicit_test_opt_in(tmp_path):
    model = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=32, n_layer=1, n_head=2,
                                     n_positions=128, bos_token_id=1, eos_token_id=2))
    model.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="Unsupported model_type"):
        load_text_model(tmp_path, local_files_only=True)
    loaded, _ = load_text_model(tmp_path, attention="eager", local_files_only=True,
                                allow_test_models=True)
    assert text_decoder(loaded) is loaded.transformer
    assert set(lora_target_modules(loaded)) == {"c_attn", "c_proj", "c_fc"}

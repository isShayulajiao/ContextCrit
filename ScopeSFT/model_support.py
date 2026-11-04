"""Native Hugging Face model loading for text-only Scope training.

Qwen3.5 Hub checkpoints have a multimodal ``qwen3_5`` configuration.  Their
language model is the native ``Qwen3_5ForCausalLM`` (``qwen3_5_text``), not an
``AutoModelForCausalLM`` instantiated with the outer configuration.  Transformers
5.3.0 includes the checkpoint conversion from ``model.language_model.*`` to
``model.*``; using that class avoids constructing or loading the vision tower.
"""
from __future__ import annotations

from copy import deepcopy


QWEN_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
QWEN35_LINEAR_TARGETS = (
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
)
SUPPORTED_MODEL_TYPES = ("qwen3", "qwen3_5", "qwen3_5_text")


def text_config(config):
    """Return an independent decoder config, preserving pretrained semantics."""
    kind = getattr(config, "model_type", None)
    if kind == "qwen3_5":
        result = deepcopy(config.text_config)
        # The conditional-generation model ties its head using the OUTER flag.
        # Its decoder config can omit this field or contain a different default.
        result.tie_word_embeddings = config.tie_word_embeddings
        if getattr(config, "_commit_hash", None) is not None:
            result._commit_hash = config._commit_hash
        return result
    if kind in ("qwen3", "qwen3_5_text", "gpt2"):
        return deepcopy(config)
    raise ValueError(
        f"Unsupported model_type={kind!r}; expected Qwen3 or Qwen3.5. "
        "No alternate model is selected automatically."
    )


def validate_context_length(config, max_length):
    """Reject an unsupported requested context; never truncate or rescale it."""
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 2:
        raise ValueError("max_length must be an integer of at least 2 tokens")
    config = text_config(config)
    maximum = getattr(config, "max_position_embeddings", None)
    if maximum is None:
        maximum = getattr(config, "n_positions", None)
    if not isinstance(maximum, int) or maximum < 2:
        raise ValueError("The checkpoint does not declare a usable context length")
    if max_length > maximum:
        raise ValueError(
            f"Requested max_length={max_length} exceeds the checkpoint's "
            f"max_position_embeddings={maximum}. No truncation or RoPE change was applied."
        )
    return maximum


def load_text_model(model_path, *, dtype=None, attention="sdpa", local_files_only=False,
                    max_length=None, allow_test_models=False):
    """Load the requested pretrained decoder and return ``(model, metadata)``.

    The optional GPT-2 path is only for explicitly requested local engineering
    tests. It never replaces a Qwen checkpoint. All missing, mismatched, or
    unrecognized text weights are errors rather than random initialization.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, Qwen3_5ForCausalLM

    source = AutoConfig.from_pretrained(
        model_path, trust_remote_code=False, local_files_only=local_files_only,
    )
    source_type = source.model_type
    if source_type not in SUPPORTED_MODEL_TYPES and not (allow_test_models and source_type == "gpt2"):
        raise ValueError(
            f"Unsupported model_type={source_type!r}; --model must be a Qwen3 or Qwen3.5 checkpoint"
        )
    config = text_config(source)
    if max_length is not None:
        validate_context_length(config, max_length)
    loader = Qwen3_5ForCausalLM if config.model_type == "qwen3_5_text" else AutoModelForCausalLM
    kwargs = dict(
        config=config, use_safetensors=True, trust_remote_code=False,
        attn_implementation=attention, local_files_only=local_files_only,
        output_loading_info=True,
    )
    if dtype is not None:
        kwargs["dtype"] = dtype
    model, loading_info = loader.from_pretrained(model_path, **kwargs)
    # The native Qwen3.5 class explicitly ignores only visual/MTP checkpoint
    # tensors. Anything remaining here indicates an incompatible text model.
    failures = {key: value for key, value in loading_info.items() if value}
    if failures:
        raise ValueError(f"Checkpoint text weights were not loaded completely: {failures}")
    info = {
        "source_model_type": source_type,
        "training_model_type": model.config.model_type,
        "model_class": type(model).__name__,
        "base_revision": getattr(source, "_commit_hash", None),
        "vision_tower_loaded": False,
        "text_only_export": model.config.model_type == "qwen3_5_text",
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
    }
    return model, info


def causal_model(model):
    """Unwrap DDP/PEFT without modifying the installed adapter modules."""
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    return model


def text_decoder(model):
    """Get the native decoder for a loss that projects hidden states in chunks."""
    model = causal_model(model)
    kind = model.config.model_type
    if kind in ("qwen3", "qwen3_5_text"):
        return model.model
    if kind == "gpt2":
        return model.transformer
    raise ValueError(f"Expected a text-only CausalLM, found model_type={kind!r}")


def lora_target_modules(model):
    """Select every existing attention/MLP projection, including GatedDeltaNet."""
    import torch

    model = causal_model(model)
    kind = model.config.model_type
    if kind == "gpt2":
        from transformers.pytorch_utils import Conv1D
        choices, linear_classes = ("c_attn", "c_proj", "c_fc"), (Conv1D,)
    elif kind in ("qwen3", "qwen3_5_text"):
        choices = QWEN_LORA_TARGETS + (QWEN35_LINEAR_TARGETS if kind == "qwen3_5_text" else ())
        linear_classes = (torch.nn.Linear,)
    else:
        raise ValueError(f"Unsupported LoRA architecture: {kind!r}")
    existing = {
        name.rsplit(".", 1)[-1] for name, module in model.named_modules()
        if isinstance(module, linear_classes)
    }
    targets = [name for name in choices if name in existing]
    required = {"gate_proj", "up_proj", "down_proj"} if kind != "gpt2" else {"c_attn", "c_proj", "c_fc"}
    layer_types = getattr(model.config, "layer_types", ()) or ()
    if kind == "qwen3" or "full_attention" in layer_types:
        required.update(("q_proj", "k_proj", "v_proj", "o_proj"))
    if kind == "qwen3_5_text" and "linear_attention" in layer_types:
        required.update(QWEN35_LINEAR_TARGETS)
    missing = required.difference(targets)
    if missing:
        raise ValueError(f"Missing expected LoRA projection modules: {sorted(missing)}")
    if not targets:
        raise ValueError("No compatible LoRA projection modules were found")
    return targets

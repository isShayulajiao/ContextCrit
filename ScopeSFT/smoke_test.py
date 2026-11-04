#!/usr/bin/env python3
"""CPU engineering smoke with tiny local Qwen3; never claims 1.7B/GPU training.

Run from the checkout with the training dependencies installed:
    python -m ScopeSFT.smoke_test
This creates only a temporary directory unless --work-dir is supplied.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from ScopeSFT.data import load_records
from ScopeSFT.test_tokenization import make_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def synthetic_sample():
    sample = deepcopy(json.loads((ROOT / "ScopeCrit/examples/merged_demo.json").read_text(encoding="utf-8"))[0])
    text = "《示例小说》只登了两段，第一第二段的描写法很好。"
    sample["document"]["text"] = text
    sample["document"]["evidence"] = [{"evidence_id": "E001", "text": text, "start": 0, "end": len(text)}]
    sample["gold"] = {
        "claims": [{"claim_id": "C1", "text": "评论者肯定第一、二段的描写法。", "aspect": "NARRATION",
                    "stance": "POSITIVE", "target_scope": "第一、二段", "evidence_ids": ["E001"]}],
        "required_evidence_ids": ["E001"], "reference_answer": "评论者肯定前两段描写[E001]。",
    }
    sample["split"] = "train"
    return sample


def run_smoke(directory):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / "synthetic_train.json"
    data_path.write_text(json.dumps([synthetic_sample()], ensure_ascii=False), encoding="utf-8")
    records = load_records(str(data_path))
    assert any(record.get("rejected") for record in records), "Smoke needs an actual source-rule pair"
    corpus = ["system user assistant"]
    for record in records:
        corpus.extend(message["content"] for message in record["messages"])
        corpus.append(record["chosen"])
        if record.get("rejected"):
            corpus.append(record["rejected"])
    tokenizer = make_tokenizer(corpus)
    base_path = directory / "tiny_base"
    base_path.mkdir(exist_ok=True)
    tokenizer.save_pretrained(base_path)
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=16, max_position_embeddings=32768,
                         sliding_window=None, use_sliding_window=False,
                         bos_token_id=None, eos_token_id=tokenizer.eos_token_id,
                         pad_token_id=tokenizer.pad_token_id, attention_dropout=0.0,
                         tie_word_embeddings=True)
    tiny = Qwen3ForCausalLM(config)
    tiny.generation_config.do_sample = True
    tiny.generation_config.temperature = 0.6
    tiny.generation_config.top_p = 0.95
    tiny.generation_config.top_k = 20
    tiny.save_pretrained(base_path, safe_serialization=True)
    summaries = {}
    for folder, mode in (("ScopeSFT", "sft"), ("ScopeCT", "ct")):
        output = directory / mode
        command = [sys.executable, str(ROOT / folder / "train.py"), "--cpu", "--data", str(data_path),
                   "--model", str(base_path), "--output", str(output), "--max-steps", "1",
                   "--batch-size", str(len(records)), "--grad-accum", "1", "--max-length", "30000",
                   "--attention", "eager"]
        environment = dict(os.environ, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
        (directory / f"{mode}.log").write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"{mode} training failed ({completed.returncode}):\n{completed.stdout}\n{completed.stderr}")
        manifest = json.loads((output / "training_manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "completed", manifest
        assert manifest["mode"] == mode, manifest
        loss = manifest["training_result"]["train_loss"]
        assert math.isfinite(loss), manifest
        if mode == "ct":
            assert manifest["valid_pairs"] > 0, manifest
            assert manifest["last_loss_components"]["ct_pairs"] > 0, manifest
            assert manifest["last_loss_components"]["ct_loss"] > 0, manifest
        adapter_state = load_file(str(output / "adapter" / "adapter_model.safetensors"))
        lora_b = [value for key, value in adapter_state.items() if "lora_B" in key]
        assert lora_b and any(bool(value.abs().sum() > 0) for value in lora_b), "LoRA B was not updated"
        reloaded_tokenizer = AutoTokenizer.from_pretrained(output / "final_model", local_files_only=True)
        assert len(reloaded_tokenizer) == len(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(base_path, local_files_only=True, attn_implementation="eager")
        adapter_model = PeftModel.from_pretrained(model, output / "adapter", local_files_only=True).eval()
        merged_model = AutoModelForCausalLM.from_pretrained(output / "final_model", local_files_only=True,
                                                           attn_implementation="eager").eval()
        inputs = tokenizer("system user assistant 描写", return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            expected = adapter_model(**inputs).logits
            actual = merged_model(**inputs).logits
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        summaries[mode] = {"train_loss": loss, "lora_updated": True,
                           "merged_logits_match_adapter": True,
                           "valid_pairs": manifest.get("valid_pairs", 0)}
    print(json.dumps({"scope": "tiny local Qwen3 CPU only; not 1.7B/four-GPU validation",
                      "results": summaries}, ensure_ascii=False, indent=2))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", help="保留合成样本、微型权重、训练日志；默认使用临时目录")
    args = parser.parse_args()
    if args.work_dir:
        run_smoke(args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="scope-training-smoke-") as directory:
            run_smoke(directory)


if __name__ == "__main__":
    main()

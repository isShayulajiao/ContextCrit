"""Shared training runtime. SFT and ScopeCT differ only in the joint loss weight."""
from __future__ import annotations

from collections import Counter
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import warnings


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DiskDataset:
    """Read only one prepared trajectory per access; no full-corpus tensor cache."""
    def __init__(self, root):
        self.path = Path(root) / "prepared.jsonl"
        self.offsets = json.loads((Path(root) / "prepared_offsets.json").read_text())

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            return json.loads(stream.readline())


def _prepare(args, mode, tokenizer=None):
    from ScopeSFT.data import iter_records
    from ScopeSFT.tokenization import encode_record
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    stats = {"actions": 0, "total_pairs": 0, "valid_pairs": 0,
             "task_actions": Counter(), "action_counts": Counter(), "pair_status": Counter()}
    samples, offsets = set(), []
    selection = {}
    maximum = 0
    with (root / "prepared.jsonl.tmp").open("wb") as prepared, \
            (root / "data_audit.jsonl.tmp").open("w", encoding="utf-8") as audit_file, \
            (root / "training_pairs.jsonl.tmp").open("w", encoding="utf-8") as pairs:
        for record in iter_records(args.data, max_samples=getattr(args, "max_samples", None), selection=selection):
            meta = record["metadata"]
            samples.add(meta["sample_id"])
            stats["actions"] += 1
            stats["task_actions"][meta["task"]] += 1
            stats["action_counts"][meta["action"]] += 1
            stats["total_pairs"] += bool(record.get("rejected"))
            if tokenizer is not None:
                encoded, audit = encode_record(tokenizer, record, args.max_length)
                maximum = max(maximum, len(encoded["input_ids"]))
                stats["valid_pairs"] += encoded["critical_positions"] >= 0
                if mode == "sft":
                    encoded.update(critical_positions=-1, positive_ids=-1, negative_ids=-1)
                offsets.append(prepared.tell())
                prepared.write((json.dumps(encoded, separators=(",", ":")) + "\n").encode())
            else:
                audit = {**meta, "pair_status": "not_tokenized"}
            stats["pair_status"][audit["pair_status"]] += 1
            audit_file.write(json.dumps(audit, ensure_ascii=False) + "\n")
            if record.get("rejected"):
                pairs.write(json.dumps({"chosen": record["chosen"], "rejected": record["rejected"],
                                        "semantic_span": record["semantic_span"], "audit": audit}, ensure_ascii=False) + "\n")
            if stats["actions"] % 1000 == 0:
                print(f"已准备 {len(samples)} 条任务 / {stats['actions']} 条动作", flush=True)
    if stats["actions"] == 0:
        raise ValueError("没有有效训练动作")
    for filename in ("prepared.jsonl", "data_audit.jsonl", "training_pairs.jsonl"):
        (root / (filename + ".tmp")).replace(root / filename)
    stats.update(samples=len(samples), max_tokens=maximum, dataset_sha256=sha256(args.data),
                 selection=selection)
    write_json(root / "prepared_offsets.json", offsets)
    write_json(root / "data_summary.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    if mode == "ct" and tokenizer is not None and stats["valid_pairs"] == 0 and not args.allow_sft_fallback:
        raise ValueError("没有通过共同 token 前缀和语义字段检查的 ScopeCT 训练对。"
                         "请检查 training_pairs.jsonl，补充文档规定的审核负例；"
                         "若只希望执行 SFT，使用 ScopeSFT 或显式 --allow-sft-fallback。")
    return stats


def _hardware(args):
    import torch
    if args.cpu:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not torch.distributed.is_initialized():
            torch.distributed.init_process_group("gloo", timeout=timedelta(hours=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。请先运行 install.sh 安装 CUDA 11.8 版 torch，检查 NVIDIA 驱动和 --gpus。")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"选择的 GPU 不存在：local_rank={local_rank}，可见 GPU={torch.cuda.device_count()}")
    torch.cuda.set_device(local_rank)
    # TITAN RTX: FP16, no BF16/FlashAttention. SDPA may use its memory-efficient
    # backend; disabling flash does not disable the memory-efficient backend.
    torch.backends.cuda.enable_flash_sdp(False)
    try:
        a = torch.randn((32, 32), device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = (a @ a).float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(a.grad).all():
            raise RuntimeError("CUDA 预检产生非有限梯度")
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group("nccl", timeout=timedelta(hours=2))
            value = torch.ones(1, device="cuda")
            torch.distributed.all_reduce(value)
            if int(value.item()) != torch.distributed.get_world_size():
                raise RuntimeError("NCCL all_reduce 结果错误")
    except RuntimeError as exc:
        raise RuntimeError("CUDA/NCCL 预检失败，尚未下载训练权重。470 驱动处于 CUDA 11.x 次版本兼容路径，"
                           "若报告 PTX/driver 错误请升级驱动；不要关闭校验或直接重跑。原始错误：" + str(exc)) from exc
    print(f"GPU local_rank={local_rank}: {torch.cuda.get_device_name(local_rank)}; "
          f"torch={torch.__version__}, CUDA runtime={torch.version.cuda}; 预检通过", flush=True)


def trainer_class():
    import torch
    from transformers import Trainer
    from ScopeCT.objective import scope_loss

    class ActionTrainer(Trainer):
        def __init__(self, *args, ct_weight=0.0, margin=1.0, low_memory=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.ct_weight, self.margin = ct_weight, margin
            self.low_memory = low_memory
            self.model_accepts_loss_kwargs = False
            self.last_loss_components = {}

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            if getattr(self, "low_memory", False):
                # Enter the wrapped model normally so DDP prepares its reducer
                # and gradient accumulation/no_sync retain their semantics.
                outputs = model(**inputs, scope_compute_loss=True, use_cache=False)
                loss, metrics = outputs.loss, outputs.scope_metrics
                if not torch.isfinite(loss):
                    raise RuntimeError("训练损失非有限，停止而不导出成功权重")
                self.last_loss_components = {k: float(v.item()) for k, v in metrics.items()}
                return (loss, outputs) if return_outputs else loss
            # Qwen3's official logits_to_keep avoids allocating vocabulary logits
            # for the system/prompt. Keep the row preceding the first target.
            start = int((labels != -100).nonzero()[:, 1].min().item()) - 1
            if start < 0:
                raise ValueError("每条轨迹必须具有非空 prompt")
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                            logits_to_keep=labels.shape[1] - start, use_cache=False)
            positions = inputs["critical_positions"]
            positions = torch.where(positions >= 0, positions - start, positions)
            loss, metrics = scope_loss(outputs.logits, labels[:, start:], positions,
                                       inputs["positive_ids"], inputs["negative_ids"],
                                       ct_weight=self.ct_weight, margin=self.margin)
            if not torch.isfinite(loss):
                raise RuntimeError("训练损失非有限，停止而不导出成功权重")
            self.last_loss_components = {k: float(v.item()) for k, v in metrics.items()}
            return (loss, outputs) if return_outputs else loss

        def log(self, logs, start_time=None):
            if "loss" in logs:
                logs.update({"last_microbatch_" + k: v for k, v in self.last_loss_components.items()})
            return super().log(logs, start_time)

    return ActionTrainer


def run(args, mode):
    root = Path(args.output)
    if args.validate_only:
        _prepare(args, mode)
        print(f"原始数据与轨迹检查完成（未下载模型、未训练）：{root}")
        return
    import torch
    from transformers import AutoTokenizer, GenerationConfig, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    from safetensors import safe_open
    from ScopeSFT.tokenization import Collator, configure_tokenizer
    from ScopeSFT.model_support import load_text_model, lora_target_modules
    from ScopeSFT.training_forward import install_training_forward, remove_training_forward
    from ScopeSFT.numerics import install_fp32_native_fallback, remove_fp32_native_fallback

    if not args.dry_run:
        _hardware(args)
    rank = int(os.environ.get("RANK", "0"))
    distributed = torch.distributed.is_initialized()
    set_seed(args.seed)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=False)
        configure_tokenizer(tokenizer)
        stats = _prepare(args, mode, tokenizer)
        tokenizer.save_pretrained(root / "tokenizer")
    if distributed:
        torch.distributed.barrier()
    if rank != 0:
        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", use_fast=True, local_files_only=True)
        configure_tokenizer(tokenizer)
        stats = json.loads((root / "data_summary.json").read_text())
    if args.dry_run:
        print(f"Token 与长度检查完成（未训练）：{root}")
        return
    if stats["actions"] < int(os.environ.get("WORLD_SIZE", "1")) * args.batch_size * args.grad_accum:
        warnings.warn("训练动作数小于一个有效 batch；这是工程短跑，不足以证明训练效果。")
    low_memory = getattr(args, "low_memory", "auto") != "off"
    precision = getattr(args, "dtype", "auto")
    if precision == "auto":
        precision = "fp32" if args.cpu else "fp16"
    if args.cpu and precision == "fp16":
        raise ValueError("CPU 工程测试请使用 --dtype fp32；FP16 训练用于 CUDA")
    attention = args.attention
    if low_memory:
        from ScopeSFT.attention import register_chunked_attention
        attention = register_chunked_attention(getattr(args, "attention_chunk_size", 128))
    model, model_info = load_text_model(
        args.model, dtype=torch.float32 if precision == "fp32" else torch.float16,
        attention=attention, max_length=args.max_length)
    revision = model_info["base_revision"]
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    numeric_stability = install_fp32_native_fallback(model)
    targets = lora_target_modules(model)
    if low_memory:
        install_training_forward(model, ct_weight=args.ct_weight if mode == "ct" else 0.0,
                                 margin=args.margin, chunk_size=getattr(args, "loss_chunk_size", 128))
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=targets))
    model.enable_input_require_grads()
    if rank == 0:
        model.print_trainable_parameters()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    updates_per_epoch = math.ceil(math.ceil(stats["actions"] / world / args.batch_size) / args.grad_accum)
    planned_updates = args.max_steps if args.max_steps > 0 else math.ceil(updates_per_epoch * args.epochs)
    training_args = TrainingArguments(
        output_dir=str(root / "checkpoints"), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate, lr_scheduler_type="cosine",
        warmup_ratio=0.0 if planned_updates < 10 else 0.03,
        max_grad_norm=1.0, weight_decay=0.01, optim="adamw_torch",
        fp16=precision == "fp16", bf16=False, use_cpu=args.cpu,
        gradient_checkpointing=getattr(args, "gradient_checkpointing", "on") == "on",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="steps", save_steps=200, save_total_limit=2,
        dataloader_num_workers=0, dataloader_pin_memory=not args.cpu,
        remove_unused_columns=False, ddp_find_unused_parameters=False, ddp_timeout=7200,
        report_to="none", seed=args.seed, data_seed=args.seed)
    manifest = {"status": "training", "mode": mode, **stats, "arguments": vars(args),
                "base_model": args.model, "base_revision": revision,
                "model_support": model_info, "lora_target_modules": targets,
                "numeric_stability": numeric_stability,
                "resolved_dtype": precision, "low_memory": low_memory,
                "resolved_attention": attention,
                "torch": torch.__version__, "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "objective": "per-example response NLL" + (" + lambda*softplus(m+z_neg-z_pos)" if mode == "ct" else ""),
                "independent_model_judging": False,
                "code_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])): sha256(p)
                                for folder in ("ScopeSFT", "ScopeCT")
                                for p in (Path(__file__).resolve().parents[1] / folder).glob("*.py")}}
    if rank == 0:
        write_json(root / "training_manifest.json", manifest)
    trainer = trainer_class()(model=model, args=training_args, train_dataset=DiskDataset(root),
                              data_collator=Collator(tokenizer.pad_token_id), processing_class=tokenizer,
                              ct_weight=args.ct_weight if mode == "ct" else 0.0, margin=args.margin,
                              low_memory=low_memory)
    result = trainer.train()
    if distributed:
        torch.distributed.barrier()
    if rank == 0:
        base = trainer.accelerator.unwrap_model(trainer.model)
        if not any(bool(param.detach().abs().sum() > 0)
                   for name, param in base.named_parameters() if "lora_B" in name):
            raise RuntimeError("LoRA B 权重没有更新；请检查步数、学习率或混合精度跳步。未导出成功权重。")
        base.save_pretrained(root / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(root / "adapter")
        # Merge on CPU to avoid an export-time GPU allocation spike.
        base = base.to("cpu")
        merged = base.merge_and_unload(safe_merge=True)
        remove_training_forward(merged)
        remove_fp32_native_fallback(merged)
        merged.config.use_cache = True
        # The custom backend is training-only; keep exports usable by ordinary
        # AutoModelForCausalLM without importing or registering our backend.
        merged.config._attn_implementation = "sdpa"
        # Official Qwen checkpoints carry sampling defaults (temperature/top_p).
        # Reset together: changing only do_sample=False can make HF reject saving.
        merged.generation_config = GenerationConfig.from_model_config(merged.config)
        merged.save_pretrained(root / "final_model", safe_serialization=True, max_shard_size="2GB",
                               save_original_format=False)
        tokenizer.save_pretrained(root / "final_model")
        files = sorted((root / "final_model").glob("*.safetensors"))
        if not files:
            raise RuntimeError("导出未产生 safetensors 权重")
        tensor_count = 0
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                tensor_count += len(handle.keys())
        if tensor_count < 10:
            raise RuntimeError("导出权重张量不完整")
        manifest.update(status="completed", training_result=result.metrics,
                        last_loss_components=trainer.last_loss_components,
                        exported_tensors=tensor_count, final_weight_sha256={p.name: sha256(p) for p in files})
        write_json(root / "training_manifest.json", manifest)
        print(f"训练完成。可直接加载的完整权重：{root / 'final_model'}\nLoRA adapter：{root / 'adapter'}", flush=True)
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

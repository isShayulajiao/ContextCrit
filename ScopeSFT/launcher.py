"""Standard-library-only launcher: prompt once, then spawn one process per GPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid


def gpu_ids(value: str) -> list[int]:
    parts = value.split(",") if "," in value else list(value)
    if not parts or any(not p.strip().isdigit() for p in parts):
        raise argparse.ArgumentTypeError("GPU 编号使用 0123 或 0,1,2,3；编号 >=10 时使用逗号。")
    ids = [int(p.strip()) for p in parts]
    if len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("GPU 编号不能重复。")
    return ids


def sample_limit(value: str) -> int | None:
    if value.lower() == "none":
        return None
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max-samples 必须为正整数或 None") from exc
    if limit < 1:
        raise argparse.ArgumentTypeError("max-samples 必须为正整数或 None")
    return limit


def parser(mode):
    p = argparse.ArgumentParser(description=f"Qwen3 / Qwen3.5 {'Scope-CT' if mode == 'ct' else 'LoRA SFT'}")
    p.add_argument("--data", help="训练 JSON 的绝对路径；省略时交互输入一次")
    p.add_argument("--gpus", type=gpu_ids, default=gpu_ids("0123"), help="默认 0123；例如 02 或 0,2")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B", help="官方模型 ID 或本地模型目录")
    p.add_argument("--output", help="新建输出目录；默认本方法 outputs 下的唯一目录")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1, help="诊断短跑用；正值覆盖 epochs")
    p.add_argument("--max-length", type=int, default=8192, help="超长样本报错，绝不截掉证据/答案")
    p.add_argument("--max-samples", type=sample_limit, default=None,
                   help="只选原文件前 N 条任务样本（保留全部动作/轮次）；默认 None 使用全部")
    p.add_argument("--batch-size", type=int, default=1, help="每卡 microbatch")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--ct-weight", type=float, default=0.2)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    p.add_argument("--loss-chunk-size", type=int, default=128, help="每次计算词表 loss 的响应 token 数")
    p.add_argument("--attention-chunk-size", type=int, default=128, help="低显存 attention 的 query 分块大小")
    p.add_argument("--low-memory", choices=["auto", "on", "off"], default="auto", help="低显存训练路径，默认 auto")
    p.add_argument("--dtype", choices=["auto", "fp16", "fp32"], default="auto", help="默认 GPU fp16 / CPU fp32")
    p.add_argument("--gradient-checkpointing", choices=["on", "off"], default="on", help="默认启用梯度检查点")
    p.add_argument("--allow-sft-fallback", action="store_true", help="仅 ScopeCT：零有效分歧对时明确允许退回 SFT")
    p.add_argument("--validate-only", action="store_true", help="只验证原始数据与教师轨迹，不下载模型")
    p.add_argument("--dry-run", action="store_true", help="下载/读取 tokenizer，检查长度和 token 分歧，不训练")
    p.add_argument("--cpu", action="store_true", help="仅供小模型 CPU 工程测试")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def launch(mode):
    p = parser(mode)
    args = p.parse_args()
    if not args.data:
        if args.worker:
            p.error("worker 必须接收 --data")
        try:
            args.data = input("请输入训练数据 JSON 的绝对路径：").strip().strip("\"'")
        except EOFError:
            p.error("非交互运行必须提供 --data /绝对路径/train.json")
    data = Path(args.data).expanduser()
    if not data.is_absolute() or not data.is_file():
        p.error("--data 必须是存在的文件的绝对路径。")
    for key in ("epochs", "learning_rate"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f"{key} 必须为有限正数")
    for key in ("batch_size", "grad_accum", "lora_r", "loss_chunk_size", "attention_chunk_size"):
        if getattr(args, key) < 1:
            p.error(f"{key} 必须 >=1")
    if args.max_length < 32 or args.max_steps == 0 or args.max_steps < -1:
        p.error("max-length 必须 >=32；max-steps 为 -1 或正整数")
    if not math.isfinite(args.ct_weight) or args.ct_weight < 0 or not math.isfinite(args.margin) or args.margin < 0:
        p.error("ct-weight 和 margin 必须为有限非负数")
    if mode == "ct" and args.ct_weight == 0 and not args.allow_sft_fallback:
        p.error("ScopeCT 的 ct-weight 必须 >0；退回 SFT 需显式允许")
    if args.output:
        args.output = str(Path(args.output).expanduser().resolve())
    else:
        folder = "ScopeCT" if mode == "ct" else "ScopeSFT"
        run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        args.output = str(Path(__file__).resolve().parents[1] / folder / "outputs" / run)
    if not args.worker and Path(args.output).exists() and any(Path(args.output).iterdir()):
        p.error("输出目录非空；请使用新目录，防止覆盖权重或混合实验。")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if not args.worker:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))
        # No model/GPU initialization for offline preparation.
        if not (args.cpu or args.validate_only or args.dry_run):
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       f"--nproc_per_node={len(args.gpus)}", str(Path(sys.argv[0]).resolve()),
                       "--worker", "--data", str(data), "--output", args.output]
            # Rebuild normalized arguments rather than forwarding the interactive input.
            for key in ("model", "epochs", "max_steps", "max_length", "batch_size", "grad_accum",
                        "learning_rate", "seed", "lora_r", "ct_weight", "margin", "attention",
                        "loss_chunk_size", "attention_chunk_size", "low_memory", "dtype",
                        "gradient_checkpointing"):
                command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
            if args.max_samples is not None:
                command.extend(["--max-samples", str(args.max_samples)])
            command.extend(["--gpus", ",".join(map(str, args.gpus))])
            if args.allow_sft_fallback:
                command.append("--allow-sft-fallback")
            raise SystemExit(subprocess.call(command))
    from ScopeSFT.runtime import run
    try:
        run(args, mode)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"训练未完成：{exc}", file=sys.stderr)
        if "out of memory" in str(exc).lower():
            print("显存不足：保持 --batch-size 1 和 --gradient-checkpointing on，尝试 "
                  "--low-memory on --loss-chunk-size 64 --attention-chunk-size 64（必要时 32）。"
                  "程序没有自动裁短样本或丢弃监督；四卡 DDP 仍受每卡显存限制。", file=sys.stderr)
        raise SystemExit(1) from exc

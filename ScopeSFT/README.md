# ScopeSFT：Qwen3-1.7B 的 LoRA 监督训练

直接读取 CultCrit 的 JSON 数组，不要求手动制作 messages。T1–T4 的 gold 主张、参考答案、逐轮记忆和通信标签会通过现有 ScopeCrit 引擎转换为合法的 ACTION 教师轨迹，再进行 **response-only causal NLL + LoRA** 训练。输入提示不包含 gold 字段；标签只用于生成教师动作。训练仅更新 ACTION 策略，不训练 SUPPORT_CHECK / SCOPE_COMPARE 模型。

`ScopeSFT` 与 `ScopeCT` 共用基础设施，请保留完整仓库。两个实验分别从 **`Qwen/Qwen3-1.7B` 原始 checkpoint** 开始，使用相同正例、随机种子和预算。这里“base 模型”指实验起点；官方模型 ID 就是 `Qwen/Qwen3-1.7B`，并非另一个名为 `*-Base` 的模型。提示使用 `enable_thinking=False`。

## 安装一次

Linux、Python 3.10–3.12（推荐 3.11）、可访问 PyTorch/PyPI/Hugging Face 的网络。建议至少 16 GB 主机内存，并为模型缓存、训练缓存和两套最终权重预留 30 GB 以上磁盘；大型数据集的动作缓存可能需要更多空间。

```bash
cd /你的路径/CultCrit-Bench
bash ScopeSFT/install.sh
source ~/.venvs/scope-qwen3/bin/activate
```

脚本创建独立虚拟环境，安装 PyTorch **2.5.1 CUDA 11.8**、Transformers **4.51.3**、PEFT **0.15.2**、Accelerate **1.6.0**。无需另装系统 CUDA Toolkit。若需要指定虚拟环境目录：`bash ScopeSFT/install.sh /绝对路径/venv`。首次运行会下载公开的官方模型，通常无需 HF token；离线机器可用 `--model /绝对路径/本地Qwen3模型`。

### 四张 TITAN RTX / 驱动 470.182.03

默认 FP16、每卡 batch 1、LoRA rank 16、gradient checkpointing、四进程 DDP；有效 batch 为 `1 × 4卡 × 4累积 = 16` 条 ACTION。四张卡各自保存一份模型，显存不是自动合并成一个 96 GB 设备。禁用 BF16、FlashAttention、量化插件和编译内核。默认 PyTorch SDPA 的非 flash 后端；可通过 `--attention eager` 切换标准注意力。

`nvidia-smi` 的 CUDA 11.4 是驱动报告的兼容版本，不是 Python 内的 CUDA runtime。470 驱动在 CUDA 11.x 次版本兼容范围内，但仍可能遇到 PTX、新驱动特性或 NCCL 限制。程序在下载权重前执行真实 CUDA FP16 矩阵反向和多卡 NCCL all-reduce；失败时保留具体错误并停止。若确为 driver/PTX 不兼容，需要升级驱动。当前开发环境无这些 GPU，未声称已在该硬件实测成功。

## 训练：只输入数据绝对路径

```bash
python ScopeSFT/train.py
# 按提示输入 /data/train_optimized_validated.json
```

默认使用 `0123` 四卡，输出自动创建到 `ScopeSFT/outputs/<唯一运行目录>/`。也可无交互运行：

```bash
python ScopeSFT/train.py --data /data/train_optimized_validated.json --gpus 0123
python ScopeSFT/train.py --data /data/train_optimized_validated.json --gpus 02
python ScopeSFT/train.py --data /data/train_optimized_validated.json --gpus 0,2 --output /data/weights/sft_run1
```

不要再外层套一次 `torchrun`；入口已经自动启动进程。GPU 编号指所选物理设备；`0123` 等价于 `0,1,2,3`，编号大于 9 时使用逗号。输出目录必须是新的或空的，防止覆盖之前的实验。

默认 `3 epochs`、学习率 `2e-4`、rank `16`、梯度累积 `4`、最长序列 `8192`。例如：

```bash
python ScopeSFT/train.py --data /data/train.json --gpus 0123 \
  --epochs 3 --learning-rate 0.0002 --grad-accum 4 --max-length 8192
```

序列长度包括协议 schema、合法观察和动作答案。**绝不静默截断正文、未来证据约束或标签**；超长时会报告样本与实际 token 数。可在显存允许时增大 `--max-length`。若显存不足，先保持 batch 1、缩短合法训练材料或调整长度；减小有效 batch 的 `--grad-accum` 不会显著降低单步激活显存。eager 长序列的注意力显存增长很快。

## 数据与检查

输入是解压后的 **UTF-8 JSON 数组**，不是 RAR/ZIP 文件。每条至少保留原数据的 `sample_id/doc_id/task/question/document/gold/events/roles/required_exchanges`，其中 `document` 有带偏移的证据，`gold` 有带证据引用的主张和参考答案。直接兼容用户提供样本的结构，不需要改成问答对或偏好对。

- T1/T2 只观察当前合法文档；T3 严格逐轮公开；T4 通过真实引擎投递共享证据。
- 使用 gold 生成的是确定性离线教师轨迹，不是在线模型成功轨迹。支持核对沿用训练标签，未调用独立裁判；不证明标签已经人工核验。
- T2 的教师比较阶段仅依据当前已提交主张描述范围，关系类型使用 `UNRESOLVED`，不伪造额外关系标签或提前提供最终参考答案；该阶段本身不是训练目标。
- 数据结构、证据偏移、引用、重复 ID、同文档材料冲突或无法合法完成的教师轨迹会直接报错，不悄悄跳过。
- 明确指定的文件就是训练集，不自动重新划分；若带 `split=test`，会警告，训练后的相同样本不能用于独立测试。请使用正式训练集。
- 大文件按条流式读取，动作以磁盘索引缓存，避免每个 GPU 进程重复持有扩展后的全量数据。

准备训练前可分别检查，不会启动付费 API：

```bash
# 不下载模型；检查原始数据和引擎轨迹
python ScopeSFT/train.py --data /data/train.json --validate-only
# 只读取 tokenizer，检查真实 token 长度与分歧位置，不下载模型权重
python ScopeSFT/train.py --data /data/train.json --dry-run
```

检查生成的目录只存检查结果，不含训练完成权重。正式训练使用新目录。

## 最终权重与审计

训练成功会打印路径并生成：

```text
<运行目录>/
  final_model/              # 合并 LoRA 后完整 safetensors 权重、config、tokenizer
  adapter/                  # 独立 LoRA adapter
  checkpoints/              # 中间训练检查点（每 200 步，最多保留 2 个）
  training_manifest.json    # 完成状态、数据/代码哈希、超参数、实际损失和权重校验和
  data_summary.json         # 原始任务数、动作数、分歧对与长度统计
  data_audit.jsonl          # 每条轨迹与分歧检查状态
  training_pairs.jsonl      # 正负例、依据及审核来源；SFT 不使用附加 CT 损失
  prepared.jsonl           # 训练 token 磁盘缓存
```

只有 `training_manifest.json` 的 `status=completed` 才表示训练及导出均成功。`final_model` 可直接通过 `AutoModelForCausalLM.from_pretrained(绝对路径, torch_dtype=torch.float16)` 加载，无需再手动合并。推理继续使用 tokenizer 的 `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`。训练输出、用户数据、缓存和权重不提交 GitHub。

## 已执行的工程验证

```bash
python -m pytest -q ScopeSFT ScopeCT
python -m ScopeSFT.smoke_test
```

测试包括精确 token 分歧、response mask、因果位置、损失/梯度、正例 logits 裁剪等价性；烟雾测试使用本地生成的微型 Qwen3，完成真实训练、确认 LoRA 权重更新、合并和重载，并核对合并模型与 adapter 输出一致。还验证了两进程 CPU/Gloo 的训练及导出流程。这些检查不等于 1.7B 四卡训练或下游文学能力验证。

用户压缩训练集只抽取前四条 T1–T4 样本检查，得到 58 条合法 ACTION 轨迹，没有逐条读取完整训练集；全量文件的最终检查发生在用户实际运行时。

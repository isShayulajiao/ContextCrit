# Qwen3.5 适配验证记录

日期：2026-11-05。适用本次 `ScopeSFT` / `ScopeCT` 适配；不涉及修改 `ScopeCrit` 的评测逻辑。

## 执行环境与结果

开发环境只有 CPU，Python 3.11、PyTorch 2.5.1+cpu、Transformers 5.3.0、PEFT 0.18.1。GPU 安装脚本采用相同 Torch 版本的 CUDA 11.8 构建。

| 实际执行 | 结果 |
| --- | --- |
| `python -m pytest ScopeSFT ScopeCT -q` | 91 passed，218 subtests passed |
| `python -m ScopeSFT.smoke_test --work-dir /tmp/q35_release_smoke` | 5 组完整训练、LoRA 更新、合并重载和缓存生成成功 |
| 两进程 `torch.distributed.run` + ScopeCT Trainer | 两步训练与导出成功，manifest 为 completed、world_size 为 2 |
| Python 编译、两个 install.sh 的 shell 语法、`pip check` | 通过 |

完整 pytest 包含真实两进程 GLOO 测试：两次 microbatch、首次 no_sync 累积后归约，两卡的全部 LoRA 梯度均与单进程合并批次参考结果一致。执行该项时已允许本机网络访问；禁止 loopback socket 的环境会明确 skip，不能将 skip 当成通过。

## 检查内容

- 原生 Qwen3.5 多模态 checkpoint 转为文本 CausalLM，确认不构造视觉编码器；缺失正文模型权重报错。
- 常规注意力、线性注意力和 MLP 的 LoRA 投影均有梯度；保持 Qwen3 和非思考 Instruct-2507 模板兼容。
- 完整因果注意力按 query 分块，输出、梯度、padding、GQA、dropout 与原公式一致；包含 FP16、head_dim=256、强制 SDPA math 后端回归。
- 词表投影按响应 token 分块并重计算，保留全部监督与 EOS；SFT 按每条响应 token 均值后再按样本平均，CT 按全批次平均。`ScopeCT/objective.py` 未改动。
- 36001 token 的头部/索引回归保留 36000 位置的 CT 目标和末尾 EOS；这项测试没有运行 36K 的模型骨干。
- 原生线性注意力回退的 FP16 小值/零向量反向问题已用 FP32 数值保护处理，含补零的微型混合模型全部 LoRA 梯度有限，并与 FP32 参考核对。
- 新增训练条数选择在原始 JSON 任务行级执行，随后保留对应任务的全部轮次和角色轨迹；清单记录原始总数、所选条数和任务分布。
- 普通原生模型重载导出权重后无需注册私有 attention，支持 KV-cache 生成。

五组烟雾测试的合成数据、随机微型模型只用于工程验证，不能用于文学能力比较：

| 微型 CPU 测试 | 单步训练损失 | 完整导出重载 |
| --- | --- | --- |
| Qwen3 SFT，低显存路径 | 6.0550537109375 | 通过 |
| Qwen3 CT，低显存路径 | 6.097451686859131 | 通过 |
| Qwen3.5 SFT，低显存路径 | 6.0800628662109375 | 通过 |
| Qwen3.5 CT，低显存路径 | 6.120739459991455 | 通过 |
| Qwen3.5 CT，标准路径 | 6.120739459991455 | 通过 |

两进程完整 Trainer 的复现命令如下，所用随机微型模型与合成数据由烟雾测试生成：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  ScopeCT/train.py --worker --cpu \
  --data /tmp/q35_release_smoke/synthetic_train.json \
  --model /tmp/q35_release_smoke/tiny_qwen3_5 \
  --output /tmp/q35_trainer_ddp_release \
  --max-samples 1 --max-length 30000 --max-steps 2 \
  --batch-size 1 --grad-accum 2
```

`--worker` 仅用于这条 CPU 分布式工程检查；正式 GPU 命令无需手动加 torchrun 或 worker。

## 未验证的边界

未运行真实 Qwen3.5-2B 权重、四张 TITAN RTX、470.182.03 驱动或 3.6 万 token 的完整 GPU 骨干反向，因此不保证实际峰值显存、运行时长或一次成功。分块降低注意力和词表输出的工作区；模型状态、层级激活和原生线性注意力递推仍占显存。65536 是长度上限，动态 padding 不会把每条序列补到此上限；程序不会自动截断或删除监督。

实际显卡启动时仍须通过代码中的 CUDA FP16 反向与 NCCL all-reduce 预检。该预检不是整套模型的显存验收。

官方实现依据：

- [Transformers v5.3.0 Qwen3.5](https://github.com/huggingface/transformers/tree/v5.3.0/src/transformers/models/qwen3_5)
- [Qwen3.5-2B tokenizer 配置](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/tokenizer_config.json)
- [Qwen3.5-2B chat template](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/chat_template.jinja)
- [Qwen3-4B-Instruct-2507 tokenizer 配置](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/tokenizer_config.json)

# 单个完整 JSON 的 DeepSeek 审核

脚本 `review_t1234.py` 读取一个已整合的 JSON 数组，每个对象是一条任务，
使用 `sample_id` 标识记录、`task` 标识 T1–T4、`split` 标识原有 train/test。
不合并其他数据文件，不重新划分集合，保留原有 `split`。

## 只填文件路径和 API Key

本云环境已在 `/workspace/.venvs/cultcrit-review` 安装 OpenAI SDK。

```bash
cd /workspace/CultCrit-Bench
source /workspace/.venvs/cultcrit-review/bin/activate
python review_t1234.py
```

脚本依次询问数据集 JSON 文件路径和 DeepSeek API Key，密钥输入不回显。
如果环境变量 `DEEPSEEK_API_KEY` 已配置，就不再询问密钥。
密钥不写进数据集、checkpoint 或代码。云环境需允许 HTTPS 访问
`api.deepseek.com`。

默认先读取接口提供的模型列表，优先选 `deepseek-chat`，否则选择列表中
可用的 chat/flash 模型，不自动选 reasoner/pro。实际模型会打印在屏幕并记录到
checkpoint；续跑沿用该模型。也可通过 `--model` 或 `DEEPSEEK_MODEL` 指定。
如果账号未提供可自动识别的聊天模型，脚本会提示使用 `--model`，不会猜模型 ID。

其他机器首次运行需安装依赖：

```bash
python -m pip install openai
```

## 设置本次处理条数

```bash
python review_t1234.py --limit 100
```

这表示本次最多处理 **100 条待处理/待重试记录**，包括处理失败的记录；
不是要保留 100 条通过样本。SDK 对同一请求的内部重试不另外占用条数。
省略 `--limit` 或设置 `--limit 0` 表示处理所有待处理记录。

也可直接指定路径，运行时只询问密钥：

```bash
python review_t1234.py --input "/你的路径/完整数据集.json" --limit 100
```

再次执行相同命令时，读取 checkpoint 并继续下一批，不会每次重复处理前 100 条。
`--limit` 可在各批次间改变，不影响恢复。

## 进度与统计

屏幕显示本次处理进度条和百分比，以及全库已有处理记录的条数和百分比：

```text
本次 [############------------] 50.0% (50/100) | 全库已有处理记录 150/1000 (15.0%)
```

失败记录也算“已处理”，进度不表示通过率。
结束时分别打印 **输入文件** 和 **累计审核通过输出** 的统计：

- 总条数、train 条数、test 条数。
- train 和 test 各自的 T1、T2、T3、T4 条数，以及合计。
- 缺失或其他 `split` 单列“未标注/其他”；不擅自分配到 train/test。
- 审核状态、剩余条数、输出和 checkpoint 的位置。

设置 limit 后，本次进度可能达到 100%，全库仍未完成；屏幕会明确提示继续下一批。

## 输出与恢复

例如输入 `/data/merged.json`，默认生成：

- `/data/merged_optimized.json`：通过审核的 JSON 数组，只含 `pass/revise`，
  保持输入顺序和原有 split，设置 `human_verified=false`。条数可能少于输入。
- `/data/merged_optimized.checkpoint.json`：完整审核记录、理由、拒收、错误及进度。

可用 `--output /路径/结果.json` 自定义输出位置。

- 重跑原命令：继续未处理的记录。
- `--retry-errors`：重试接口/模型回复错误与修改后结构校验失败。
- `--retry-abstain`：重新审核此前拒收的记录。
- 两个重试选项都受本次 `--limit` 约束。
- 输入文件、实际模型、提示词、输出 token 上限或审核版本改变时，使用新的输出名。
  本版保留 split，旧 v2 checkpoint 不能混用，请用新的输出名。
- 认证、余额、权限或模型/接口配置错误会停止批次，已完成结果保留。

`attempted_all` 表示每条已有记录，`review_complete` 表示每条都有有效审核决定
（包括拒收），`all_accepted` 才表示所有输入均进入输出。
退出码 0 表示全库审核流程完成，2 表示仍有错误或未处理项（包括本次 limit 用完），
130 表示手动中断。

## 审核与验证

允许模型替换 `question/gold/events/roles/required_exchanges` 的完整顶层值。
原文、证据及位置、样本和文档 ID、任务类型、split 和来源信息不可修改。
本地检查类型、证据位置及标准命名的证据 ID 引用；模型检查答案依据、任务清晰度、
轮次和角色要求。`SYSTEM_PROMPT` 中可补充更具体的 T1–T4 规则；模型审核不等于人工验证。

离线测试只用合成样本和模拟接口，无 API 费用：

```bash
python -m unittest -v test_review_t1234.py
```

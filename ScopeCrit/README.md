# ScopeCrit：文学评论 Agent 与三裁判评测

ScopeCrit 读取已构造完成的 T1–T4 数据集，以 `deepseek-flash` 运行 Agent，再由三个独立模型评分：`qwen3.8-flash`、`minimax-m3.1-flash`、`gemini-3.1-flash-lite`。支持一个或多个数据文件、每文件处理条数上限、断点续跑和单独评分。程序不训练模型，不修改输入，不重新划分 train/test。

## 快速开始

需要 Python 3.11 或以上。进入 `CultCrit-Bench/ScopeCrit` 目录：

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

复制 `settings.example.yaml` 为 `settings.local.yaml`，在**复制的本地文件**直接填写路径和 API Key。Windows 可以手动复制，或执行 `Copy-Item settings.example.yaml settings.local.yaml`；Linux/macOS 执行 `cp settings.example.yaml settings.local.yaml`。

```yaml
inputs:
  - 'F:\2027申博\博士面试\CultCrit-Bench-main\CultCrit-Bench-main\dataset\test.json'
  # 可继续添加其他数据文件绝对路径
agent_config: configs/deepseek.yaml
actor_workers: 4
max_samples: null
run_dir: null
judges_config: configs/judges.yaml
api_keys:
  deepseek: '填写 DeepSeek API Key'
  qwen: '填写阿里云百炼 API Key'
  minimax: '填写 APIKEY.FAN API Key'
  gemini: '填写 APIKEY.FAN API Key'
qwen_workspace_id: '填写北京业务空间 ID'
qwen_base_url: ''
minimax_base_url: https://api.apikey.fan/v1
gemini_base_url: https://api.apikey.fan/v1
```

Windows 路径使用 YAML **单引号**，避免反斜杠被当作转义字符。Linux 示例为 `/home/user/dataset/test.json`。相对路径以本地设置文件所在目录为基准。`inputs` 与兼容旧版本的单文件 `input` 只能填写一个；使用 `inputs` 时删除旧的 `input` 行。

先处理每个文件前 4 条任务，再处理全部：

```bash
python run.py --settings settings.local.yaml --max-samples 4
python run.py --settings settings.local.yaml --max-samples None
```

不填写 `--settings` 时默认读取程序目录的 `settings.local.yaml`。`max_samples: null` 也表示全部；命令行优先。条数指 JSON 数组中的**任务样本数**，不是评论文档数；选中一条 T3 后仍完整运行它的全部轮次。

`settings.local.yaml` 已加入 Git 忽略。真实 Key 不填入公开模板或 Python 源码，不写入运行清单与报表。用户无需执行 `export` 或设置系统环境变量；程序在当前进程内向 SDK 提供认证。

## 并发运行

`settings.local.yaml` 的 `actor_workers: 4` 表示 DeepSeek 最多同时处理 4 条任务样本；也可用 `--workers 4` 覆盖（范围 1–32）。先完成本数据文件的答题阶段，再开始三个裁判的并发评分，尚未完成的答卷不会交给裁判。

`configs/judges.yaml` 的顶层 `workers: 3` 表示最多 3 位裁判同时工作。每位裁判内部仍逐样本评分；这是总计最多 3 个评分请求同时在途，而不是每个裁判各 3 个请求。改为 `1` 可串行评分。

```bash
python run.py --settings settings.local.yaml --max-samples 20 --workers 4 --run-dir runs/parallel_001
```

并发边界为不同任务样本；同一条 T3 的轮次、同一条 T4 的消息交接和提交顺序不变。数据库事务和导出使用锁保护，不把网络调用放入数据库锁。遇到认证错误或 fail_fast 时停止启动新样本，已经在途的样本保存后退出；未启动样本保持 PENDING。中断也先等待在途工作收尾，再关闭状态库。

续跑会继承原运行的并发数，不要同时指定 `--workers`。更新代码后首次并发实验使用新输出目录；旧版本运行继续使用对应旧版本代码恢复。API 返回 429 时沿用原有退避重试，可降低并发数后新建实验。本地模型建议从 `actor_workers: 1` 开始，按推理服务容量调整。

## 答题模型与三个裁判

| 用途 | 模型 ID | 接口 |
|---|---|---|
| ScopeCrit 决策、支持核对和范围比较 | `deepseek-flash` | `https://api.deepseek.com` |
| 独立裁判一 | `qwen3.8-flash` | 百炼北京业务空间 OpenAI 兼容接口 |
| 独立裁判二 | `minimax-m3.1-flash` | `https://api.apikey.fan/v1` |
| 独立裁判三 | `gemini-3.1-flash-lite` | `https://api.apikey.fan/v1` |

裁判配置在 `configs/judges.yaml`，Agent 配置在 `configs/deepseek.yaml`。旧版 DeepSeek、GLM、Qwen、Grok 四裁判组合已由以上三裁判替代；DeepSeek 仍是答题模型。

Qwen 需要真实 WorkspaceId，程序根据 `qwen_workspace_id` 构造：

```text
https://你的WorkspaceId.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

WorkspaceId 不是 API Key，程序不能从 Key 推断。若已有完整接口地址，填写 `qwen_base_url` 并将 `qwen_workspace_id` 留空，两者不能同时填。MiniMax 与 Gemini 可以用同一个网关 Key，但该 Key 必须分别有相应模型权限。

模型名按配置原样发送，不静默替换。APIKEY.FAN 文档确认 OpenAI Chat Completions 使用 `/v1/chat/completions`；本次未使用真实凭据确认上述两个精确模型别名的账号可用性。网络、余额、模型权限及远端参数兼容性仍需在用户环境实际运行确认。各模型只接收自身参数，不照搬 DeepSeek 的 `thinking` 参数。

接口依据：[APIKEY.FAN 接入文档](https://apikey.fan/docs/api-scripts)、[百炼接口地址](https://help.aliyun.com/en/model-studio/base-url)、[Qwen3.8-Flash 模型说明](https://help.aliyun.com/en/model-studio/qwen3-8-flash)、[Qwen 结构化输出](https://help.aliyun.com/en/model-studio/qwen-structured-output)。

## 多文件、条数、输出与恢复

`inputs` 中各文件独立运行、独立评分。`--max-samples 4` 表示**每个文件**各取原始顺序前 4 条；无效条目也占原位置，不从后面补齐。不会把多个文件合并后共同截取，也不会把未选中的尾部当作失败答卷。

`run_dir: null` 自动创建并打印输出位置。单文件时可以指定一个运行目录；多文件时该路径作为输出根目录，各文件写入独立子目录。恢复时可指定单个运行目录，也可指定多文件运行的根目录。

```bash
# 指定本次输出位置
python run.py --settings settings.local.yaml --max-samples 4 --run-dir runs/trial_001

# 只运行 Agent，暂不调用三个裁判
python run.py --settings settings.local.yaml --max-samples 4 --no-judge

# 已有预测只评分，不重新答题
python run.py --settings settings.local.yaml --evaluate-only --run-dir runs/trial_001

# 恢复已有答题状态，随后继续评分
python run.py --settings settings.local.yaml --resume --run-dir runs/trial_001
```

恢复和单独评分沿用原运行输入与已选范围，不能通过重新设置条数扩大旧实验。保持原始输入、Agent 配置、提示词和代码一致；指纹变化时拒绝混合续跑。修改模型或评分规则的实验使用新目录。

已完成裁判响应从缓存复用。遗留 `STARTED` 请求代表远端结果不确定，默认暂停；显式 `--retry-uncertain-judges` 才重试裁判，`--retry-uncertain` 才重试答题，两者都可能重复计费。程序不承诺远端 API 的 exactly-once。

## Agent 实际流程

控制器按当前证据缺口选择读取、检索、提出主张、范围比较和提交。提交前由支持核对阶段检查证据，未提交候选不进入最终主张。检索使用字符二元组 Jaccard，只负责召回，不按关键词猜立场。

T1 提取指定范围评价；T2 整合独立片段并解释关系；T3 随证据公开维护累计主张；T4 通过 archivist 与 interpreter 的真实消息交换完成任务。程序分别限制未来证据和角色私有材料。Gold 仅进入独立评分，不反馈给答题 Agent。SQLite 用于运行状态和恢复，不是跨评论的语义长期记忆。

## 可选：本地或训练后的 Qwen

默认保持 DeepSeek。需要比较小模型时，可另建 Agent 配置，例如 `configs/qwen-local.yaml`：

```yaml
provider:
  kind: openai
  model: Qwen/Qwen3-1.7B
  base_url: http://127.0.0.1:8000/v1
  api_key_env: null
  extra_body:
    chat_template_kwargs:
      enable_thinking: false
evaluation:
  semantic_alignment: llm
  enable_paid_judge: true
  judges_config: judges.yaml
```

在 `settings.local.yaml` 将 `agent_config` 指向该文件。需提前启动 OpenAI Chat Completions 本地服务，`model` 必须与服务注册的模型 ID 一致。`chat_template_kwargs` 需服务端支持；程序不会自动启动服务、下载权重或训练。无认证只允许本机回环接口。

还可保留 `provider` 为 DeepSeek，只在 `stage_providers.ACTION` 配置上述本地 provider：决策用 Qwen、支持核对和范围比较仍用 DeepSeek。阶段名称为 `ACTION`、`SUPPORT_CHECK`、`SCOPE_COMPARE`。这种混合配置须作为独立实验报告，不能称为全部由小模型完成。

远端自定义答题服务需认证时，在 provider 设置 `api_key_env: SCOPECRIT_ACTOR_API_KEY`，在本地文件填写 `api_keys.actor`。这是程序内部字段名，用户仍无需设置环境变量。三个裁判不随答题模型切换。

## 评分规则与报表

每位裁判独立完成主张的一对一语义对齐，依据命题、对象、范围、阶段、条件、声音和不确定性。配对不预先要求 aspect、stance 或证据相同，避免循环定义。T2 另评关系解释，T3 逐轮评累计记忆，T4 评联合答案是否实质依赖双方材料。证据集合、引用合法性、真实投递和权限由程序核验。

先按每位裁判的对齐分别计算指标，再平均**相同指标**。正式 `mean` 要求三位预定裁判对该指标完整有效评分；失败、无法判断和不完整覆盖为 `null`，另列 `available_mean`、有效裁判数、标准差与范围。两位的可用均值不能称为三裁判均值。无分母指标也可能为 `null`，应结合原始计数判断。

| 任务 | 原始指标 | 可选展示综合分 |
|---|---|---|
| T1 | Aspect Micro-F1、Stance Macro-F1、Evidence F1 | 三项等权平均 |
| T2 | Claim Coverage、Claim Micro-F1、关系解释、Evidence F1 | 主张 F1、关系解释、证据 F1 等权平均 |
| T3 | 逐轮及最终轮 Memory State F1、False Overwrite | 全部预期轮次 Memory F1 平均 |
| T4 | Task Success、Expected Exchange Recall | 语义 Task Success rate |

关系解释 `correct=1`、`partial=0.5`、`incorrect=0`，未评分保持缺失。四任务综合分均有效时，整体分为四者等权宏平均。综合分是额外约定的展示指标，不替代原始指标；不依赖当前实验做 min–max 归一化。False Overwrite、交换召回、权限诊断、调用数、token、耗时单列，不重复混入综合分。不计算 Attribution Accuracy。

各数据文件独立输出：

| 文件 | 用途 |
|---|---|
| `judge_summary.json` | 三裁判指标、均值、完整性、任务分与整体分 |
| `metrics_summary.csv` | 各任务和整体完整指标表 |
| `per_sample_metrics.csv` | 逐样本指标，T3 含逐轮结果 |
| `report.md` | 可读结果与评分规则 |
| `judges/results.json` | 裁判状态、失败原因及用量 |
| `judges/<名称哈希>/alignment.json`、`metrics.json` | 各裁判配对及原始指标 |
| `judges/state.sqlite3` | 独立评分缓存和断点 |
| `predictions.jsonl` | 答案、主张、证据、T3 快照、T4 消息 |
| `traces.jsonl`、`messages.jsonl` | 实际动作与协作通信 |
| `state.sqlite3`、`progress.json` | 答题恢复状态 |
| `input_validation.json`、`failures.jsonl` | 输入校验和失败隔离 |
| `metrics.json` | 原始程序评测；多裁判汇总看 `judge_summary.json` |

详细定义见 [评测方法说明.md](评测方法说明.md)。自动模型评分不等同于人工金标准；论文应记录实际模型 ID、接口、参数、评分覆盖率和裁判分歧。

## 离线检查与限制

```bash
python -m pytest -q
python run.py --input examples/merged_demo.json --config configs/mock.yaml --run-dir runs/demo --no-judge
python -m scopecrit inspect --run-dir runs/demo
```

离线演示是明确标记的 `SYNTHETIC` fixture，Mock 使用预设响应，不从 gold 生成预测。本地 HTTP 模拟检验适配和流程，不证明真实账号可用或模型文学能力。具体检查边界见 [VALIDATION.md](VALIDATION.md)。

输入必须是 JSON 数组；条目按 schema 校验，问题记录保留，不静默改写数据。退出码 `0` 为请求流程完整完成，`1` 为部分、失败或未完整评分，`2` 为配置或输入错误。429、连接错误与部分 5xx 有限重试，400/401/403/404/422 不盲重试。未调用真实 API 前，不能承诺端到端没有 bug。

代码入口为 `run.py` → `local_settings.py` → `cli.py` → `engine.py`；评分由 `judging.py`、`judge_providers.py`、`reporting.py` 和 `evaluation.py` 完成。

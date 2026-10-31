# ScopeCrit：文学评论 Agent 与四模型自动评测

单个 JSON 数组输入，不训练模型、不重新划分数据。DeepSeek 完成 T1–T4；
答题结束后，四个独立裁判分别对齐预测与 gold，程序计算指标，再对相同指标取平均。
搜索仍为字符二元组 Jaccard；无需下载模型、安装 PyTorch 或配置 GPU。

## 1. 安装：Python 3.11 或以上

推荐 Python 3.11，本次实测 Python 3.12。

```bash
conda create -n scopecrit python=3.11 -y
conda activate scopecrit
cd /你的路径/CultCrit-Bench/ScopeCrit
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

没有 Conda 可以使用 `python3 -m venv .venv`，激活后安装 requirements。
依赖包括 OpenAI SDK、智谱官方 `zai-sdk`、Pydantic、PyYAML、tqdm 和 pytest。
`requirements-lock.txt` 保存本次验证的精确依赖版本。
**直接运行 `python run.py` 不需要 `pip install -e .`。**

## 2. 直接填写 API Key，不需要设置环境变量

复制模板：

```bash
cp settings.example.yaml settings.local.yaml
```

Windows PowerShell 使用 `Copy-Item settings.example.yaml settings.local.yaml`。
然后在 **settings.local.yaml** 中填写：

```yaml
input: ../CritPred_test.json
max_samples: null
run_dir: null
judges_config: configs/judges.yaml
api_keys:
  deepseek: "填你的 DeepSeek Key"
  glm: "填你的智谱 Key"
  qwen: "填你的 Qwen Key"
  grok: "填你的 apikey.fan 服务 Key"
qwen_workspace_id: "填你的业务空间ID"
qwen_base_url: ""
grok_base_url: https://api.apikey.fan/v1
```

Qwen 固定为 **华北 2（北京），`qwen3.8-flash`**。程序根据业务空间 ID 构造：
`https://业务空间ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`。
地域不能替代业务空间 ID，程序无法从 API Key 推断 WorkspaceId。
如果使用完整 `qwen_base_url`，请将 `qwen_workspace_id` 留空，二者选一个。

数据和输出等相对路径以本地设置文件所在目录为基准。
**settings.local.yaml 已加入 Git 忽略，不会放入运行 manifest、提示词或报表。**
GitHub 上只提交空模板；不要把真实 Key 填进 settings.example.yaml 或 Python 源码。
SDK 所需的认证在程序内部处理，不要求用户执行 export。

填好后直接运行：

```bash
python run.py
```

输出目录默认自动生成并打印。默认先检查四家配置是否齐全，再开始答题，避免答完后才发现无法评分。
可通过 `--settings /路径/私有配置.yaml` 使用其他本地文件。

## 3. 四个裁判的调用方式

| 裁判 | 默认 model | SDK / 地址 |
|---|---|---|
| DeepSeek | `deepseek-flash` | OpenAI，`https://api.deepseek.com`；high reasoning，thinking enabled |
| GLM | `glm-4.5-air` | 智谱官方 `zai-sdk` 的 `ZhipuAiClient` |
| Qwen | `qwen3.8-flash` | OpenAI，北京业务空间地址 |
| Grok | `grok-build-0.1` | OpenAI，用户指定的 `api.apikey.fan` 服务，默认 `/v1` |

配置在 `configs/judges.yaml`。按用户明确要求选择模型，**不会悄悄换成 Qwen Max、GLM-5.3 或其他模型**。
Grok 的 `/v1` 是兼容接口约定，实际路径可在本地 `grok_base_url` 修改；不会把 Key 发送到其他服务。
不同模型只接收自身配置的参数，DeepSeek 的 thinking 参数不会照搬给其他服务。
账户是否支持这些模型、第三方 Grok 接口及远端参数仍须实际调用验证，本次测试不产生真实 API 费用。

## 4. 处理条数与单独评分

`max_samples: null` 是 Python 的 None，处理全部；改成 `521` 则处理原始顺序前 521 条。
无效条目也占原始位置，不跳过后从尾部补齐。每条 T3 的全部轮次、T4 协作仍完整运行。
命令行可以覆盖设置：

```bash
python run.py --max-samples 521
python run.py --max-samples None
python run.py --input /其他路径/dataset.json
```

JSON 顶层必须是完整数组，不接受聊天文本中的 HTML 转义或多余反斜杠。
读取和校验不修改输入。`split` 字段可保留但不影响运行。
选中的范围保存到 manifest，后续评分使用相同范围，不会把未运行尾部当作失败答卷。

只答题、暂不付费评分：

```bash
python run.py --no-judge
```

已有运行只做四模型评测，不重新生成预测：

```bash
python run.py --evaluate-only --run-dir runs/你的输出目录
```

该命令使用本地设置的 input 作为 gold，必须与原运行输入文件一致。
已完成裁判响应会从缓存复用；失败裁判可以在修正 Key 后重试，其余已完成的结果保留。
更改模型、评分规则或已经评分的预测时，程序拒绝混用旧缓存，请使用新的实验目录。
裁判也不能把看到的 gold 回传给答题 Agent。

## 5. 指标、四裁判均值与综合分

每位裁判独立判断一对一语义等价；依据命题、对象、范围、阶段、条件、声音与不确定性。
aspect/stance 和证据标签不作为等价配对的前提，程序在配对后独立计算这些指标。
T3 逐轮对齐并评价最终状态；T2 另评关系解释；T4 另评回答是否实质依赖双方材料。
证据集合 F1、合法投递、权限和引用指标由程序计算一次，不交给模型投票。

**先各自对齐并计算指标，再平均同一指标。** 比如四位裁判的方面 F1 为 0.8、0.9、0.7、0.8，均值是 0.8。
报表提供每位裁判分数、正式 mean、可用 available_mean、完整裁判数、标准差和范围。
正式 mean 要求全部预定裁判对该指标所需目标完整评分。
请求失败、无法判断、不完整覆盖保留为 null；不能把三个裁判的平均冒充四裁判平均。
没有实际分母的指标也可能为 null，这与请求失败不同，可查看原始计数与裁判状态。

所有质量指标原本在 0–1，不做依赖当前实验样本的 min–max 归一化。
新增的综合分是**明确约定的展示指标，不是原技术文档中的标准指标**：

| 综合分 | 固定定义 |
|---|---|
| T1 | 方面 Micro-F1、立场 Macro-F1、证据 F1 等权平均 |
| T2 | 主张 Micro-F1、关系解释质量、证据 F1 等权平均 |
| T3 | 全部预期轮次 Memory F1 的平均；另保留最终轮 Memory F1 |
| T4 | 语义 Task Success rate，避免重复计算其组成条件 |
| 整体 | 四任务综合分等权宏平均；四个任务都有效才给正式整体分 |

T2 主张 F1 包含多报惩罚，同时保留原有 Claim Coverage；关系解释 correct=1、partial=0.5、incorrect=0，unscored=null。
T3 False Overwrite、T4 Exchange Recall 作为独立诊断项，不重复混入综合分。
费用、调用数、token、耗时、评分覆盖率不混进质量平均。
实际越权交付、未来/私有引用及不存在 ID 的引用单独报告；T4 的成功判定包含权限条件。
自动模型评分不是人工金标准。论文分析仍应展示原始指标及裁判分歧，不能只展示总分。

## 6. 每个数据集的输出文件

每个输入数据集使用独立输出目录，报表带数据集名称，保留逐样本、逐任务与整体结果。

| 文件 | 内容 |
|---|---|
| `judge_summary.json` | 四裁判全部指标、正式均值、可用均值、标准差、任务分与整体分 |
| `metrics_summary.csv` | 数据集与各任务所有质量指标；每位裁判各一列，均值与完整性另列 |
| `per_sample_metrics.csv` | 每条样本指标与均值，T3 额外有逐轮行 |
| `report.md` | 可读的完整指标表、综合分和评分规则 |
| `judges/results.json` | 各裁判状态、失败原因、输出目录、调用和 token 统计 |
| `judges/<名称哈希>/alignment.json` | 对应裁判的一对一语义配对 |
| `judges/<名称哈希>/metrics.json` | 对应裁判原始完整指标 |
| `judges/<名称哈希>/calls.jsonl` | 脱敏调用记录，不包含 Key 或隐藏思维链 |
| `judges/state.sqlite3` | 独立评分缓存，保留每次尝试和断点 |
| `metrics.json` | 程序原始结构评测，保持旧格式；四模型结果请看 judge_summary.json |
| `predictions.jsonl` | 答案、已提交主张、证据、T3 快照、T4 消息和答题用量 |
| `traces.jsonl / messages.jsonl` | 答题动作及真实协作通信 |
| `state.sqlite3 / progress.json` | 答题状态、恢复进度及样本状态 |
| `input_validation.json / failures.jsonl` | 输入问题、任务失败和隔离记录 |

日志会打印当前样本、角色、动作、模型阶段及裁判进度。
语义错误、网络错误、输入错误和未评分分别保留；失败样本不会悄悄从总数中删掉。
旧的人工对齐入口仍可作为可选兼容功能，但全自动模式不需要人工标注。

## 7. 无网络验收与故障恢复

```bash
python -m pytest -q
python run.py --input examples/merged_demo.json --config configs/mock.yaml --run-dir runs/demo --no-judge
python -m scopecrit inspect --run-dir runs/demo
```

演示为明确标记的 SYNTHETIC 人工 fixture，Mock 使用预设响应，不从 gold 生成预测。
四服务 SDK 的测试使用本地 HTTP 模拟，不访问网络，不收费；演示分数不代表模型文学能力。

如果使用了本地配置，可以用程序接口恢复答题；本地配置仅由 run.py 加载：
先用 `python run.py --evaluate-only --run-dir ...` 恢复评分；需要恢复答题时见下方 Python 调用或继续使用已有环境凭证的 CLI。

```python
from scopecrit.cli import main
from scopecrit.local_settings import configured_api, load_local_settings
with configured_api(load_local_settings("settings.local.yaml")):
    main(["resume", "--run-dir", "runs/你的输出目录"])
```

遗留 STARTED 裁判请求默认暂停，只有显式 `--retry-uncertain-judges` 才重试，可能重复计费。
答题阶段的不确定调用对应 `resume --retry-uncertain`。系统不承诺远端 API 的 exactly-once。
429、连接错误与部分 5xx 受限重试；400/401/403 不盲重试。裁判错误不会改写预测。
退出码：0 为请求的流程完整完成；1 为存在部分/失败/未完整评分；2 为配置或输入错误。
原有运行的输入/代码/提示词指纹变化时拒绝直接续跑，避免混合实验。

当前训练文件中 CritPred48:T1–T4 证据编号不连续，严格校验会隔离这四条，原数据不自动改写。
首版文档未包含后续用户要求的四模型评分，本 README 和代码记录当前扩展；原 Word 文件保持不变。

## 8. 阅读代码

`run.py` → `local_settings.py`（本地配置）→ `cli.py`（工作流）→ `engine.py`（T1–T4）→
`judging.py`（独立对齐）→ `reporting.py`（平均与报表）。
`judge_config.py / judge_providers.py` 分别负责四服务配置与 SDK 参数，`evaluation.py` 保留确定性指标计算。
`storage.py` 负责答题事务，裁判另用独立缓存；SQLite 不是跨任务语义长期记忆。

# ScopeCrit：文学评论 Agent

按仓库中 `../docs/ScopeCrit_研究方案与技术文档.docx` 实现的 Python 项目；全部代码位于 `ScopeCrit/`。
**不训练模型，不拆分数据集，只输入一个 UTF-8 JSON 数组文件。**
真实推理使用 DeepSeek 官方兼容 OpenAI 的接口；离线演示使用明确标记的人工 Mock。
无需安装 PyTorch、CUDA、Embedding 模型或向量数据库。

## 1. 配置环境

推荐 **Python 3.11**，最低版本也是 **3.11**，Python 3.12 同样支持。
这里需要区分：Python 是 Conda 环境的解释器版本；`DEEPSEEK_API_KEY` 是环境变量，两者分别设置。
从克隆后的仓库进入 `ScopeCrit` 子目录：

```bash
conda create -n scopecrit python=3.11 -y
conda activate scopecrit
cd /你的路径/CultCrit-Bench/ScopeCrit
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python --version
```

没有 Conda 也可以：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` 包含运行依赖和 pytest；`requirements-lock.txt` 保存本次验证环境的精确依赖版本。
代码在 Python 3.12 环境实测。**仅安装 requirements.txt 即可直接使用 `python run.py`。**
文档中的 `scopecrit` 简写需要额外执行一次 `python -m pip install -e . --no-deps`；
也可以始终使用 `python -m scopecrit`，不必安装该命令行简写。

## 2. 先运行不收费的演示

```bash
python -m pytest -q
python -m scopecrit validate --input examples/merged_demo.json --fixture --report runs/demo_validation.json
python run.py --input examples/merged_demo.json --config configs/mock.yaml --run-dir runs/demo
python -m scopecrit inspect --run-dir runs/demo
```

四条演示样本是 **SYNTHETIC 人工 fixture**，分别覆盖 T1–T4；不是论文实验数据。
Mock 按 `examples/mock_script.json` 回放人工编写的响应，不读 gold，不衡量模型文学能力。
`--fixture` 是演示数据校验开关；真实数据不需要。脚本缺项时会失败，不会伪造答案。
已存在的目录不能再次 `run`，请使用新的目录或 `resume`。

## 3. 用 DeepSeek 运行自己的 JSON

DeepSeek 官方 API **必须提供 API Key**。这里只使用一个环境变量，不配置密钥管理服务，
不把密钥写进代码或运行文件。官方说明：<https://api-docs.deepseek.com/zh-cn/>。

Linux/macOS：

```bash
export DEEPSEEK_API_KEY="你自己的 DeepSeek API Key"
python run.py --input /你的数据路径/dataset.json
```

Windows PowerShell 对应设置：

```powershell
$env:DEEPSEEK_API_KEY="你自己的 DeepSeek API Key"
python run.py --input "D:\数据\dataset.json"
```

完成环境安装后，**只需替换数据路径和 Key，默认就会调用 DeepSeek**。
无需填写配置路径或输出目录；结果自动写入当前目录 `runs/deepseek_时间_随机标识/`，屏幕显示完整路径。
重复执行会生成新目录；需要续跑时用 `python -m scopecrit resume --run-dir 上次输出目录`。
也可以直接使用仓库内数据：`python run.py --input ../CritPred_test.json`。
数据必须是合法 JSON：`[{第一条完整样本}, {第二条完整样本}]`，不能带聊天消息中的反斜杠或 `&#x20;`。
`split` 字段可保留但不参与运行；不需要另传 train 文件或 test 文件。

### 控制处理条数：max_samples

```bash
# 默认 None，处理整个文件
python run.py --input ../CritPred_train.json
python run.py --input ../CritPred_train.json --max-samples None

# 只处理原始顺序的前 521 条，每条任务完整执行
python run.py --input ../CritPred_train.json --max-samples 521
```

参数也可以写在 `configs/deepseek.yaml` 中：

```yaml
runtime:
  max_samples: null  # Python 中的 None；改为 521 就处理前 521 条
```

命令行参数优先；显式 `--max-samples None` 可覆盖配置中的数字。`0`、负数、小数、布尔值等会被拒绝。
超过数据总数时处理全部现有条目。先截取前 N 条再校验，不会跳过无效条目后补取尾部数据；
未选择的条目不推理、不进入本次评分。T3 各轮、T4 协作仍然完整执行，不会按轮次截断。
JSON 文件整体必须语法合法，参数不会免除对整个文件的解析。
条数设置写入 manifest，`resume` 和 `evaluate` 自动使用同一批数据。

`configs/deepseek.yaml` 保留技术文档指定的 `deepseek-flash`、thinking、`reasoning_effort: high`。
**模型名称和参数须以你的账户当前支持为准，真实 API 尚未验证**。服务拒绝参数时会明确记录错误，
不会偷偷换模型或关闭 thinking。需要更换时编辑 YAML，然后使用新的运行目录。
云环境若限制出站访问，还需在环境设置允许 `api.deepseek.com`；本次未更改已发布环境配置。

核心调用保持官方 SDK 方式，完整实现在 `scopecrit/providers.py`：

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    max_retries=0,  # 应用统一管理重试，避免重试次数相乘
)
response = client.chat.completions.create(
    model="deepseek-flash",
    messages=messages,  # 只包含当前角色合法可见的材料
    response_format={"type": "json_object"},
    stream=False,
)
```

## 4. 运行到底做什么

运行同时产生 **预测答案、主张记忆、动作轨迹、协作消息和分任务指标**。

| 任务 | 行为 |
|---|---|
| T1 | 阅读全文，提出带评价方面、立场、范围和证据的主张，支持核对后提交 |
| T2 | 在 T1 能力上进行范围比较；最终必须包含多证据和关系解释 |
| T3 | 每轮只公开新增证据，保留此前已提交主张，输出逐轮完整记忆 |
| T4 | archivist / interpreter 独立上下文；通过真实队列投递证据，由 interpreter 回答 |

动作是 `PLAN_GAPS / READ / SEARCH / PROPOSE / COMPARE / COMMIT / DEFER /
REQUEST_PEER / SEND_EVIDENCE / YIELD / FINALIZE`。
不是每条样本机械执行固定顺序，模型根据当前材料和缺口自行选择。
搜索采用字符二元组集合 Jaccard 相似度，分数相同按证据 ID 排序。

模型永远看不到 `gold`、预期操作、标准记忆和预期交换。
T3/T4 请求根本不构造全文字段；不同样本不共享内容记忆或索引。
支持核对只接收该候选及相关可见证据，比较只接收所选主张和证据。
不同评价范围的正反意见可以共存；提交后的记忆采用追加式，不提供删除工具。

T4 每次激活默认最多四个动作，每轮 archivist 先运行，interpreter 后运行，最多六轮。
消息发送只进入 outbox，接收方下次激活才增加证据权限。
`COMPLETED` 表示合法完成控制流程，**不代表语义答案一定正确**。
程序退出码：`0` 表示本批任务全部正常完成；`1` 表示存在无效、部分完成、失败或待处理条目；
`2` 表示输入/配置错误。API 返回 400/401/403 时停止批次，其余条目保持待处理，避免重复提交无效请求。
已标记 PARTIAL 的样本保留为历史结果；修正认证或模型参数后，请重新 `run` 使用新目录，避免把失败批次当作成功续跑。

## 5. 输出与指标

屏幕示例：

```text
[Demo:T2] 第1轮 / solo / 决策 5
  [模型] solo / ACTION / 调用 7
  [模型] solo / SCOPE_COMPARE / 调用 8
  [动作] COMPARE → 成功
[结果] Demo:T2 → COMPLETED (FINALIZED)
```

| 文件 | 内容 |
|---|---|
| `manifest.json` | 输入、配置、代码及提示词指纹，不包含密钥 |
| `state.sqlite3` | 恢复依据；动作、主张、消息事务和模型调用记录 |
| `input_validation.json` | 数据问题和隔离记录 |
| `progress.json` | 已持久化样本状态及调用数 |
| `predictions.jsonl` | 每条任务的答案、主张、证据、T3 快照、消息、用量 |
| `traces.jsonl` | 动作日志及模型请求/最终响应，不保存隐藏思维链 |
| `messages.jsonl` | 真实协作消息与投递序号 |
| `failures.jsonl` | 失败、无效、部分完成与不确定调用 |
| `metrics.json` | `overall / per_task / per_sample` 指标 |
| `alignment_template.json` | 人工语义对齐模板 |

每个子任务都会单列统计，缺少的任务不会补造。默认自动计算证据 F1、完成状态、调用数、权限与引用问题，
T4 还计算方向性预期交换召回率。**证据 F1 只比较证据集合，不证明解释正确。**

技术文档规定：主张语义必须独立对齐，不能用词重叠或主张 ID 相同冒充正确。
因此缺少完整对齐时，方面/立场 F1、主张覆盖、记忆 F1、语义任务成功等值为 `null`，
报告中同时列出语义评分覆盖率；未知 token 数也是 `null`。
真实失败不会从分母删掉；有预测但未对齐的结果保持未评分。

对齐方法：复制 `alignment_template.json`，人工检查预测和 gold，填写一对一的等价主张配对，
完成后将 `alignment_complete` 改为 `true`。不等价的预测留作未匹配项；方面、立场及证据是否相同不是判定命题等价的前提。
T3 要逐轮对齐。T2 的 `relation_quality` 及 T4 的 `dual_dependency` 也是人工判断。

```bash
# 演示对齐只能用于演示数据，不能用于真实数据！
scopecrit evaluate --run-dir runs/demo --gold examples/merged_demo.json --alignment examples/alignment_demo.json

# 实验：gold 仍是运行时同一个输入文件，无需拆成另一份数据集
scopecrit evaluate --run-dir runs/deepseek_001 --gold data/dataset.json --alignment annotations/alignment.json
```

`examples/alignment_demo.json` 展示格式；程序拒绝重复配对和不存在的主张 ID。
未实现自动语义裁判、论文对照实验与 bootstrap 置信区间，不能将演示分数当作研究结论。

## 6. 断点、故障与排查

```bash
scopecrit resume --run-dir runs/deepseek_001
scopecrit export --run-dir runs/deepseek_001
scopecrit inspect --run-dir runs/deepseek_001
```

已持久化的模型响应会复用，动作与消息保持幂等。输入、配置、代码、提示词或 Mock 脚本变化时拒绝续跑，
避免混合实验；请新建运行目录。模型请求超长时保存部分结果，不静默截断正文。
预算耗尽后只汇总已提交主张，答案明确标记为“部分结果”，不会伪造正常完成。
429、部分 5xx、连接与超时错误受统一预算约束重试；400/401/403 等不盲重试。

如果进程在 API 返回前后崩溃，仅留下 `CALL_STARTED`，无法知道远端是否已经收费，默认暂停该样本、继续其他样本。
只有明确接受可能重复计费时才运行：

```bash
scopecrit resume --run-dir runs/deepseek_001 --retry-uncertain
```

本框架不承诺远端 API 的 exactly-once。状态库、轨迹可能包含原文和预测，分享时自行选择范围。
进度条旁的日志用于定位错误，完整模型响应在 `traces.jsonl`；密钥和隐藏思维链不记录。

仓库现有 `CritPred_train.json` 和 `RAW_Dataset/train.json` 中，`CritPred48:T1–T4`
存在证据编号从 E026 跳至 E046 的问题，严格校验将隔离这四条；不自动改写输入。
其他样本继续运行，问题会保留在校验报告中。

## 7. 从哪里读代码

建议按这个顺序阅读，变量和函数尽量直白，关键权限与事务位置有中文注释：

1. `cli.py`：程序入口、进度条、输出、自动评分。
2. `data.py`：只读输入校验和文档一致性。
3. `schemas.py`：模型允许返回什么动作，禁止哪些字段。
4. `engine.py`：可见材料投影、缺口/主张工具、支持核对、T1–T4 控制循环。
5. `providers.py`：DeepSeek SDK 与离线 Mock。
6. `storage.py`：SQLite 事务、断点和幂等。
7. `evaluation.py`：独立评分，不回传 gold 给运行中的模型。

当前交付范围是技术文档中的首版系统；本地 Qwen 推理与 Embedding 检索不包含在本次实现中。
离线测试验证程序契约，不证明模型的文学理解、范围判断或方法创新已获实验支持。

# T1–T4 五维 κ / α 一致性检验（v3，支持单个或多个JSON）

本工具对 GPT 构造、DeepSeek 优化后的数据集做三模型独立质量复评。输入为**一个或多个 UTF-8 JSON 数组文件**，支持既有 T1–T4 协议。三位裁判分别为 `qwen3.8-flash`、`glm-4.5-air`、`grok-build-0.1`，每条完整任务样本独立调用三次 API，互不可见其他裁判的评分。不会运行 ScopeCrit Agent，不改写输入或重新划分 train/test。

## 1. 填配置，直接运行

需要 Python 3.10 或以上（推荐 3.11+）。**仅使用 Python 标准库，无需 pip 安装 openai 或 zai-sdk。** 三家均通过 HTTP 的 OpenAI 兼容 Chat Completions API 调用，GLM 使用智谱中国站的兼容接口。

解压后编辑 `config.json`：

```json
"input_files": ["你的数据集.json"],
"limit": 100,
"qwen_workspace_id": "你的北京业务空间ID"
```

再分别填写三个裁判的 `api_key`（真实Key填这里），交付时全部留空。`api_key_env`填写环境变量名称，不能把真实Key填到这个字段。也可保持 Key 为空，分别使用环境变量 `DASHSCOPE_API_KEY`、`ZAI_API_KEY`、`GROK_API_KEY`。

Windows JSON 路径建议使用正斜杠，例如 `D:/data/train.json`，或将反斜杠写成 `\\`。相对路径默认相对于 `config.json` 所在目录。多个输入不能重复指向同一个文件。

在解压后的文件夹中运行：

```bash
python consistency_check.py
```

**只有一个数据集时，直接指定一个文件即可：**

```bash
python consistency_check.py --input dataset.json --check-only
python consistency_check.py --input dataset.json --limit 100
```

有多个文件时按顺序列出：

```bash
python consistency_check.py --input train.json test.json --limit 100
python consistency_check.py --input a.json b.json c.json --limit 100
```

命令行`--input`优先于配置文件。配置中的`input_files`支持单元素列表、多个路径列表或单个路径字符串；兼容`input_file`单路径字段。旧配置如`["dataset.json", ""]`会忽略空占位；使用`--input`时可不填写配置中的路径。

`limit` 是**本次最多处理的待评样本条数**，按输入文件的列出顺序及各文件内部的原始顺序推进；每条正常由三位裁判分别评分，重试会增加实际请求数。`0` 表示处理全部待评样本。再次运行同一命令继续下一批，已经成功的评分直接复用。它不表示每个任务各100条，也不是每个文件各100条。正式解释整体一致性前，应完成全量或预先设计的抽样；不能用顺序前100条代表全库。

Qwen 北京地址会自动替换 `{WorkspaceId}`，模型固定为 `qwen3.8-flash`。业务空间ID不是Key，仍需填写。GLM 模型已改为 `glm-4.5-air`，没有使用示例中的 `glm-5.3`。

Grok 根据提供的域名，默认采用 `https://api.apikey.fan/v1/chat/completions`。该第三方网关的模型权限及兼容参数未做真实账号验证；若网关文档要求不同路径，修改 `base_url`（不含 `/chat/completions`）。不会自动替换为其他模型。Qwen/GLM 默认启用 JSON 模式并关闭思考；Grok 默认只以提示词要求 JSON，避免假定网关支持 `response_format`。参数和模型返回的实际 `model` 均保留记录。

## 2. 你要看的结果

默认输出到 `consistency_results/`：

| 文件 | 内容 |
|---|---|
| **report.md** | 可直接阅读的 T1/T2/T3/T4 结果表，含所有输入合并结果及各文件各自结果 |
| **task_scores.csv / task_scores.json** | 每个范围四行，分别为T1–T4；五个`kappa_维度`和五个`alpha_维度`列，并列各指标有效样本数、未定义原因；`label_kappa`仅为补充 |
| summary.json | 完整统计、每维分布/均值/中位数、三模型成对分歧、按任务及按文件结果 |
| ratings.csv / ratings.json | 逐样本、逐模型评分及状态；CSV可直接用Excel打开 |
| review_queue.json | 待复核样本、完整原样本与触发理由，不自动修改或淘汰数据 |
| checkpoint.sqlite3 | 断点与每次真实请求记录；请保留 |
| calls.jsonl | 每次请求的时间、耗时、最终回复、返回模型、用量和错误；未知用量为null |
| run_manifest.json / rubric.txt | 冻结Rubric、模型、温度、输入指纹、调用顺序等可追溯资料 |
| input_validation.json | 不调用模型即可判断的结构错误 |

`scope=COMBINED` 代表所有输入文件合并；其他scope是对应输入文件路径。单文件时合并结果与该文件结果相同。即使缺少某类任务，结果表仍保留T1–T4四行，统计量为null，并注明没有足够评级。

**κ 与 α 均按任务、按五个维度分别计算**。α沿用附件的有序评级定义，κ按本次要求扩展到五维评分：

- **κ：Fleiss' κ（nominal，未加权）**，每个维度分别将1、2、3、4、5分视为五个离散类别，只使用该维度三位裁判均有有效评分的样本。这衡量精确同分一致性，不区分1与2和1与5的分差；顺序信息由ordinal α保留。
- **α：Krippendorff's α（ordinal）**，针对五维1–5有序评分，五个维度**分别计算**：Accuracy、Naturalness、Informativeness、Contextual Fidelity、Evidence Grounding。采用经验边际频数的有序距离，不误用等距平方差。

因此每个T任务的主结果为**5个维度 ×（κ、α）=10个系数**。Accuracy、Naturalness、Informativeness、Contextual Fidelity、Evidence Grounding各有一对系数；所有输入合并的T1–T4合计40个主系数，另有各输入文件分别的同类结果。PASS/REVIEW/FAIL的标签κ保留为`label_kappa`补充统计，不替代五维κ。`report.md`按“任务｜维度｜κ｜α”展示，每个范围20行；CSV/JSON保留每任务一行的宽表。失败评级不填0；α可使用至少两位有效裁判的样本，单独报告样本数。所有人始终给同一类别时，机会校正分母为0，返回null而不是强行记作1。系数可能为负，不截成0。

五维的1分/5分端点来自附件；中间2/3/4分锚点与以下标签阈值是本实现预先明确的操作规则，已写入同一冻结提示词：任一维≤2→FAIL；否则任一维=3或有证据错误→REVIEW；其余→PASS。输出不符合阈值或类型的回复视为格式错误并重试，不静默修改模型输出。正式批量评分后若修改规则，需要新输出目录。

结构无效样本不发送API，单独列出；有效但API失败的样本保留在覆盖率分母。`review_queue.json` 包含结构无效、评分缺失、任一裁判非PASS、证据问题、标签不一致、任一维最大分差≥2的样本。**一致性不是事实正确性的证明**，也不把模型评分当成人工验证。

## 3. 校验、恢复及补评

先检查真实输入的结构，不需要Key、不调用API：

```bash
python consistency_check.py --check-only
```

用附带的四条自编合成样本验证安装和输入流程：

```bash
python consistency_check.py --input example_1.json example_2.json --check-only --output-dir demo_validation
```

这四条样本仅用于软件测试，不是实验数据，不能用其模拟结果证明模型质量。

补评先前网络/格式失败或中断的裁判；其他成功评分保留：

```bash
python consistency_check.py --retry-errors --limit 100
```

认证、余额、权限、模型或请求配置等明确4xx错误会停止当前批次，避免持续请求。修正后用 `--retry-errors` 继续。超时/429/5xx或无效JSON最多尝试配置中的 `max_attempts` 次，之后保留缺失结果，其他样本可继续。输入较长时若 `finish_reason=length`，结果不进入统计，应提高 `max_tokens` 并使用新的输出目录；若服务限制上下文，不会偷偷截断原文。

只从已有checkpoint重新导出报告，不需要Key：

```bash
python consistency_check.py --report-only
```

**v3单文件支持更新：** 只需替换`consistency_check.py`，保留已填写的`config.json`和结果目录。旧版双文件checkpoint仍可续跑。将既有多文件批次改成不同的文件集合时需换输出目录，因为评审样本集合发生了变化。

**从v1评分记录升级五维κ：** 保留已填写的`config.json`和`consistency_results/`，替换三个运行用Python文件（`consistency_check.py`、`agreement.py`、`dataset_validation.py`），然后运行`python consistency_check.py --report-only`。本次只扩展本地统计与报表，评分Rubric/API请求指纹未变，已有成功评分无需重新调用模型。

Key改变、limit改变不影响恢复。输入内容/文件分组、模型、地址、温度、输出token上限、JSON参数或提示词改变时，拒绝混用旧结果；指定 `--output-dir 新目录` 建立新批次。多个文件之间的重复sample_id会阻止运行，避免统计重复样本；同一doc_id跨任务的材料冲突会隔离相关样本。

Ctrl+C会保留成功评分，中断请求记录为uncertain。服务端是否已经计费无法由客户端保证，明确使用 `--retry-errors` 才补发。退出码0=全部输入均完成三裁判评分；2=仍有待评/失败/结构无效，或配置错误（限条批次结束而全库未完成时也为2）；130=手动中断。非PASS是有效评分，不单独导致运行失败。

## 4. 完成数据构造闭环

先冻结Rubric并运行评审，再按T1–T4检查五维 κ/α、缺失覆盖率及分歧样本。对待复核队列进行人工裁决或修订，并在独立数据文件中保存修订版；随后用新输出目录重评，保留前后批次追溯。工具不凭一致性系数自动宣布数据集通过。

如有真实人工评分，可填写 `human_ratings.example.json` 同样的四个评分字段及sample_id；文件是JSON数组，可只包含人工抽查样本：

```bash
python consistency_check.py --report-only --human-ratings 人工评分.json
```

人工与每个模型分别比较，不混入三模型的Fleiss κ。没有人工文件时明确标为未提供。

## 5. 离线测试与接口依据

```bash
python -m unittest discover -v
```

测试使用手算统计例、合成任务和本地HTTP模拟服务；不会读取真实Key或访问付费接口。已验证三模型独立请求、单/多文件限条、恢复/补评、错误隔离、输出与类型校验。真实账号权限、余额、限流及网关行为需要你填写配置后验证，不能由离线测试保证。

接口依据：

- [阿里云文本生成与北京业务空间地址](https://help.aliyun.com/en/model-studio/text-generation)
- [Qwen深度思考参数与qwen3.8-flash](https://help.aliyun.com/zh/model-studio/deep-thinking)
- [智谱OpenAI兼容地址](https://docs.bigmodel.cn/cn/guide/develop/openai/introduction)
- [GLM-4.5 / Air及thinking参数](https://docs.bigmodel.cn/cn/guide/models/text/glm-4.5)

本工具仅依据附件的数据协议与一致性方案实现当前请求，没有执行附件附录中开发完整ScopeCrit系统的指令。

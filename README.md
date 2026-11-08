# ContextCrit

中文文学评论 T1–T4 数据审核、质量一致性检验、ScopeCT 训练与 ScopeCrit Agent 评测。

流程：**按原始文档划分 train/test → 分别构建 T1–T4 数据集并审核优化 → local check → 一致性检验 → ScopeCT 训练 → Agent 评测**。各步骤可单独运行，已经完成的步骤可以跳过。

先按原始评论文档的 `doc_id` 固定划分，再在各自集合中构建任务；同一评论的全部任务和轮次继承相同 split。后续审核与检查不重新分配。已有正确划分且文档互不重叠的 train/test 可直接沿用；本仓库不提供数据构建或划分命令。

保留目录为 `ScopeCT/`、`ScopeCrit_Consistency/`、`ScopeCrit/`、`ScopeSFT/`。根目录 `review_t1234.py` 是审核入口，`test_review_t1234.py` 是其离线测试，两者需同时保留。

## 1. 环境配置

以下安装命令在项目根目录执行。API 审核、一致性检验和 Agent 评测共用环境：

```bash
conda create -n scopecrit python=3.11 pip -y
conda activate scopecrit
python -m pip install -r ScopeCrit/requirements.txt
python -m pip install -e ./ScopeCrit
```

ScopeCT 训练使用独立 Conda 环境（CUDA 11.8 版 PyTorch）：

```bash
conda create -n scope-qwen35 python=3.11 pip -y
conda activate scope-qwen35
python -m pip install "torch==2.5.1+cu118" --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r ScopeSFT/requirements.txt
python -m pip check
```

数据和本地模型放在仓库外，命令中的 `/path/to/...` 均为占位示例，请替换为实际绝对路径；包含空格时加引号。各模块需保持同级目录关系。

配置模板不包含真实 Key。请复制成本地配置后填写自己的路径、API Key 和接口地址，不将本地配置提交到 Git。公开服务地址是可配置的默认接口，不是认证凭证。

## 2. 审核优化

在项目根目录运行，分别处理 train/test，输入为 T1–T4 JSON 数组。以下以 test 为例：

```bash
conda activate scopecrit
python review_t1234.py --input /path/to/test.json --output /path/to/test_optimized.json \
  --model deepseek-flash --max-tokens 32768 --limit 0
```

未提供 API Key 时，程序会隐藏输入提示。输出接受样本及 `test_optimized.checkpoint.json`，相同命令可以续跑；`--limit 100` 表示本次最多处理 100 条待审核记录，`0` 表示全部。

## 3. Local check：本地校验与导出

审核优化后，使用 `ScopeCrit_Consistency/local_check.py` 检查结构、证据引用、T3 轮次及 T4 角色约束，不调用模型：

```bash
python ScopeCrit_Consistency/local_check.py --input /path/to/test_optimized.json \
  --output-dir /path/to/validation_output
```

输出 `test_optimized_validated.json`、`test_optimized_invalid.json` 和 `test_optimized_validation_report.json`，分别为通过数据、未通过数据和校验原因。train 同样执行一次。这个步骤不划分集合；有问题的记录单独保留供复核，输入文件不变。

## 4. 三模型一致性检验

复制配置，在本地文件中填写数据路径、三个裁判的模型、接口地址和 API Key：

```bash
cp ScopeCrit_Consistency/config.json ScopeCrit_Consistency/config.local.json
python ScopeCrit_Consistency/consistency_check.py \
  --config ScopeCrit_Consistency/config.local.json \
  --input /path/to/validation_output/test_optimized_validated.json \
  --limit 0 --workers 4 --output-dir /path/to/consistency_results
```

输入可在 `--input` 后连续指定多个文件。`--workers` 控制总 API 请求并发；增加 `--check-only` 只校验结构，`--report-only` 只从已有断点重新导出报告，`--retry-errors` 补评失败项。

主要输出：`task_scores.csv`、`summary.json`、`review_queue.json` 和 `checkpoint.sqlite3`，包含五维质量评分及 κ/α 一致性统计。模型一致性不等于人工验证。

train/test 可分别运行并使用各自输出目录，也可作为多个输入统一检验。完成分歧复核后，训练继续使用 train，独立测试继续使用 test，保持最初划分。

## 5. ScopeCT 主实验训练

ScopeCT 将训练标签展开为教师 ACTION 轨迹，使用完整正例监督与有效语义分歧约束进行单阶段训练。

```bash
conda activate scope-qwen35
cd ScopeCT
python train.py --data /path/to/validation_output/train_optimized_validated.json --model /path/to/models/Qwen3.5-2B \
  --max-length 65536 --gpus 0123 --max-samples None
cd ..
```

`--max-samples 20` 选取前 20 条任务，再展开全部动作；`None` 或省略表示全部。默认启用低显存路径，长度上限不代表固定填充长度。全训练集零有效负例对时默认报错，不会把纯正例训练记作 ScopeCT。

默认输出到 `ScopeCT/outputs/<运行目录>/`：`final_model/` 为合并模型，`adapter/` 为适配器，另有 `data_summary.json`、训练日志和负例审计。实际显存需求取决于序列长度和硬件。

## 6. ScopeCrit Agent 评测

```bash
conda activate scopecrit
cd ScopeCrit
cp settings.example.yaml settings.local.yaml
```

在 `settings.local.yaml` 中填写一个或多个测试集绝对路径（如 `/path/to/validation_output/test_optimized_validated.json`）、决策模型及三裁判 API Key。默认决策配置为 `configs/deepseek.yaml`；模型和接口参数在配置文件中修改。真实 Key 仅保存在被 Git 忽略的本地配置中。

运行 Agent 并自动调用三裁判评分：

```bash
python run.py --settings settings.local.yaml --max-samples 600 \
  --workers 4 --run-dir runs/parallel_test
```

`--workers 4` 表示最多同时处理 4 条样本，样本内部仍遵守动作、轮次和消息投递顺序。`--max-samples None` 表示全部；多输入时限制分别应用于每个文件。新实验使用新输出目录。

也可分开执行 Agent 和评分：

```bash
python run.py --settings settings.local.yaml --max-samples 600 \
  --workers 4 --run-dir runs/agent_only --no-judge
python run.py --settings settings.local.yaml --evaluate-only --run-dir runs/agent_only
```

主要结果为运行目录中的 `judge_summary.json`、`metrics_summary.csv`、`per_sample_metrics.csv`、`report.md`，以及预测、轨迹和消息记录。多输入分别写入子目录；缺失评分保留 `null`。

训练后的本地模型需先部署为 OpenAI 兼容服务，再参考 `configs/local_openai.yaml` 配置接入；该配置本身不启动模型服务。

## 离线检查

审核脚本测试在项目根目录执行；以下检查不调用付费 API：

```bash
python -m unittest -q test_review_t1234
python -m unittest discover -s ScopeCrit_Consistency -q
cd ScopeCrit && python -m pytest -q
```

训练工程检查在项目根目录、训练环境中执行：`python -m pytest -q ScopeSFT ScopeCT`。

# 本次交付验证

- Python 3.12.14；依赖已实际安装，`pip check` 无冲突。
- `python -m pytest -q`：**112 passed**。
- `validate / run / export / inspect / resume / evaluate` 均通过离线验收。
- 人工合成演示：T1–T4 四条全部 COMPLETED，共 35 次 Mock 调用，T4 两条消息实际投递。
- 无语义对齐时，语义指标为 null；提供演示人工对齐后，各任务预期指标通过。
- 已验证已完成运行续跑不新增调用；调用响应复用、提交事务恢复、遗留调用暂停及显式重试均有测试。
- 已验证 gold、未来材料、同伴私有资料及额外证据元数据的请求隔离。
- 真实 `CritPred_test.json`：372 条输入校验通过。
- 真实 `CritPred_train.json`：813 条中四条 CritPred48:T1–T4 的证据编号不连续，按规范隔离。
- 原始数据与技术文档未修改。
- `max_samples=None / 521`、命令行覆盖、先截取后校验、所选范围的恢复与评分均有回归测试。
- 直接运行 `run.py` 不要求安装 editable 项目；默认 DeepSeek 配置和自动输出目录已验证。
- 使用真实 OpenAI SDK 配合本地 HTTP 模拟，验证只填写 JSON 路径和 Key 即可衔接 T1–T4 的全部 35 次请求；没有真实付费调用。
- API 400/401/403 首次失败停止批次并返回非零退出码；测试验证不会对整批重复无效调用。

**未验证：真实 DeepSeek API、账户模型/参数兼容性、模型文学理解和真实数据上的预测质量。**
本次未调用付费 API、未发布云环境、未把 Mock 演示分数当作实验结论。
Conda 与清华镜像命令是便捷配置说明，本次实际使用 venv 和默认 Python 软件源安装。

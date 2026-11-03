# 验证范围与交付限制

本次检查针对现有 ScopeCrit T1–T4 执行和评分，并按最新要求调整为 DeepSeek 答题、Qwen/MiniMax/Gemini 三裁判。首版技术方案中“不默认额外调用模型”的评测约定，由用户后续明确要求的裁判流程扩展；`--no-judge` 仍保留纯答题与程序指标模式。

## 可复现检查命令

```bash
python -m pytest -q
python run.py --input examples/merged_demo.json --config configs/mock.yaml --run-dir runs/demo --no-judge
python -m scopecrit inspect --run-dir runs/demo
```

2026-11-03 本次实测：`238 passed`（Python 3.12，按 `requirements-lock.txt` 安装，并以 editable 方式安装项目）。`pip check` 无依赖冲突。命令行 Mock 演示 T1–T4 均为 `COMPLETED`。

真实 OpenAI SDK 配合 `httpx.MockTransport` 验证了 DeepSeek 与本地 Qwen 两种答题后端的完整流程：每种后端处理四个演示任务，各发生 35 次答题阶段请求与 12 次三裁判请求。双文件测试验证独立缓存与同名样本隔离。这些是模拟响应的工程验证，不是模型实测成绩。

修复了尚无主张时的中断状态被误计为证据答卷、Windows UTF-8 BOM 输入读取不一致，以及后续文件的非法 JSON 常量未在批次开始前被拒绝的问题。评分前检查原状态库，缺失时不创建空库；模型不存在、认证失败和无效参数会停止对应裁判批次。

## 检查项目

| 范围 | 应验证行为 |
|---|---|
| 数据输入 | 一个或多个 JSON 数组；非法条目隔离；重复 ID 与 Unicode 证据偏移 |
| 条数控制 | null/None 为全部；正整数每文件按原顺序截取；不补齐非法条目；T3 轮次完整 |
| 本地配置 | Key 直接填私有 YAML；绝对路径与配置相对路径；凭据不进入报表和清单 |
| T1/T2 | 候选经支持核对提交；不同范围可共存；T2 跨片段关系解释 |
| T3 | 新证据逐轮公开；累计记忆；未来材料不进入未授权请求 |
| T4 | 初始资料隔离；实际消息投递；幂等；双方资料依赖 |
| 恢复 | 响应复用；提交与通信恢复；未知请求暂停；继承原范围；拒绝指纹变化 |
| 三裁判 | 独立一对一对齐；配对后计算；三裁判完整才给正式均值；缺失为 null |
| 报表 | 每文件独立；逐样本、逐任务及整体指标和覆盖率；保留原始指标 |
| 适配器 | 参数隔离；本地 OpenAI 兼容模型；可选按阶段指定 provider |
| 失败处理 | 400/401/403/404/422 不盲重试；网络/格式失败保留；非零退出码表示未完成 |

测试使用预设的 SYNTHETIC fixture 和本地 HTTP 模拟。Mock 不读取 gold 生成预测；Gold 只进入评分。请求隔离检查应覆盖所有阶段，不能只检查最终答案。

## 已确认的外部接口资料

- [APIKEY.FAN 接入文档](https://apikey.fan/docs/api-scripts)说明 Chat Completions 路径为 `/v1/chat/completions`，SDK 基址为 `https://api.apikey.fan/v1`，使用 Bearer API Key。
- [百炼接口地址](https://help.aliyun.com/en/model-studio/base-url)确认北京业务空间基址包含 WorkspaceId 和 `/compatible-mode/v1`。
- [Qwen3.8-Flash 官方说明](https://help.aliyun.com/en/model-studio/qwen3-8-flash)确认模型 ID；[结构化输出说明](https://help.aliyun.com/en/model-studio/qwen-structured-output)列出 JSON 输出能力。

## 尚未证明的内容

本次未进行真实付费调用，未核实用户账号的网络、余额、配额、权限、实际费用或服务稳定性。APIKEY.FAN 中 `minimax-m3.1-flash`、`gemini-3.1-flash-lite` 两个精确文本模型别名按用户要求保留，其账号可用性没有通过公开目录或真实请求确认；程序不会自动换成其他模型。

本地 HTTP 模拟可以证明适配、缓存、恢复和指标路径符合测试条件，不能证明文学理解、模型裁判可靠性或任意真实接口都能运行。未宣称代码不存在任何 bug，未把 Mock 分数或 SDK 测试当作论文实验结果。数据语义质量不由结构校验自动证明。

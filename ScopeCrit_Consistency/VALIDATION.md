# 交付验证记录

- 环境：Python 3.12.14 / Linux；工具要求Python 3.10+。
- 命令：`python -m unittest discover -v`
- 结果：67项测试全部通过，耗时约11.5秒。
- 双文件校验：`python consistency_check.py --input example_1.json example_2.json --check-only --output-dir demo_validation`，四条T1–T4合成样本有效，无API请求。
- 统计：手算ordinal α与Fleiss κ、负系数、缺失评级、常数/空集分母退化；另经独立公式实现核对各1000组随机α/κ输入，误差小于1e-12。
- HTTP/CLI：仅localhost模拟服务；验证三个模型独立请求、两文件限条、成功评分复用、失败补评、401停批、类型/格式异常、截断用量保存、脱敏、指纹变化拒绝续跑及分任务输出。
- 报告：合并及各输入文件分别保留T1–T4，CSV/JSON合计12行宽表，每行含五维κ/α；report.md按任务/维度展开合计60行；空输入文件仍保留各任务各维度并注明统计量未定义。
- 未进行真实付费API调用，未使用用户密钥；交付config.json三个api_key均为空。
- Windows专用文件锁分支尚未在Windows执行；真实账号、模型权限、网关兼容性与限流需配置后验证。

## v2 五维κ补充验证

- 五维Fleiss κ将1–5分视为五个名义类别，保留ordinal α；原标签κ另列label_kappa。
- 新增五分类手算、负κ、缺失/常数/严格类型验证，并独立对照1000组随机五分类数据。
- 端到端例中五维κ=5/11、ordinal α=7/8，而标签κ=1，证明报表未混用两种κ。
- 使用旧版代码生成的SQLite checkpoint，在新版运行report-only成功；API请求数保持不变，未重新评分。

## v3 单/多文件输入修复

- CLI支持一个或多个JSON、可重复使用--input；配置支持单路径字符串、路径列表与input_file别名。
- 空路径占位自动忽略；CLI可覆盖未填写的配置路径；无路径/不存在路径给出明确错误。
- 单文件评分、限条、无Key报告恢复、三文件合并及原双文件checkpoint继续处理均通过本地模拟接口测试。
- 多文件来源与统计表使用实际文件数量，保留每任务五维κ/α；未修改冻结评分Rubric。

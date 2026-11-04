# ScopeCT：范围关键分歧的单阶段训练

本实现依据用户提供的《ScopeCT_单阶段训练机制方案》，起点为 **Qwen/Qwen3-1.7B、non-thinking**。它和 [ScopeSFT](../ScopeSFT/README.md) 使用同一份 ACTION 正例、同一 LoRA 配置与默认训练预算，单独从原始模型开始。**不要先做 SFT 再把输出传给本实验**，否则不再是文档中的公平单阶段对照。

## 实际使用的方法

1. **离线教师轨迹与监督微调**：从 CultCrit T1–T4 标签构造合法 ACTION，完整正例进行 response-only causal NLL。
2. **LoRA 参数高效更新**：只训练注意力和 MLP 的低秩适配器，完成后合并导出完整权重。
3. **原文锚定的困难负例**：保持相同序列化、方面和证据，只修改一个范围、声音或条件因素。记录自动规则及原文引句；无法证明的样本仅保留正例监督。
4. **首个有效语义 token 分歧约束**：精确检查共同 token 前缀，只在指定语义值的首次分歧上施加额外梯度。
5. **单阶段、无参考模型的联合目标**：一次正例前向同时取得正确与错误 token logits，无完整负例前向、无额外 DPO 阶段、无在线 rollout。

ORPO、关键 token 偏好优化、unlikelihood 等是文档讨论的相关机制和比较对象；本代码没有把它们作为额外阶段串联，也不宣称实现完整 ORPO/DPO 或已经证明新颖性。

公式与原文一致：

```text
L_i = mean_response_tokens[-log p(y+_j | x,y+_<j)]
      + λ * I(valid_pair_i) * softplus(m + z_t(v) - z_t(u))
L_batch = mean_examples(L_i)
```

默认 `λ=0.2`、`m=1.0`。每个训练样本只有一个 token 分歧；其余正例 token 仍受完整监督。不存在有效对时该样本的附加项为零；批次按全部样本平均，避免只按本卡负例数归一化导致 DDP 权重偏差。首次分歧后历史已经不同，**不能继续用正例 logits 计算负例后续概率**，代码没有这样做。

## 运行

环境与 TITAN RTX / 470 驱动注意事项见 [ScopeSFT 安装说明](../ScopeSFT/README.md)。两套代码只安装一次：

```bash
cd /你的路径/CultCrit-Bench
bash ScopeCT/install.sh
source ~/.venvs/scope-qwen3/bin/activate
python ScopeCT/train.py
# 只需按提示输入 /data/train_optimized_validated.json
```

默认四卡 `0123`，自动下载原始模型、转换数据、训练、合并导出。也可明确指定参数：

```bash
python ScopeCT/train.py --data /data/train_optimized_validated.json --gpus 0123
python ScopeCT/train.py --data /data/train_optimized_validated.json --gpus 02 \
  --ct-weight 0.2 --margin 1.0 --output /data/weights/ct_run1
```

不需要手动制作偏好 JSON，也不调用付费模型构造负例。默认输出为 `ScopeCT/outputs/<唯一运行目录>/final_model`，这是已经合并的完整权重。另保存 `adapter`、训练日志、规则依据、token 分歧与最终权重哈希。完整参数、长度限制和文件说明见 ScopeSFT README。

## 自动负例的范围与边界

当前保守规则覆盖：

- 引文明确只评价已刊两段，正例相应限定前两段：将正例文字和范围同步扩大到全书。
- 原文明确使用“预约券/该是”，正例保留“预期”：将其同步改成已经确定的事实。
- 原文先具名引入某人的文章并紧接引文，正例明确说明这是转引而非本文叙述者评价：同步改成本文叙述者直接断言，并移除与错误归属冲突的免责声明。

只有满足规则所需源证据条件才生成；不会随机翻转立场，不能以“局部永远对、整体永远错”为训练规则。自动规则对**不是人工已审核的金标准偏好标签**，即使引用了原文，也仍须检查 `training_pairs.jsonl` 的语义可靠性和类别偏差。其余正例回退 NLL；整个训练集若一个有效 token 对都没有，默认报错，绝不悄悄把 SFT 称为 ScopeCT。仅在明确需要时使用 `--allow-sft-fallback`。

用户真实训练集抽取的 T1–T4 四条样本，共生成 58 条正例，包含 2 组有源证据的声音归属负例，来自同一主张在 T1/T3 的出现；不能将它们当成两条独立的原始语义证据，也不能把该样本覆盖率推断为全量覆盖率。

有经过数据作者审核的额外负例时，可选地在原始样本增加字段（不是运行必需）：

```json
"scope_ct_negatives": [{
  "claim_id": "C1",
  "reviewed": true,
  "semantic_field": "text",
  "justification": "原文哪些文字证明该范围/声音/条件错误",
  "evidence_ids": ["E001"],
  "rejected_claim": {
    "text": "经审核的错误主张",
    "aspect": "NARRATION",
    "target_scope": "错误范围",
    "stance": "POSITIVE",
    "evidence_ids": ["E001"]
  }
}]
```

字段名和证据 ID 仅为结构示例，必须与原正例对应。`reviewed` 是数据作者的声明，代码不会替你作独立人工核验。`semantic_field` 允许 `text/target_scope/stance`，固定 JSON 字段顺序和空白，并检查正负字符串的首次字符分歧、token 分歧均落在对应语义值中。格式、字段顺序、标点或证据编号的差异不会作为有效 CT 信号。

## 检查与实验解释

```bash
python ScopeCT/train.py --data /data/train.json --validate-only
python ScopeCT/train.py --data /data/train.json --dry-run
python -m pytest -q ScopeSFT ScopeCT
python -m ScopeSFT.smoke_test
```

`validate-only` 不下载模型；`dry-run` 用实际 tokenizer 检查长度、共同前缀、语义位置和有效对数。训练和检查须各用新的输出目录。

微型 Qwen3 CPU 工程测试已验证联合损失激活、梯度及 LoRA 实际更新、完整权重导出重载；两进程 CPU/Gloo 路径也通过。尚未实测用户 TITAN RTX、真实 Qwen3-1.7B 的完整训练时长、显存峰值或文学能力提升。论文实验应固定基座/正例/预算/ScopeCrit 执行器和独立测试集，比较原模型、SFT、ScopeCT；训练损失下降不等于裁判认可或下游效果改善。

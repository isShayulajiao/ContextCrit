"""输入只读校验：结构错误隔离到样本，共享正文冲突隔离到整个文档。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ASPECTS = frozenset("THEME SOCIAL CHARACTER PLOT STRUCTURE LANGUAGE NARRATION EMOTION STYLE RECEPTION OVERALL OTHER".split())
STANCES = frozenset("POSITIVE NEGATIVE MIXED NEUTRAL".split())
TASKS = frozenset(("T1", "T2", "T3", "T4"))
ROLES = frozenset(("archivist", "interpreter"))
CITATION = re.compile(r"\[(E\d+)\]")


def fingerprint(obj: Any) -> str:
    """只排序对象键，不改变数组顺序、Unicode、空白或正文内容。"""
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_key(row: Any, index: int) -> str:
    """无法取得合法 sample_id 时，仍给校验报告分配稳定的位置键。"""
    if isinstance(row, dict) and isinstance(row.get("sample_id"), str) and row["sample_id"]:
        return row["sample_id"]
    return f"__invalid_{index}"


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _strings(value: Any, *, nonempty: bool = False) -> bool:
    return isinstance(value, list) and (bool(value) or not nonempty) and all(_nonempty(x) for x in value)


def _unique_strings(value: Any, *, nonempty: bool = False) -> bool:
    return _strings(value, nonempty=nonempty) and len(value) == len(set(value))


def _reject_nonfinite(value: str) -> None:
    """Python JSON 默认接受 NaN/Infinity，数据协议使用严格 JSON。"""
    raise ValueError(f"非法 JSON 常量: {value}")


def _validate_row(row: Any, fixture: bool, warnings: list[str], key: str) -> list[str]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    if not isinstance(row, dict):
        return ["样本必须是 JSON 对象"]
    for field in ("sample_id", "doc_id", "question", "generation_method"):
        require(_nonempty(row.get(field)), f"{field} 必须是非空字符串")
    require(row.get("task") in TASKS if isinstance(row.get("task"), str) else False, "task 必须是 T1–T4")
    require(row.get("sample_id") == f"{row.get('doc_id')}:{row.get('task')}", "sample_id 必须等于 doc_id:task")
    require(row.get("human_verified") is False, "human_verified 必须为 false")
    require("query" not in row, "不允许旧字段 query")
    require(row.get("material_origin") == "AUTHENTIC" or (fixture and row.get("material_origin") == "SYNTHETIC"), "material_origin 必须为 AUTHENTIC；SYNTHETIC 仅限 fixture 模式")
    expected_origin = "CONTROLLED" if row.get("task") in ("T3", "T4") else "DERIVED"
    require(row.get("task_origin") == expected_origin, f"task_origin 必须为 {expected_origin}")
    if "split" in row:
        require(isinstance(row["split"], str), "split 必须是字符串；程序不使用它重新划分")

    doc = row.get("document")
    if not isinstance(doc, dict):
        return errors + ["document 必须是对象"]
    require(doc.get("doc_id") == row.get("doc_id"), "document.doc_id 与样本不一致")
    require(_nonempty(doc.get("work_title")), "document.work_title 必须非空")
    require(_nonempty(doc.get("text")), "document.text 必须非空")
    text = doc.get("text") if isinstance(doc.get("text"), str) else ""
    evidence = doc.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return errors + ["document.evidence 必须是非空数组"]
    evidence_ids: set[str] = set()
    previous_start = -1
    previous_end = -1
    for i, item in enumerate(evidence, 1):
        label = f"document.evidence[{i - 1}]"
        if not isinstance(item, dict):
            errors.append(f"{label} 必须是对象")
            continue
        eid = item.get("evidence_id")
        require(eid == f"E{i:03d}", f"{label}.evidence_id 必须按 E001 连续编号")
        if isinstance(eid, str):
            require(eid not in evidence_ids, f"{label} 的证据 ID 重复")
            evidence_ids.add(eid)
        require(_nonempty(item.get("text")), f"{label}.text 必须非空")
        a, b = item.get("start"), item.get("end")
        offsets_ok = type(a) is int and type(b) is int and 0 <= a < b <= len(text)
        require(offsets_ok, f"{label} 的偏移必须为有效严格整数（不能是 bool）")
        if offsets_ok:
            require(text[a:b] == item.get("text"), f"{label} 的证据文字与正文偏移不一致")
            require(a >= previous_start, f"{label} 未按原文位置排序")
            if a < previous_end:
                warnings.append(f"{key}: {label} 与前一条跨度重叠；仅警告")
            previous_start, previous_end = a, b

    gold = row.get("gold")
    if not isinstance(gold, dict):
        return errors + ["gold 必须是对象"]
    claims = gold.get("claims")
    if not isinstance(claims, list) or not claims:
        return errors + ["gold.claims 必须是非空数组"]
    gold_by_id: dict[str, dict] = {}
    union: set[str] = set()
    for i, claim in enumerate(claims):
        label = f"gold.claims[{i}]"
        if not isinstance(claim, dict):
            errors.append(f"{label} 必须是对象")
            continue
        for field in ("claim_id", "text", "target_scope"):
            require(_nonempty(claim.get(field)), f"{label}.{field} 必须非空")
        cid = claim.get("claim_id")
        if _nonempty(cid):
            require(cid not in gold_by_id, f"{label} 的 claim_id 重复")
            gold_by_id[cid] = claim
        require(isinstance(claim.get("aspect"), str) and claim["aspect"] in ASPECTS, f"{label}.aspect 非法")
        require(isinstance(claim.get("stance"), str) and claim["stance"] in STANCES, f"{label}.stance 非法")
        require("holder" not in claim, f"{label} 不允许 holder 字段；声音写在 text 中")
        refs = claim.get("evidence_ids")
        require(_unique_strings(refs, nonempty=True), f"{label}.evidence_ids 必须是非空、不重复字符串数组")
        if _strings(refs):
            require(set(refs) <= evidence_ids, f"{label} 引用了不存在的证据")
            union.update(refs)
    required = gold.get("required_evidence_ids")
    require(_unique_strings(required, nonempty=True), "gold.required_evidence_ids 必须是非空、不重复字符串数组")
    if _strings(required):
        require(set(required) == union, "gold.required_evidence_ids 不等于主张证据并集")
    answer = gold.get("reference_answer")
    require(_nonempty(answer), "gold.reference_answer 必须非空")
    if isinstance(answer, str):
        require(set(CITATION.findall(answer)) == union, "gold.reference_answer 引用集合不等于主张证据并集")

    task = row.get("task")
    events, roles, exchanges = row.get("events"), row.get("roles"), row.get("required_exchanges")
    require(isinstance(events, list), "events 必须是数组")
    require(isinstance(roles, dict), "roles 必须是对象")
    require(isinstance(exchanges, list), "required_exchanges 必须是数组")
    if task in ("T1", "T2"):
        require(events == [] and roles == {} and exchanges == [], "T1/T2 的 events、roles、required_exchanges 必须为空")
    elif task == "T3":
        require(roles == {} and exchanges == [], "T3 的 roles 和 required_exchanges 必须为空")
        if not isinstance(events, list):
            return errors
        require(2 <= len(events) <= 4, "T3 必须有 2–4 轮")
        visible: set[str] = set()
        previous_memory: set[str] = set()
        has_expand = False
        for i, event in enumerate(events, 1):
            label = f"events[{i - 1}]"
            if not isinstance(event, dict):
                errors.append(f"{label} 必须是对象")
                continue
            require(type(event.get("turn")) is int and event["turn"] == i, f"{label}.turn 必须从 1 连续编号")
            require(_nonempty(event.get("question")), f"{label}.question 必须非空")
            op = event.get("expected_operation")
            require(op == "ADD" if i == 1 else op in ("EXPAND", "DEFER"), f"{label}.expected_operation 非法")
            has_expand |= op == "EXPAND"
            new = event.get("visible_evidence_ids")
            require(_unique_strings(new), f"{label}.visible_evidence_ids 必须是不重复字符串数组")
            if _strings(new):
                require(set(new) <= evidence_ids, f"{label} 公开了不存在的证据")
                require(not visible.intersection(new), f"{label} 重复公开证据")
                visible.update(new)
            memory = event.get("gold_memory_state")
            require(_unique_strings(memory, nonempty=True), f"{label}.gold_memory_state 必须是非空、不重复字符串数组")
            if _strings(memory):
                state = set(memory)
                require(state <= gold_by_id.keys(), f"{label} 引用了不存在的 gold 主张")
                require(previous_memory <= state, f"{label} 删除了旧有效主张")
                if i > 1 and op == "EXPAND":
                    require(state > previous_memory, f"{label} EXPAND 必须增加主张")
                if i > 1 and op == "DEFER":
                    require(state == previous_memory, f"{label} DEFER 必须保留原有主张")
                refs: set[str] = set()
                for cid in state & gold_by_id.keys():
                    claim_refs = gold_by_id[cid].get("evidence_ids")
                    if _strings(claim_refs):
                        refs.update(claim_refs)
                require(refs <= visible, f"{label} 的标准主张引用了尚未公开的证据")
                event_answer = event.get("reference_answer")
                require(_nonempty(event_answer), f"{label}.reference_answer 必须非空")
                if isinstance(event_answer, str):
                    require(set(CITATION.findall(event_answer)) == refs, f"{label}.reference_answer 引用不等于本轮标准主张证据并集")
                previous_memory = state
        require(has_expand, "T3 至少包含一次 EXPAND")
        require(previous_memory == gold_by_id.keys(), "T3 最后一轮必须覆盖全部 gold 主张")
    elif task == "T4":
        require(events == [], "T4 的 events 必须为空")
        if not isinstance(roles, dict):
            return errors
        require(set(roles) == ROLES, "T4 必须且仅有 archivist、interpreter 两个角色")
        allocation: dict[str, set[str]] = {}
        for role in sorted(ROLES):
            assigned = roles.get(role)
            require(_unique_strings(assigned, nonempty=True), f"roles.{role} 必须是非空、不重复字符串数组")
            allocation[role] = set(assigned) if _strings(assigned) else set()
            require(allocation[role] <= evidence_ids, f"roles.{role} 包含不存在的证据")
            require(bool(allocation[role] & union), f"roles.{role} 必须持有必要证据")
        require(not allocation["archivist"] & allocation["interpreter"], "T4 双方初始证据必须互斥")
        require(union <= allocation["archivist"] | allocation["interpreter"], "T4 必要证据未分配给双方")
        if isinstance(exchanges, list):
            require(bool(exchanges), "T4 required_exchanges 必须非空")
            seen_exchanges: set[tuple] = set()
            for i, exchange in enumerate(exchanges):
                label = f"required_exchanges[{i}]"
                if not isinstance(exchange, dict):
                    errors.append(f"{label} 必须是对象")
                    continue
                source, target, eid = (exchange.get(x) for x in ("from_agent", "to_agent", "evidence_id"))
                direction_ok = isinstance(source, str) and isinstance(target, str) and source in ROLES and target in ROLES and source != target
                require(direction_ok, f"{label} 交换方向非法")
                if direction_ok:
                    require(isinstance(eid, str) and eid in allocation[source], f"{label} 的证据不属于发送方")
                require(_nonempty(exchange.get("reason")), f"{label}.reason 必须非空")
                identity = (source, target, eid)
                if all(isinstance(x, str) for x in identity):
                    require(identity not in seen_exchanges, f"{label} 重复预期交换")
                    seen_exchanges.add(identity)
    return errors


def load_dataset(path: str | Path, fixture: bool = False, max_samples: int | None = None) -> tuple[list, dict]:
    """读取一个 JSON 数组。不修改数据，不训练模型，也不划分 train/test。

    max_samples=None 读取全部；正整数只处理输入顺序的前 N 条（包含无效条目）。
    先选择，再校验选中范围，未选择的尾部不进入运行或评分。
    顶层无法读取、不是数组、选中范围重复 sample_id 为整次运行错误；
    其他错误写入 report.invalid，调用者仍可运行未被隔离的样本。
    """
    if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
        raise ValueError("max_samples 必须是正整数或 None")
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"无法读取 JSON 数据集: {exc}") from exc
    if not isinstance(rows, list):
        raise ValueError("数据集顶层必须是 JSON 数组")
    source_total = len(rows)
    if max_samples is not None:
        rows = rows[:max_samples]
    ids: set[str] = set()
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("sample_id"), str):
            if row["sample_id"] in ids:
                raise ValueError(f"sample_id 重复: {row['sample_id']}")
            ids.add(row["sample_id"])
    # 极端情况下真实 ID 与报告占位键冲突，避免错误被覆盖。
    if any(sample_key(row, i) in ids for i, row in enumerate(rows) if not isinstance(row, dict) or not _nonempty(row.get("sample_id"))):
        raise ValueError("sample_id 与非法样本的报告位置键冲突")
    warnings: list[str] = []
    invalid: dict[str, list[str]] = {}
    by_doc: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    by_text: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for i, row in enumerate(rows):
        key = sample_key(row, i)
        errors = _validate_row(row, fixture, warnings, key)
        if errors:
            invalid[key] = errors
        if isinstance(row, dict) and isinstance(row.get("document"), dict) and _nonempty(row.get("doc_id")):
            doc = row["document"]
            by_doc[row["doc_id"]].append((key, doc))
            if isinstance(doc.get("text"), str) and isinstance(doc.get("work_title"), str):
                by_text[doc["text"]].append((key, row))
    for doc_id, group in by_doc.items():
        materials = [{k: d.get(k) for k in ("work_title", "text", "evidence")} for _, d in group]
        if len({fingerprint(material) for material in materials}) > 1:
            for key, _ in group:
                invalid.setdefault(key, []).append(f"共享 doc_id={doc_id} 的材料不一致；隔离该文档全部任务")
    for group in by_text.values():
        titles = {row["document"]["work_title"] for _, row in group}
        if len(titles) > 1:
            for key, row in group:
                invalid.setdefault(key, []).append("相同正文对应不同作品名；隔离相关文档")
                # 将同文档其他正文冲突的条目也纳入隔离。
                for related_key, _ in by_doc[row["doc_id"]]:
                    message = "相同正文对应不同作品名；隔离相关文档"
                    if message not in invalid.setdefault(related_key, []):
                        invalid[related_key].append(message)
        elif len({row["doc_id"] for _, row in group}) > 1:
            warnings.append("同正文同书名存在多个 doc_id: " + ", ".join(sorted({row["doc_id"] for _, row in group})))
    return rows, {"valid": not invalid, "total": len(rows), "source_total": source_total,
                  "max_samples": max_samples, "invalid": invalid, "warnings": warnings}

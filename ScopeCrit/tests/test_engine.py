"""人工 SYNTHETIC 样例检验控制协议；不衡量文学语义能力，不访问网络。"""

from copy import deepcopy
import json

import pytest

from scopecrit.config import Config
from scopecrit.engine import Engine, ToolError
from scopecrit.storage import Store


def action(name, **arguments):
    return {"action": name, "arguments": arguments, "reason": "人工测试动作"}


def claim(eid="E001", text="整体结构严谨"):
    return {"text": text, "aspect": "STRUCTURE", "target_scope": "整体",
            "stance": "POSITIVE", "evidence_ids": [eid]}


def check(cid="P001", eid="E001", support="SUPPORTED"):
    return {"candidate_id": cid, "support": support, "scope_consistent": True,
            "voice_preserved": True, "modality_preserved": True,
            "evidence_ids": [eid], "reason": "人工预设支持结果"}


def relation(cid="P001", eid="E001"):
    return {"claim_ids": [cid], "evidence_ids": [eid], "relation_type": "LIMITS",
            "explanation": "人工预设范围关系"}


def sample(task="T1", identifier=None):
    first, second = "VISIBLE_FIRST", "PRIVATE_FUTURE_CANARY"
    text = first + second + "FULL_DOCUMENT_ONLY_CANARY"
    return {"sample_id": identifier or f"Synthetic:{task}", "doc_id": "Synthetic",
            "task": task, "question": "评价有什么范围限制？", "material_origin": "SYNTHETIC",
            "document": {"doc_id": "Synthetic", "work_title": "人工测试作品", "text": text,
                         "evidence": [
                             {"evidence_id": "E001", "text": first, "start": 0, "end": len(first)},
                             {"evidence_id": "E002", "text": second, "start": len(first),
                              "end": len(first + second)}]},
            "gold": {"claims": [{"text": "GOLD_SECRET_CANARY"}]},
            "events": [
                {"turn": 1, "question": "当前已知评价？", "visible_evidence_ids": ["E001"],
                 "reference_answer": "GOLD_TURN_CANARY", "gold_memory_state": ["C1"]},
                {"turn": 2, "question": "新证据如何补充？", "visible_evidence_ids": ["E002"],
                 "reference_answer": "GOLD_FUTURE_CANARY", "gold_memory_state": ["C1", "C2"]}],
            "roles": {"archivist": ["E001"], "interpreter": ["E002"]},
            "required_exchanges": [{"reason": "GOLD_EXCHANGE_CANARY"}]}


class ScriptProvider:
    def __init__(self, rows):
        self.rows = rows
        self.requests = []

    def complete(self, stage, request, call_id):
        self.requests.append((stage, deepcopy(request), call_id))
        expected, response = self.rows[len(self.requests) - 1]
        assert expected == stage
        return {"content": response if isinstance(response, str) else json.dumps(response, ensure_ascii=False),
                "request_id": call_id, "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                "latency_ms": 0, "provider_name": "test", "model_name": "fixture", "finish_reason": "stop"}


def engine(tmp_path, rows, task="T1", config=None):
    store = Store(tmp_path)
    provider = ScriptProvider(rows)
    instance = Engine(sample(task), config or Config().model_dump(), provider, store, log=lambda _: None)
    return instance, provider, store


def test_t1_supported_commit_and_completed_cached_response_not_reissued(tmp_path):
    rows = [("ACTION", action("PROPOSE", claims=[claim()])),
            ("ACTION", action("COMMIT", candidate_id="P001")),
            ("SUPPORT_CHECK", check()),
            ("ACTION", action("FINALIZE", claim_ids=["P001"], answer="结构严谨[E001]"))]
    first, provider, store = engine(tmp_path, rows)
    first.initialize()
    # 模拟模型响应已入库，但尚未执行动作便退出。
    response = first.call("ACTION", first.build_view(), "0")
    assert json.loads(response)["action"] == "PROPOSE"
    recovered = Engine(sample(), first.config, provider, store, log=lambda _: None)
    output = recovered.run()
    assert output["status"] == "COMPLETED"
    assert len(provider.requests) == 4
    assert output["usage"]["model_calls"] == 4
    assert output["usage"]["total_tokens"] == 20
    assert len(output["prediction"]["claims"]) == 1
    assert len(store.list_events()) == 4  # INITIALIZED + 三个实际动作


def test_partial_support_cannot_commit(tmp_path):
    rows = [("ACTION", action("PROPOSE", claims=[claim()])),
            ("ACTION", action("COMMIT", candidate_id="P001")),
            ("SUPPORT_CHECK", check(support="PARTIAL")),
            ("ACTION", action("FINALIZE", claim_ids=["P001"], answer="错误提交[E001]"))]
    cfg = Config().model_dump()
    cfg["limits"]["max_actions_per_episode"] = 3
    runner, _, _ = engine(tmp_path, rows, config=cfg)
    output = runner.run()
    assert output["status"] == "PARTIAL"
    assert output["prediction"]["claims"] == []
    assert runner.role["candidates"]["P001"]["status"] == "DEFERRED"


def test_model_budget_includes_support_check_and_keeps_partial(tmp_path):
    cfg = Config().model_dump()
    cfg["limits"]["max_model_calls_per_episode"] = 2
    runner, provider, _ = engine(tmp_path, [
        ("ACTION", action("PROPOSE", claims=[claim()])),
        ("ACTION", action("COMMIT", candidate_id="P001"))], config=cfg)
    output = runner.run()
    assert output["termination_reason"] == "BUDGET_EXHAUSTED"
    assert output["status"] == "PARTIAL"
    assert output["prediction"]["claims"] == []
    assert len(provider.requests) == 2


def test_format_repair_is_counted_as_real_call(tmp_path):
    cfg = Config().model_dump()
    cfg["limits"]["max_actions_per_episode"] = 1
    runner, provider, _ = engine(tmp_path, [
        ("ACTION", "malformed"),
        ("ACTION", action("READ", evidence_ids=["E001"]))], config=cfg)
    output = runner.run()
    assert len(provider.requests) == output["usage"]["model_calls"] == 2
    assert "format_repair" in provider.requests[1][1]
    assert output["termination_reason"] == "ACTION_BUDGET_EXHAUSTED"


def test_unknown_or_future_read_share_one_denial(tmp_path):
    runner, _, _ = engine(tmp_path, [], task="T3")
    runner.initialize()
    codes = []
    for eid in ("E002", "E999"):
        with pytest.raises(ToolError) as caught:
            runner.execute(action("READ", evidence_ids=[eid]))
        codes.append((caught.value.code, caught.value.message))
    assert codes[0] == codes[1]
    assert set(runner.role["visible"]) == {"E001"}


def test_t3_all_stages_exclude_gold_full_text_and_future_before_release(tmp_path):
    rows = [
        ("ACTION", action("PROPOSE", claims=[claim()])),
        ("ACTION", action("COMMIT", candidate_id="P001")),
        ("SUPPORT_CHECK", check()),
        ("ACTION", action("COMPARE", claim_ids=["P001"], evidence_ids=["E001"], question="评价范围？")),
        ("SCOPE_COMPARE", relation()),
        ("ACTION", action("FINALIZE", claim_ids=["P001"], relation_ids=["R001"], answer="整体评价[E001]")),
        ("ACTION", action("PROPOSE", claims=[claim("E002", "结尾仓促")])),
        ("ACTION", action("COMMIT", candidate_id="P002")),
        ("SUPPORT_CHECK", check("P002", "E002")),
        ("ACTION", action("FINALIZE", claim_ids=["P001", "P002"], answer="整体与局部[E001][E002]"))]
    runner, provider, _ = engine(tmp_path, rows, task="T3")
    output = runner.run()
    assert output["status"] == "COMPLETED"
    assert [turn["operation"] for turn in output["turns"]] == ["ADD", "EXPAND"]
    assert [len(turn["claims"]) for turn in output["turns"]] == [1, 2]
    for stage, request, _ in provider.requests:
        serialized = json.dumps(request)
        assert "GOLD_" not in serialized
        assert "FULL_DOCUMENT_ONLY_CANARY" not in serialized
        assert "document_text" not in request
    for _, request, _ in provider.requests[:6]:
        assert "PRIVATE_FUTURE_CANARY" not in json.dumps(request)
    assert "PRIVATE_FUTURE_CANARY" in json.dumps(provider.requests[6][1])


def test_t4_private_material_not_in_any_archivist_stage_and_outbox_not_visible(tmp_path):
    rows = [("ACTION", action("PROPOSE", claims=[claim()])),
            ("ACTION", action("COMMIT", candidate_id="A:P001")),
            ("SUPPORT_CHECK", check("A:P001")),
            ("ACTION", action("COMPARE", claim_ids=["A:P001"], evidence_ids=["E001"], question="范围？")),
            ("SCOPE_COMPARE", relation("A:P001")),
            ("ACTION", action("SEND_EVIDENCE", evidence_ids=["E001"], purpose="提供依据")),
            ("ACTION", action("FINALIZE", claim_ids=[], answer="未形成主张"))]
    runner, provider, _ = engine(tmp_path, rows, task="T4")
    runner.initialize()
    runner.execute(action("SEND_EVIDENCE", evidence_ids=["E001"], purpose="提供依据", request_id=None))
    assert set(runner.state["role_states"]["interpreter"]["visible"]) == {"E002"}
    runner.state["messages"] = []  # 上面的直接工具探针不入 checkpoint；真实运行从空 outbox 开始。
    output = runner.run()
    assert output["status"] == "COMPLETED"
    assert output["messages"][0]["delivered_event_seq"] is not None
    for _, request, _ in provider.requests[:6]:
        serialized = json.dumps(request)
        assert "PRIVATE_FUTURE_CANARY" not in serialized
        assert "GOLD_" not in serialized
        assert "FULL_DOCUMENT_ONLY_CANARY" not in serialized
    final_view = provider.requests[6][1]
    assert {e["evidence_id"] for e in final_view["visible_evidence"]} == {"E001", "E002"}


def test_uncertain_call_pauses_without_contacting_provider(tmp_path):
    runner, provider, store = engine(tmp_path, [])
    runner.initialize()
    runner.state["usage"]["model_calls"] = 1
    runner.state["call_keys"]["0:ACTION:0:0"] = "Synthetic:T1:0001"
    store.save_state(runner.state["sample_id"], runner.state)
    store.start_call("Synthetic:T1:0001", "Synthetic:T1", "ACTION", runner.build_view())
    resumed = Engine(sample(), runner.config, provider, store, log=lambda _: None)
    output = resumed.run()
    assert output["status"] == "PAUSED_UNCERTAIN"
    assert provider.requests == []
    assert len(store.uncertain_calls()) == 1
    assert output["usage"]["usage_complete"] is False
    assert output["usage"]["total_tokens"] is None


def test_t3_resume_uncertain_keeps_one_snapshot_per_turn_and_unknown_old_usage(tmp_path):
    rows = [("ACTION", action("PROPOSE", claims=[claim()])),
            ("ACTION", action("COMMIT", candidate_id="P001")),
            ("SUPPORT_CHECK", check()),
            ("ACTION", action("FINALIZE", claim_ids=["P001"], answer="整体[E001]")),
            ("ACTION", action("PROPOSE", claims=[claim("E002", "局部评价")])),
            ("ACTION", action("COMMIT", candidate_id="P002")),
            ("SUPPORT_CHECK", check("P002", "E002")),
            ("ACTION", action("FINALIZE", claim_ids=["P001", "P002"], answer="整体与局部[E001][E002]"))]
    runner, provider, store = engine(tmp_path, rows, task="T3")
    runner.initialize()
    runner.state["usage"]["model_calls"] = 1
    runner.state["call_keys"]["0:ACTION:0:0"] = "Synthetic:T3:0001"
    store.save_state("Synthetic:T3", runner.state)
    store.start_call("Synthetic:T3:0001", "Synthetic:T3", "ACTION", runner.build_view())
    paused = Engine(sample("T3"), runner.config, provider, store, log=lambda _: None).run()
    assert paused["status"] == "PAUSED_UNCERTAIN"
    resumed = Engine(sample("T3"), runner.config, provider, store,
                     log=lambda _: None, retry_uncertain=True).run()
    assert resumed["status"] == "COMPLETED"
    assert [snapshot["turn"] for snapshot in resumed["turns"]] == [1, 2]
    assert resumed["usage"]["model_calls"] == 9
    assert resumed["usage"]["usage_complete"] is False
    assert resumed["usage"]["total_tokens"] is None


def test_propose_batch_business_error_rolls_back_previous_mutations(tmp_path):
    cfg = Config().model_dump()
    cfg["limits"]["max_actions_per_episode"] = 1
    runner, _, _ = engine(tmp_path, [("ACTION", action("PROPOSE", claims=[claim(), claim("E999")]))], config=cfg)
    output = runner.run()
    assert runner.role["candidates"] == {}
    assert output["access_violations"] == [{"kind": "access_denied", "step": 0}]


def test_extra_evidence_metadata_is_not_projected_to_model_requests(tmp_path):
    item = sample()
    item["document"]["evidence"][0]["gold"] = {"text": "NESTED_GOLD_METADATA_CANARY"}
    item["document"]["evidence"][0]["future_material"] = "UNAPPROVED_METADATA_CANARY"
    provider = ScriptProvider([
        ("ACTION", action("PROPOSE", claims=[claim()])),
        ("ACTION", action("COMMIT", candidate_id="P001")),
        ("SUPPORT_CHECK", check()),
        ("ACTION", action("COMPARE", claim_ids=["P001"], evidence_ids=["E001"], question="范围？")),
        ("SCOPE_COMPARE", relation()),
        ("ACTION", action("FINALIZE", claim_ids=["P001"], answer="评价[E001]"))])
    output = Engine(item, Config().model_dump(), provider, Store(tmp_path), log=lambda _: None).run()
    assert output["status"] == "COMPLETED"
    assert {stage for stage, _, _ in provider.requests} == {"ACTION", "SUPPORT_CHECK", "SCOPE_COMPARE"}
    for _, request, _ in provider.requests:
        text = json.dumps(request)
        assert "NESTED_GOLD_METADATA_CANARY" not in text
        assert "UNAPPROVED_METADATA_CANARY" not in text


def test_two_t4_samples_do_not_collide_on_message_ids(tmp_path):
    store = Store(tmp_path)
    ids = []
    for number in (1, 2):
        item = sample("T4", identifier=f"Synthetic{number}:T4")
        runner = Engine(item, Config().model_dump(), ScriptProvider([]), store, log=lambda _: None)
        runner.initialize()
        runner.execute(action("SEND_EVIDENCE", evidence_ids=["E001"], purpose="提供依据", request_id=None))
        runner.checkpoint("TEST_SEND", {})
        ids.append(runner.state["messages"][0]["message_id"])
    assert len(set(ids)) == 2
    assert len(store.list_messages()) == 2


def test_crash_after_support_response_reuses_both_responses_and_commits_once(tmp_path):
    class CrashBeforeCommitStore(Store):
        crash_enabled = True

        def save_state(self, sample_id, state, event=None, messages=None, prediction=None):
            if (self.crash_enabled and event and event.get("kind") == "ACTION"
                    and event.get("action", {}).get("action") == "COMMIT"):
                self.crash_enabled = False
                raise KeyboardInterrupt("人工模拟支持核对响应入库后、动作事务提交前退出")
            return super().save_state(sample_id, state, event, messages, prediction)

    store = CrashBeforeCommitStore(tmp_path)
    provider = ScriptProvider([
        ("ACTION", action("PROPOSE", claims=[claim()])),
        ("ACTION", action("COMMIT", candidate_id="P001")),
        ("SUPPORT_CHECK", check()),
        ("ACTION", action("FINALIZE", claim_ids=["P001"], answer="整体评价[E001]"))])
    cfg = Config().model_dump()
    runner = Engine(sample(), cfg, provider, store, log=lambda _: None)
    with pytest.raises(KeyboardInterrupt):
        runner.run()
    persisted = store.get_state("Synthetic:T1")
    assert persisted["role_states"]["solo"]["committed"] == {}
    assert len(provider.requests) == 3
    assert store.get_call("Synthetic:T1:0003")["status"] == "COMPLETED"
    recovered = Engine(sample(), cfg, provider, store, log=lambda _: None).run()
    assert recovered["status"] == "COMPLETED"
    assert len(recovered["prediction"]["claims"]) == 1
    assert recovered["usage"]["model_calls"] == 4
    assert len(provider.requests) == 4
    commits = [event for event in store.list_events()
               if event.get("action", {}).get("action") == "COMMIT"]
    assert len(commits) == 1


@pytest.mark.parametrize("call_budget, expected_attempts", [(1, 1), (2, 2)])
def test_timeout_retries_are_budgeted_and_unknown_usage_not_zero(tmp_path, monkeypatch,
                                                               call_budget, expected_attempts):
    monkeypatch.setattr("scopecrit.engine.time.sleep", lambda _: None)

    class TimeoutThenRead:
        def __init__(self):
            self.attempts = 0

        def complete(self, stage, request, call_id):
            self.attempts += 1
            if self.attempts == 1:
                raise TimeoutError("人工超时；异常原文不应保存在日志")
            return {"content": json.dumps(action("READ", evidence_ids=["E001"])),
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    cfg = Config().model_dump()
    cfg["limits"]["max_actions_per_episode"] = 1
    cfg["limits"]["max_model_calls_per_episode"] = call_budget
    provider = TimeoutThenRead()
    store = Store(tmp_path)
    output = Engine(sample(), cfg, provider, store, log=lambda _: None).run()
    assert output["status"] == "PARTIAL"
    assert provider.attempts == expected_attempts
    assert output["usage"]["model_calls"] == expected_attempts
    assert output["usage"]["total_tokens"] is None
    assert output["usage"]["usage_complete"] is False
    assert store.list_calls()[0]["status"] == "FAILED"
    assert "人工超时" not in json.dumps(store.list_calls(), ensure_ascii=False)

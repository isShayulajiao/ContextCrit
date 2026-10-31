"""模型的输入动作和输出结构：严格拒绝未知字段，业务权限由 engine 检查。"""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


Nonempty = Annotated[str, Field(min_length=1)]
Aspect = Literal["THEME", "SOCIAL", "CHARACTER", "PLOT", "STRUCTURE", "LANGUAGE",
                 "NARRATION", "EMOTION", "STYLE", "RECEPTION", "OVERALL", "OTHER"]
Stance = Literal["POSITIVE", "NEGATIVE", "MIXED", "NEUTRAL"]


class Claim(StrictModel):
    text: Nonempty
    aspect: Aspect
    target_scope: Nonempty
    stance: Stance
    evidence_ids: Annotated[list[Nonempty], Field(min_length=1)]

    @field_validator("text", "target_scope")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("主张文字及范围不能为空白")
        return value


class Gap(StrictModel):
    gap_id: str | None = None
    kind: Literal["SUPPORT", "SCOPE", "RELATION", "PEER_INFORMATION"]
    question: Nonempty
    related_claim_ids: list[Nonempty] = Field(default_factory=list)
    status: Literal["OPEN", "RESOLVED", "DEFERRED"] = "OPEN"
    resolution_claim_ids: list[Nonempty] = Field(default_factory=list)


class PlanArgs(StrictModel):
    gaps: Annotated[list[Gap], Field(min_length=1, max_length=8)]


class ReadArgs(StrictModel):
    evidence_ids: Annotated[list[Nonempty], Field(min_length=1)]


class SearchArgs(StrictModel):
    query: Nonempty
    top_k: Annotated[int, Field(ge=1, le=10)] = 5


class ProposeArgs(StrictModel):
    claims: Annotated[list[Claim], Field(min_length=1, max_length=4)]


class CompareArgs(StrictModel):
    claim_ids: Annotated[list[Nonempty], Field(min_length=1)]
    evidence_ids: list[Nonempty] = Field(default_factory=list)
    question: Nonempty


class CommitArgs(StrictModel):
    candidate_id: Nonempty


class DeferArgs(StrictModel):
    gap_id: Nonempty
    reason: Nonempty


class RequestArgs(StrictModel):
    question: Nonempty


class SendArgs(ReadArgs):
    purpose: Nonempty
    request_id: str | None = None


class EmptyArgs(StrictModel):
    pass


class FinalizeArgs(StrictModel):
    claim_ids: list[Nonempty]
    relation_ids: list[Nonempty] = Field(default_factory=list)
    answer: Nonempty


class ActionBase(StrictModel):
    reason: Annotated[str, Field(max_length=240)]


class PlanAction(ActionBase):
    action: Literal["PLAN_GAPS"]
    arguments: PlanArgs


class ReadAction(ActionBase):
    action: Literal["READ"]
    arguments: ReadArgs


class SearchAction(ActionBase):
    action: Literal["SEARCH"]
    arguments: SearchArgs


class ProposeAction(ActionBase):
    action: Literal["PROPOSE"]
    arguments: ProposeArgs


class CompareAction(ActionBase):
    action: Literal["COMPARE"]
    arguments: CompareArgs


class CommitAction(ActionBase):
    action: Literal["COMMIT"]
    arguments: CommitArgs


class DeferAction(ActionBase):
    action: Literal["DEFER"]
    arguments: DeferArgs


class RequestAction(ActionBase):
    action: Literal["REQUEST_PEER"]
    arguments: RequestArgs


class SendAction(ActionBase):
    action: Literal["SEND_EVIDENCE"]
    arguments: SendArgs


class YieldAction(ActionBase):
    action: Literal["YIELD"]
    arguments: EmptyArgs


class FinalizeAction(ActionBase):
    action: Literal["FINALIZE"]
    arguments: FinalizeArgs


Action = Annotated[Union[PlanAction, ReadAction, SearchAction, ProposeAction,
                        CompareAction, CommitAction, DeferAction, RequestAction,
                        SendAction, YieldAction, FinalizeAction], Field(discriminator="action")]
ACTION_ADAPTER = TypeAdapter(Action)


def parse_action(text: str) -> dict:
    """不接受 Markdown 代码围栏；格式错误交给受预算约束的修复流程。"""
    return ACTION_ADAPTER.validate_json(text).model_dump()


def action_schema() -> dict:
    return ACTION_ADAPTER.json_schema()


class SupportCheck(StrictModel):
    candidate_id: Nonempty
    support: Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED", "UNCERTAIN"]
    scope_consistent: bool
    voice_preserved: bool
    modality_preserved: bool
    evidence_ids: Annotated[list[Nonempty], Field(min_length=1)]
    reason: Annotated[str, Field(min_length=1, max_length=240)]


class Relation(StrictModel):
    claim_ids: Annotated[list[Nonempty], Field(min_length=1)]
    evidence_ids: list[Nonempty]
    relation_type: Literal["COEXISTS", "LIMITS", "CONTRASTS", "SAME_MEANING", "UNRESOLVED"]
    explanation: Nonempty

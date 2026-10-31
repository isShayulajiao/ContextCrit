"""读取 YAML 配置。密钥只从环境变量读取，不写入配置及运行清单。"""
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, model_validator

from .schemas import StrictModel

Positive = Annotated[int, Field(gt=0)]
Nonnegative = Annotated[int, Field(ge=0)]


class ProviderConfig(StrictModel):
    kind: Literal["mock", "deepseek"] = "mock"
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout_seconds: Annotated[float, Field(gt=0)] = 180.0
    reasoning_effort: Literal["low", "medium", "high", "xhigh"] | None = "high"
    thinking_enabled: bool = True
    response_format_json: bool = True
    max_output_tokens: Positive = 4096


class RuntimeConfig(StrictModel):
    concurrency: Literal[1] = 1
    seed: int = 42
    fail_fast: bool = False
    # None（YAML 写 null）处理全部；正整数处理输入顺序的前 N 条。
    max_samples: Positive | None = None


class LimitsConfig(StrictModel):
    max_actions_per_episode: Positive = 64
    max_actions_per_t3_turn: Positive = 16
    max_actions_per_t4_activation: Positive = 4
    max_t4_rounds: Positive = 6
    max_model_calls_per_episode: Positive = 80
    max_t4_messages: Positive = 16
    max_consecutive_invalid_actions: Positive = 3
    max_format_repairs_per_response: Nonnegative = 1
    max_api_retries: Nonnegative = 3
    max_input_chars_per_request: Positive = 60000
    max_reported_total_tokens_per_episode: Positive = 120000


class MemoryConfig(StrictModel):
    max_candidates: Positive = 64
    max_committed_claims: Positive = 48
    max_active_gaps: Annotated[int, Field(ge=1, le=8)] = 8
    max_claims_per_propose: Annotated[int, Field(ge=1, le=4)] = 4


class RetrievalConfig(StrictModel):
    method: Literal["char_bigram"] = "char_bigram"
    default_top_k: Annotated[int, Field(ge=1, le=10)] = 5
    max_top_k: Annotated[int, Field(ge=1, le=10)] = 10

    @model_validator(mode="after")
    def check_top_k(self):
        if self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k 不能大于 max_top_k")
        return self


class EvaluationConfig(StrictModel):
    semantic_alignment: Literal["manual"] = "manual"
    enable_paid_judge: Literal[False] = False


class MockConfig(StrictModel):
    script: str | None = None


class Config(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    fixture: bool = False
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    mock: MockConfig = Field(default_factory=MockConfig)


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("配置文件必须是 YAML 对象")
    config = Config.model_validate(raw).model_dump()
    if config["mock"]["script"]:
        script = Path(config["mock"]["script"])
        if not script.is_absolute():
            script = path.parent / script
        config["mock"]["script"] = str(script.resolve())
    return config

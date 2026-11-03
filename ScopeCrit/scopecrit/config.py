"""读取运行配置；凭证由本地私有配置注入，不进入运行清单。"""
import ipaddress
import json
import re
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import Field, JsonValue, field_validator, model_validator

from .schemas import StrictModel

Positive = Annotated[int, Field(gt=0)]
Nonnegative = Annotated[int, Field(ge=0)]
Stage = Literal["ACTION", "SUPPORT_CHECK", "SCOPE_COMPARE"]


def is_loopback_url(url: str) -> bool:
    hostname = urlsplit(url).hostname
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname or "").is_loopback
    except ValueError:
        return False


def _check_extra_body(value: JsonValue) -> None:
    """extra_body 会合并到 SDK 请求，不能覆盖权限投影后的消息或藏入凭证。"""
    private = {"api_key", "apikey", "authorization", "headers", "password", "secret",
               "access_token", "api_token"}
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower().replace("-", "_") in private:
                raise ValueError("extra_body 不能包含凭证；请使用本地 API Key 配置")
            _check_extra_body(item)
    elif isinstance(value, list):
        for item in value:
            _check_extra_body(item)


class ProviderConfig(StrictModel):
    kind: Literal["mock", "deepseek", "openai"] = "mock"
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str | None = "DEEPSEEK_API_KEY"
    timeout_seconds: Annotated[float, Field(gt=0)] = 180.0
    reasoning_effort: Literal["low", "medium", "high", "xhigh"] | None = "high"
    thinking_enabled: bool | None = True
    response_format_json: bool = True
    max_output_tokens: Positive = 4096
    extra_body: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def generic_defaults(cls, value):
        if isinstance(value, dict) and value.get("kind") == "openai":
            value = dict(value)
            if not value.get("model") or not value.get("base_url"):
                raise ValueError("openai 适配器必须指定 model 和 base_url")
            # 通用兼容接口不默认发送任何 DeepSeek 专属参数。
            for key, default in (("api_key_env", None), ("reasoning_effort", None),
                                 ("thinking_enabled", None), ("response_format_json", False)):
                value.setdefault(key, default)
            if value.get("api_key_env") == "":
                value["api_key_env"] = None
        return value

    @field_validator("model")
    @classmethod
    def model_not_blank(cls, value):
        if not value.strip() or value != value.strip():
            raise ValueError("model 必须为不含首尾空白的非空字符串")
        return value

    @field_validator("base_url")
    @classmethod
    def valid_endpoint(cls, value):
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError:
            raise ValueError("base_url 不是合法 URL") from None
        if (value != value.strip() or any(ord(char) < 33 for char in value)
                or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment
                or (port is not None and port < 1)):
            raise ValueError("base_url 不能包含认证信息、查询参数或空白")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and is_loopback_url(value)):
            raise ValueError("base_url 必须使用 HTTPS；本机回环地址允许 HTTP")
        return value

    @field_validator("api_key_env")
    @classmethod
    def valid_key_reference(cls, value):
        if value is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("api_key_env 必须是内部环境变量名，不能直接填 API Key")
        return value

    @field_validator("extra_body")
    @classmethod
    def safe_extra_body(cls, value):
        reserved = {"model", "messages", "stream", "max_tokens", "max_completion_tokens",
                    "response_format", "reasoning_effort"}
        if reserved.intersection(value):
            raise ValueError("extra_body 不能覆盖 model、messages 或 SDK 标准请求参数")
        _check_extra_body(value)
        try:
            json.dumps(value, allow_nan=False)
        except ValueError:
            raise ValueError("extra_body 必须为有限数值的 JSON") from None
        return value

    @model_validator(mode="after")
    def validate_auth_and_options(self):
        if self.kind == "deepseek" and not self.api_key_env:
            raise ValueError("DeepSeek 必须配置认证变量 api_key_env")
        if self.kind == "deepseek" and self.thinking_enabled is None:
            raise ValueError("DeepSeek 的 thinking_enabled 必须为 true 或 false")
        if self.kind == "openai":
            if not self.api_key_env and not is_loopback_url(self.base_url):
                raise ValueError("远端 OpenAI 兼容接口必须配置 api_key_env")
            if self.thinking_enabled is not None:
                raise ValueError("openai 适配器请用 extra_body 配置模型专属思考参数")
        if self.kind == "deepseek" and "thinking" in self.extra_body:
            raise ValueError("DeepSeek 请用 thinking_enabled 配置思考，避免重复参数")
        return self


class RuntimeConfig(StrictModel):
    concurrency: Annotated[int, Field(ge=1, le=32)] = 1
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
    semantic_alignment: Literal["manual", "llm"] = "manual"
    enable_paid_judge: bool = False
    judges_config: str | None = None

    @model_validator(mode="after")
    def paid_judging_is_explicit(self):
        if self.semantic_alignment == "llm" and not self.enable_paid_judge:
            raise ValueError("模型自动对齐需要 enable_paid_judge: true")
        return self


class MockConfig(StrictModel):
    script: str | None = None


class Config(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    fixture: bool = False
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    # 未指定阶段沿用 provider；各覆盖项独立配置，避免切换模型时继承错误参数。
    stage_providers: dict[Stage, ProviderConfig] = Field(default_factory=dict)
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
    judges_path = config["evaluation"]["judges_config"]
    if judges_path:
        judge_file = Path(judges_path)
        if not judge_file.is_absolute():
            judge_file = path.parent / judge_file
        config["evaluation"]["judges_config"] = str(judge_file.resolve())
    return config


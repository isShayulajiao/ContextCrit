"""独立裁判配置：共享配置只保存内部变量名，不保存密钥。

答题 Agent 与裁判使用不同的配置与独立会话。Qwen 的业务空间地址从
DASHSCOPE_BASE_URL 读取，因为用户所在地域和 WorkspaceId 无法猜测。
这些运行时值可由本地设置载入，无须用户手动导出环境变量。
"""
from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, field_validator, model_validator
import yaml

from .schemas import StrictModel


Nonempty = Annotated[str, Field(min_length=1)]
Positive = Annotated[int, Field(gt=0)]
Nonnegative = Annotated[int, Field(ge=0)]


def validate_base_url(value: str) -> str:
    """拒绝占位地址及 URL 内嵌认证，不把实际地址写进错误消息。"""
    parts = urlsplit(value)
    if (parts.scheme != "https" or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment or "{" in value or "}" in value):
        raise ValueError("接口地址须为真实 HTTPS 地址，不能含占位符、凭证或查询参数")
    return value.rstrip("/")


class JudgeConfig(StrictModel):
    name: Nonempty
    provider: Literal["openai", "zai"] = "openai"
    model: Nonempty
    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: Nonempty
    temperature: Annotated[float, Field(ge=0, le=2)] | None = None
    reasoning_effort: str | None = None
    # DeepSeek 与 Qwen 使用不同的 thinking 请求结构，不能互相替代。
    thinking_enabled: bool | None = None
    enable_thinking: bool | None = None
    response_format_json: bool = False
    timeout_seconds: Annotated[float, Field(gt=0)] = 180.0
    max_output_tokens: Positive = 8192

    @field_validator("name", "model")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("名称和模型不能为空白")
        return value

    @field_validator("api_key_env", "base_url_env")
    @classmethod
    def environment_name(cls, value):
        if value is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("必须填写合法的环境变量名称，而不是变量值")
        return value

    @field_validator("base_url")
    @classmethod
    def endpoint(cls, value):
        return validate_base_url(value) if value is not None else None

    @model_validator(mode="after")
    def check_endpoint(self):
        if self.provider == "openai" and self.base_url is None and self.base_url_env is None:
            raise ValueError("OpenAI 兼容裁判必须指定接口地址或地址环境变量")
        if self.thinking_enabled is not None and self.enable_thinking is not None:
            raise ValueError("不能同时配置 DeepSeek thinking_enabled 与 Qwen enable_thinking")
        return self


class JudgesConfig(StrictModel):
    enabled: bool = True
    fixture: bool = False
    judges: Annotated[list[JudgeConfig], Field(min_length=1, max_length=16)]
    max_api_retries: Nonnegative = 2
    max_format_repairs: Nonnegative = 1
    max_input_chars: Positive = 120000

    @model_validator(mode="after")
    def unique_names(self):
        names = [judge.name for judge in self.judges]
        if len(names) != len(set(names)):
            raise ValueError("裁判名称不能重复")
        return self


def load_judges(path: str | Path) -> dict:
    """返回完整默认值；校验错误只报告位置和类型，不回显输入值。"""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError:
        raise ValueError("裁判配置不是合法 YAML") from None
    if not isinstance(raw, dict):
        raise ValueError("裁判配置必须是 YAML 对象")
    try:
        return JudgesConfig.model_validate(raw).model_dump()
    except ValidationError as exc:
        positions = []
        for error in exc.errors(include_input=False):
            parts = list(error["loc"])
            if error["type"] == "extra_forbidden" and parts:
                parts[-1] = "<unknown_field>"
            positions.append(".".join(map(str, parts)) + ":" + error["type"])
        raise ValueError("裁判配置校验失败：" + "; ".join(positions)) from None


def resolved_base_url(judge: dict) -> str | None:
    """运行时解析地址，不将解析值回写进可持久化的配置字典。"""
    name = judge.get("base_url_env")
    if name:
        value = os.environ.get(name)
        if not value:
            if judge.get("base_url"):
                return judge["base_url"]
            raise ValueError(f"未配置接口地址（{name}），请在本地设置中填写对应地址")
        try:
            return validate_base_url(value)
        except (ValueError, TypeError):
            raise ValueError(f"运行时配置 {name} 不是有效的真实 HTTPS 接口地址") from None
    return judge.get("base_url")


def preflight_credentials(config: dict) -> list[dict]:
    """仅返回名称、是否配置和错误码；不发网络请求、不打印任何变量值。"""
    outcomes = []
    for judge in config["judges"]:
        names = [judge["api_key_env"]]
        if judge.get("base_url_env") and not judge.get("base_url"):
            names.append(judge["base_url_env"])
        missing = [name for name in names if not os.environ.get(name)]
        outcome = {"judge": judge["name"], "status": "missing" if missing else "ready", "missing": missing}
        if not missing:
            try:
                resolved_base_url(judge)
            except ValueError:
                outcome.update(status="invalid", error_code="INVALID_BASE_URL")
        outcomes.append(outcome)
    return outcomes


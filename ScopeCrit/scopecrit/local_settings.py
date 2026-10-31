"""读取用户本地私有设置，临时提供 SDK 所需的变量。

用户只需编辑一份本地 YAML，无需执行 export。真实密钥仅在该进程的
配置上下文中使用，不进入裁判配置、manifest、模型请求或提示词。
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import re
from typing import Annotated

from pydantic import Field, ValidationError, field_validator, model_validator
import yaml

from .judge_config import validate_base_url
from .schemas import StrictModel


class ApiKeys(StrictModel):
    deepseek: str = ""
    glm: str = ""
    qwen: str = ""
    grok: str = ""

    @field_validator("deepseek", "glm", "qwen", "grok")
    @classmethod
    def header_safe(cls, value):
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("Key 不能包含换行或控制字符")
        if value and value != value.strip():
            raise ValueError("Key 首尾不能包含空白")
        return value


class LocalSettings(StrictModel):
    input: Annotated[str, Field(min_length=1)]
    max_samples: Annotated[int, Field(gt=0)] | None = None
    run_dir: str | None = None
    judges_config: str | None = None
    api_keys: ApiKeys = Field(default_factory=ApiKeys)
    qwen_workspace_id: str = ""
    qwen_base_url: str = ""
    grok_base_url: str | None = None

    @field_validator("input")
    @classmethod
    def input_nonblank(cls, value):
        if not value.strip():
            raise ValueError("必须填写数据集路径")
        return value

    @field_validator("qwen_base_url", "grok_base_url")
    @classmethod
    def endpoint(cls, value):
        return validate_base_url(value) if value else value

    @field_validator("qwen_workspace_id")
    @classmethod
    def workspace_label(cls, value):
        if value and not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", value):
            raise ValueError("业务空间 ID 必须是合法的 DNS 标签")
        return value

    @model_validator(mode="after")
    def one_qwen_address_source(self):
        if self.qwen_workspace_id and self.qwen_base_url:
            raise ValueError("Qwen 业务空间 ID 与完整地址二选一")
        return self


def load_local_settings(path: str | Path) -> dict:
    """校验输入，不在异常中回显私有文件内容或字段值。"""
    path = Path(path).resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        raise ValueError("本地设置不是合法 YAML；请检查缩进和引号") from None
    if not isinstance(raw, dict):
        raise ValueError("本地设置必须是 YAML 对象")
    try:
        result = LocalSettings.model_validate(raw).model_dump()
    except ValidationError as exc:
        locations = []
        for item in exc.errors(include_input=False):
            parts = list(item["loc"])
            if item["type"] == "extra_forbidden" and parts:
                parts[-1] = "<unknown_field>"
            locations.append(".".join(map(str, parts)) + ":" + item["type"])
        raise ValueError("本地设置校验失败：" + "; ".join(locations)) from None
    if result["qwen_workspace_id"]:
        workspace = result["qwen_workspace_id"]
        result["qwen_base_url"] = f"https://{workspace}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    for key in ("input", "run_dir", "judges_config"):
        if result[key]:
            value = Path(result[key]).expanduser()
            if not value.is_absolute():
                value = path.parent / value
            result[key] = str(value.resolve())
        elif key != "input":
            result[key] = None
    return result


@contextmanager
def configured_api(settings: dict):
    """只在 with 块中设置 SDK 变量；即使调用失败也恢复原进程配置。"""
    names = {"deepseek": "DEEPSEEK_API_KEY", "glm": "ZHIPU_API_KEY",
             "qwen": "DASHSCOPE_API_KEY", "grok": "GROK_API_KEY"}
    values = {names[name]: value for name, value in settings["api_keys"].items() if value}
    if settings.get("qwen_base_url"):
        values["DASHSCOPE_BASE_URL"] = settings["qwen_base_url"]
    if settings.get("grok_base_url"):
        values["GROK_BASE_URL"] = settings["grok_base_url"]
    elif "GROK_BASE_URL" not in os.environ:
        values["GROK_BASE_URL"] = "https://api.apikey.fan/v1"
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

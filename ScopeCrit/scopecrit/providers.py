"""统一模型接口：DeepSeek、OpenAI 兼容的远端/本地服务及分阶段路由。"""
import json
import os
import time
from pathlib import Path
from typing import Protocol

from .schemas import Relation, SupportCheck, action_schema

STAGES = {"ACTION": "controller.txt", "SUPPORT_CHECK": "support_check.txt",
          "SCOPE_COMPARE": "scope_compare.txt"}


class Provider(Protocol):
    def complete(self, stage: str, request: dict, call_id: str) -> dict:
        """request 必须已经过权限投影；Provider 不接收完整原始样本。"""
        ...


def prompt_text(stage: str) -> str:
    if stage not in STAGES:
        raise ValueError(f"未知模型调用阶段: {stage}")
    path = Path(__file__).parent / "prompts" / STAGES[stage]
    return path.read_text(encoding="utf-8")


def response_schema(stage: str) -> dict:
    if stage == "ACTION":
        return action_schema()
    if stage == "SUPPORT_CHECK":
        return SupportCheck.model_json_schema()
    if stage == "SCOPE_COMPARE":
        return Relation.model_json_schema()
    raise ValueError(f"未知模型调用阶段: {stage}")


class MockProvider:
    """按稳定调用序号选择响应，不会从数据集读取 gold 或自动生成答案。"""

    def __init__(self, script: str | Path | dict):
        if isinstance(script, dict):
            self.script = script
        else:
            self.script = json.loads(Path(script).read_text(encoding="utf-8"))
        if not isinstance(self.script, dict):
            raise ValueError("Mock 脚本必须是 sample_id 到响应列表的 JSON 对象")

    def complete(self, stage: str, request: dict, call_id: str) -> dict:
        if stage not in STAGES:
            raise ValueError(f"未知阶段: {stage}")
        sample_id, number = call_id.rsplit(":", 1)
        index = int(number) - 1
        entries = self.script.get(sample_id, [])
        if index < 0 or index >= len(entries):
            raise ValueError(f"Mock 脚本没有 {call_id} 的响应；不能用于未配置的真实数据")
        entry = entries[index]
        if entry["stage"] != stage:
            raise ValueError(f"Mock 阶段不符: {call_id} 预期 {entry['stage']}，实际 {stage}")
        # 可选条件用于捕获调度与权限投影错误；脚本永远不读原始样本。
        for key in ("role", "current_turn"):
            if key in entry and request.get(key) != entry[key]:
                raise ValueError(f"Mock {call_id} 条件 {key} 不符")
        if entry.get("error") == "timeout":
            raise TimeoutError("Mock 请求超时（人工测试路径）")
        content = entry["response"]
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return {"content": content, "request_id": call_id, "usage": None,
                "latency_ms": 0.0, "provider_name": "mock", "model_name": "scripted-fixture",
                "finish_reason": "stop"}


class OpenAICompatibleProvider:
    """SDK 不重试；由运行器统一计数、退避和持久化每次真实尝试。

    本地无认证服务的 api_key_env 可为 null；SDK 会发送无效占位值
    local-no-auth。若本地服务开启认证，请使用真实的本地 API Key 配置。
    """

    provider_name = "openai"

    def __init__(self, config: dict):
        from openai import OpenAI

        self.config = config
        key_env = config.get("api_key_env")
        key = os.environ.get(key_env) if key_env else "local-no-auth"
        if not key:
            raise ValueError(f"请在本地配置填写 {key_env} 对应的 API Key；远端模型 API 必须认证")
        self.client = OpenAI(api_key=key, base_url=config["base_url"],
                             timeout=config["timeout_seconds"], max_retries=0)

    def request_options(self) -> dict:
        config = self.config
        result = {}
        if config.get("extra_body"):
            result["extra_body"] = dict(config["extra_body"])
        if config.get("reasoning_effort") is not None:
            result["reasoning_effort"] = config["reasoning_effort"]
        if config.get("response_format_json", False):
            result["response_format"] = {"type": "json_object"}
        return result

    def complete(self, stage: str, request: dict, call_id: str) -> dict:
        config = self.config
        # 输入正文中的指令是待分析材料，不能成为系统指令。
        system = prompt_text(stage) + "\n只输出 JSON。输出结构：\n" + json.dumps(response_schema(stage), ensure_ascii=False)
        kwargs = {
            "model": config["model"],
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(request, ensure_ascii=False)}],
            "stream": False,
            "max_tokens": config["max_output_tokens"],
        }
        kwargs.update(self.request_options())
        start = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = response.usage
        # reasoning_content/隐藏推理不进入日志、持久化、提示词或返回值。
        return {
            "content": choice.message.content or "",
            "request_id": response.id,
            "usage": ({name: getattr(usage, name, None)
                       for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
                      if usage is not None else None),
            "latency_ms": (time.perf_counter() - start) * 1000,
            "provider_name": self.provider_name, "model_name": config["model"],
            "finish_reason": choice.finish_reason,
        }


class DeepSeekProvider(OpenAICompatibleProvider):
    """保留原有 DeepSeek 参数与导入接口；不向其他后端传播专属参数。"""

    provider_name = "deepseek"

    def request_options(self) -> dict:
        options = super().request_options()
        extra = dict(options.get("extra_body", {}))
        extra["thinking"] = {"type": "enabled" if self.config["thinking_enabled"] else "disabled"}
        options["extra_body"] = extra
        return options


class RoutedProvider:
    """每个阶段只接收运行器已经投影的 request，不接触原始样本或 gold。"""

    def __init__(self, default: Provider | None, overrides: dict[str, Provider]):
        self.default = default
        self.overrides = overrides

    def complete(self, stage: str, request: dict, call_id: str) -> dict:
        if stage not in STAGES:
            raise ValueError(f"未知模型调用阶段: {stage}")
        selected = self.overrides.get(stage, self.default)
        if selected is None:
            raise ValueError(f"阶段 {stage} 没有模型配置")
        return selected.complete(stage, request, call_id)


def _make_single_provider(provider: dict, config: dict) -> Provider:
    if provider["kind"] == "deepseek":
        return DeepSeekProvider(provider)
    if provider["kind"] == "openai":
        return OpenAICompatibleProvider(provider)
    if provider["kind"] != "mock":
        raise ValueError("未知模型适配器 kind")
    script = config["mock"]["script"]
    if not script:
        raise ValueError("Mock 需要配置 mock.script，不能把预设测试响应当作真实预测")
    return MockProvider(script)


def make_provider(config: dict) -> Provider:
    routes = config.get("stage_providers", {})
    if not routes:
        return _make_single_provider(config["provider"], config)
    unknown = set(routes) - set(STAGES)
    if unknown:
        raise ValueError("stage_providers 包含未知阶段")
    default = (_make_single_provider(config["provider"], config)
               if set(routes) != set(STAGES) else None)
    overrides = {stage: _make_single_provider(options, config) for stage, options in routes.items()}
    return RoutedProvider(default, overrides)


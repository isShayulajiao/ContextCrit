"""模型适配器：DeepSeek 使用官方 OpenAI SDK 方式；Mock 只读人工脚本。"""
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


class DeepSeekProvider:
    """SDK 不重试；由运行器统一计数、退避和持久化每次真实尝试。"""

    def __init__(self, config: dict):
        from openai import OpenAI

        self.config = config
        key = os.environ.get(config["api_key_env"])
        if not key:
            raise ValueError(f"请先设置环境变量 {config['api_key_env']}；DeepSeek 官方 API 必须认证")
        self.client = OpenAI(api_key=key, base_url=config["base_url"],
                             timeout=config["timeout_seconds"], max_retries=0)

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
            "extra_body": {"thinking": {"type": "enabled" if config["thinking_enabled"] else "disabled"}},
        }
        if config["reasoning_effort"] is not None:
            kwargs["reasoning_effort"] = config["reasoning_effort"]
        if config["response_format_json"]:
            kwargs["response_format"] = {"type": "json_object"}
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
            "provider_name": "deepseek", "model_name": config["model"],
            "finish_reason": choice.finish_reason,
        }


def make_provider(config: dict) -> Provider:
    if config["provider"]["kind"] == "deepseek":
        return DeepSeekProvider(config["provider"])
    script = config["mock"]["script"]
    if not script:
        raise ValueError("Mock 需要配置 mock.script，不能把预设测试响应当作真实预测")
    return MockProvider(script)

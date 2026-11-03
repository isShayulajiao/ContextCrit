"""裁判 API 适配：各自独立客户端、独立请求，不复用答题 Agent 会话。

Qwen、MiniMax 和 Gemini 网关使用 OpenAI 兼容接口。保留旧版 DeepSeek、
Grok 和 GLM 配置的支持，其中 GLM 使用 zai-sdk。
SDK 重试统一设为零，实际重试次数由上层裁判运行器记录。
"""
from __future__ import annotations

import json
import os
import time

from .judge_config import JudgeConfig, resolved_base_url


def _response(response, judge: dict, start: float) -> dict:
    choice = response.choices[0]
    usage = response.usage
    return {
        "content": choice.message.content or "",
        "request_id": response.id,
        "usage": ({name: getattr(usage, name, None)
                   for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
                  if usage is not None else None),
        "latency_ms": (time.perf_counter() - start) * 1000,
        "provider_name": judge["provider"],
        "model_name": judge["model"],
        "finish_reason": choice.finish_reason,
    }


class JudgeProvider:
    """共同响应协议；不保存 key，不返回 SDK headers 或 reasoning_content。"""

    def __init__(self, judge: dict):
        self.judge = JudgeConfig.model_validate(judge).model_dump()
        name = self.judge["api_key_env"]
        key = os.environ.get(name)
        if not key:
            raise ValueError(f"未配置裁判 API Key（{name}），请在本地设置的 api_keys 中填写对应密钥")
        endpoint = resolved_base_url(self.judge)
        options = {"api_key": key, "timeout": self.judge["timeout_seconds"], "max_retries": 0}
        if endpoint:
            options["base_url"] = endpoint
        if self.judge["provider"] == "zai":
            try:
                from zai import ZhipuAiClient
            except ImportError:
                raise ValueError("GLM 裁判需要 zai-sdk，请重新安装 requirements.txt") from None
            self.client = ZhipuAiClient(**options)
        else:
            from openai import OpenAI
            self.client = OpenAI(**options)

    def complete(self, system_prompt: str, request: dict, call_id: str) -> dict:
        """模型特有参数按配置添加，不把 DeepSeek 参数传给其他服务。"""
        judge = self.judge
        kwargs = {
            "model": judge["model"],
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": json.dumps(request, ensure_ascii=False)}],
            "stream": False,
            "max_tokens": judge["max_output_tokens"],
        }
        if judge["temperature"] is not None:
            kwargs["temperature"] = judge["temperature"]
        if judge["reasoning_effort"] is not None:
            kwargs["reasoning_effort"] = judge["reasoning_effort"]
        if judge["thinking_enabled"] is not None:
            kwargs["extra_body"] = {"thinking": {"type": "enabled" if judge["thinking_enabled"] else "disabled"}}
        if judge["enable_thinking"] is not None:
            kwargs["extra_body"] = {"enable_thinking": judge["enable_thinking"]}
        if judge["response_format_json"]:
            kwargs["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        return _response(response, judge, start)


def make_judge_provider(judge: dict) -> JudgeProvider:
    return JudgeProvider(judge)


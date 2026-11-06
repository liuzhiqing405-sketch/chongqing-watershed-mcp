"""Optional language-model access.

The model proposes routes and grounded entities, rewords evidence-bound
replies and answers engineering knowledge questions. It has no authority to
execute or confirm workflow actions.

The endpoint is OpenAI-compatible Chat Completions, Responses, or native
Anthropic Messages (including 4Router Claude). The key is
held in a ``SecretStr`` from the environment or local connection form and is
never logged or written into conversation history.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import quote, urlsplit

from .config import LLMSettings


class LLMUnavailable(RuntimeError):
    """The model is not configured, or its client library is missing."""


class LLMRequestFailed(RuntimeError):
    """The endpoint refused or failed the request."""

    def __init__(self, message: str, *, reason: str = "request_failed") -> None:
        super().__init__(message)
        self.reason = reason


#: ```json ... ``` fences, which models add even when asked for bare JSON.
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> dict[str, Any]:
    """Parse a model's reply into a JSON object.

    Tolerates code fences and surrounding prose, and refuses anything that is
    not an object — a list or a bare string is not a usable intent.
    """
    if not isinstance(text, str) or not text.strip():
        raise LLMRequestFailed("模型返回为空", reason="empty_response")
    cleaned = _FENCE.sub("", text.strip()).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMRequestFailed("模型未返回 JSON 对象", reason="invalid_json") from None
        try:
            payload = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMRequestFailed(f"模型返回的 JSON 无法解析：{exc}", reason="invalid_json") from exc
    if not isinstance(payload, dict):
        raise LLMRequestFailed("模型返回的不是 JSON 对象", reason="invalid_json")
    return payload


def api_endpoint(settings: LLMSettings, operation: str) -> str:
    base = str(settings.base_url).rstrip("/")
    if settings.api_format == "gemini_generate_content":
        version = "v1" if operation == "models" and urlsplit(base).hostname == "4router.net" else "v1beta"
        if base.endswith(("/v1", "/v1beta")):
            base = base.rsplit("/", 1)[0]
        base += "/" + version
        if operation == "models":
            return base + "/models"
        model = str(settings.model).removeprefix("models/")
        return f"{base}/models/{quote(model, safe='')}:generateContent"
    if settings.api_format == "anthropic_messages" and not urlsplit(base).path.rstrip("/").endswith("/v1"):
        base += "/v1"
    return f"{base}/{operation}"


def model_headers(settings: LLMSettings, *, json_body: bool = False) -> dict[str, str]:
    headers = {"Content-Type": "application/json"} if json_body else {}
    anthropic = settings.api_format == "anthropic_messages"
    gemini = settings.api_format == "gemini_generate_content"
    if anthropic:
        headers["anthropic-version"] = "2023-06-01"
    if settings.api_key is not None:
        key = settings.api_key.get_secret_value()
        if anthropic:
            headers["x-api-key"] = key
        if gemini:
            headers["x-goog-api-key"] = key
        # 4Router documents ANTHROPIC_AUTH_TOKEN (Bearer authentication).
        if not (anthropic or gemini) or urlsplit(str(settings.base_url)).hostname == "4router.net":
            headers["Authorization"] = f"Bearer {key}"
    return headers


class OpenAICompatibleClient:
    """A JSON client shared by existing OpenAI and native Claude transports."""

    def __init__(self, settings: LLMSettings,
                 logger: logging.Logger | None = None) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger("slope_agent")
        self.last_response_model: str | None = None

    def _client(self):
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise LLMUnavailable(
                "缺少 httpx；安装方式：pip install -e \".[llm]\"") from exc
        return httpx

    async def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        """Ask for a JSON object and return it, or raise."""
        if not self.settings.is_available():
            raise LLMUnavailable(self.settings.unavailable_reason() or "语言模型不可用")
        httpx = self._client()
        responses = self.settings.api_format == "responses"
        anthropic = self.settings.api_format == "anthropic_messages"
        gemini = self.settings.api_format == "gemini_generate_content"
        url = api_endpoint(self.settings, "generateContent" if gemini else "messages" if anthropic else "responses" if responses else "chat/completions")
        headers = model_headers(self.settings, json_body=True)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        token_field = "maxOutputTokens" if gemini else "max_output_tokens" if responses else "max_tokens"
        body = {"model": self.settings.model, token_field: self.settings.maximum_tokens}
        if self.settings.temperature is not None:
            body["temperature"] = self.settings.temperature
        if gemini:
            generation = {token_field: self.settings.maximum_tokens}
            if self.settings.temperature is not None:
                generation["temperature"] = self.settings.temperature
            if self.settings.json_mode:
                generation["responseMimeType"] = "application/json"
            effort = self.settings.reasoning_effort
            budget = self.settings.gemini_thinking_budget_tokens
            if budget is not None:
                generation["thinkingConfig"] = {"thinkingBudget": budget}
            elif effort is not None:
                model = str(self.settings.model).lower()
                if effort == "none":
                    generation["thinkingConfig"] = {"thinkingBudget": 0}
                elif "gemini-2.5" in model:
                    generation["thinkingConfig"] = {"thinkingBudget": {"minimal":1024,"low":1024,"medium":8192,"high":24576}[effort]}
                else:
                    generation["thinkingConfig"] = {"thinkingLevel": effort.upper()}
            body = {"systemInstruction":{"parts":[{"text":system}]},
                    "contents":[{"role":"user","parts":[{"text":user}]}],"generationConfig":generation}
        elif anthropic:
            body.update(system=system, messages=[{"role": "user", "content": user}], stream=True)
            if self.settings.json_mode:
                # Native Messages has no OpenAI json_object mode. Keep the
                # same application schema validation without forcing tool use.
                body["system"] += "\n只输出完整JSON对象，不要Markdown围栏或额外说明。"
            thinking = self.settings.anthropic_thinking
            if thinking != "default":
                body["thinking"] = {"type": thinking}
                if thinking == "enabled":
                    body["thinking"]["budget_tokens"] = self.settings.anthropic_thinking_budget_tokens
            if self.settings.reasoning_effort is not None:
                body["output_config"] = {"effort": self.settings.reasoning_effort}
        elif responses:
            body.update(input=messages, store=False, stream=True)
        else:
            body["messages"] = messages
            if self.settings.chat_streaming:
                body["stream"] = True
        if self.settings.provider == "qwen":
            if self.settings.qwen_thinking != "default":
                body["enable_thinking"] = self.settings.qwen_thinking == "enabled"
            if self.settings.qwen_thinking_budget_tokens is not None:
                body["thinking_budget"] = self.settings.qwen_thinking_budget_tokens
        if self.settings.provider == "gemini" and not gemini and self.settings.gemini_thinking_budget_tokens is not None:
            # This is the REST shape; the SDK's outer extra_body argument is
            # not copied into this direct HTTP request.
            body["extra_body"] = {"google":{"thinking_config":{"thinking_budget":self.settings.gemini_thinking_budget_tokens}}}
        # DeepSeek and other OpenAI-compatible providers can guarantee a JSON
        # object when this flag is present.  It complements (rather than
        # replaces) the explicit JSON instruction in both agent prompts.
        if self.settings.json_mode and not (anthropic or gemini):
            if responses:
                body["text"] = {"format": {"type": "json_object"}}
            else:
                body["response_format"] = {"type": "json_object"}
        if self.settings.reasoning_effort is not None and not (anthropic or gemini):
            if responses:
                body["reasoning"] = {"effort": self.settings.reasoning_effort}
            else:
                body["reasoning_effort"] = self.settings.reasoning_effort
        if self.settings.provider == "deepseek":
            effort = self.settings.reasoning_effort
            thinking = ("disabled" if effort == "none" else "enabled") if effort is not None else self.settings.deepseek_thinking
            if responses:
                if effort is None and thinking == "disabled":
                    body["reasoning"] = {"effort": "none"}
            else:
                body["thinking"] = {"type": thinking}
        last: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.settings.request_timeout_s) as client:
                    response = await client.post(url, headers=headers, json=body)
                if response.status_code >= 400:
                    # The body may echo the request; keep the status only.
                    raise LLMRequestFailed(f"语言模型返回 HTTP {response.status_code}",
                                           reason=f"http_{response.status_code}")
                payload = (response.json() if gemini else self._anthropic_payload(response) if anthropic
                           else self._payload(response) if responses else self._chat_payload(response))
                result = extract_json(self._gemini_content(payload) if gemini else self._anthropic_content(payload) if anthropic else
                                      self._responses_content(payload) if responses else self._content(payload))
                model = payload.get("modelVersion" if gemini else "model") if isinstance(payload, dict) else None
                if isinstance(model, str) and model.strip() and len(model) <= 200 and not any(ord(c) < 32 for c in model):
                    self.last_response_model = model.strip()
                return result
            except LLMRequestFailed as exc:
                last = exc
                if ((exc.reason in {"response_truncated", "empty_response"}
                     or (gemini and exc.reason == "invalid_json"))
                        and attempt < self.settings.max_retries):
                    if exc.reason == "response_truncated":
                        budget_body = body["generationConfig"] if gemini else body
                        budget_body[token_field] = min(budget_body[token_field] * 2, 32000)
                    continue
                break
            except Exception as exc:  # noqa: BLE001 - retried below
                last = exc
                # Transport exceptions can contain a credential or remote body.
                self.logger.debug("语言模型请求失败（第 %d 次）：%s", attempt + 1, type(exc).__name__)
        detail = str(last) if isinstance(last, LLMRequestFailed) else type(last).__name__
        raise LLMRequestFailed(f"语言模型请求失败：{detail}",
                               reason=getattr(last, "reason", "transport_error"))

    @staticmethod
    def _gemini_content(payload: Any) -> str:
        if not isinstance(payload, dict) or payload.get("error"):
            raise LLMRequestFailed("Gemini响应无效", reason="invalid_response")
        if (payload.get("promptFeedback") or {}).get("blockReason"):
            raise LLMRequestFailed("Gemini拒绝了本次请求", reason="request_failed")
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
            raise LLMRequestFailed("Gemini缺少正文候选", reason="empty_response")
        candidate = candidates[0]
        if candidate.get("finishReason") == "MAX_TOKENS":
            raise LLMRequestFailed("Gemini回答未完整返回", reason="response_truncated")
        if candidate.get("finishReason") != "STOP":
            raise LLMRequestFailed("Gemini未正常完成回答", reason="request_failed")
        parts = (candidate.get("content") or {}).get("parts")
        if not isinstance(parts, list):
            raise LLMRequestFailed("Gemini内容块无效", reason="invalid_response")
        if any(isinstance(part, dict) and "functionCall" in part for part in parts):
            raise LLMRequestFailed("Gemini返回未经授权的工具请求", reason="unexpected_tool_call")
        text = "".join(p["text"] for p in parts if isinstance(p, dict) and not p.get("thought") and isinstance(p.get("text"), str))
        if not text.strip():
            raise LLMRequestFailed("Gemini未返回正文", reason="empty_response")
        return text

    @staticmethod
    def _chat_payload(response: Any) -> dict[str, Any]:
        if "text/event-stream" not in getattr(response, "headers", {}).get("content-type", ""):
            return response.json()
        text, model, finish = [], None, None
        data_lines: list[str] = []
        for line in (str(response.text).replace("\r\n", "\n") + "\n\n").split("\n"):
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip()); continue
            if line or not data_lines:
                continue
            data = "\n".join(data_lines); data_lines = []
            if data == "[DONE]":
                if finish is None:
                    raise LLMRequestFailed("兼容接口事件流缺少结束原因", reason="response_truncated")
                return {"model":model,"choices":[{"finish_reason":finish,"message":{"content":"".join(text)}}]}
            try:
                event = json.loads(data)
            except (ValueError, TypeError):
                raise LLMRequestFailed("兼容接口事件无法解析", reason="invalid_response") from None
            if not isinstance(event, dict) or event.get("error"):
                raise LLMRequestFailed("兼容接口返回错误事件", reason="request_failed")
            if event.get("model"):
                model = event["model"]
            choices = event.get("choices") or []
            if not choices:
                continue
            if len(choices) != 1 or not isinstance(choices[0], dict):
                raise LLMRequestFailed("兼容接口内容候选无效", reason="invalid_response")
            choice = choices[0]; delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise LLMRequestFailed("兼容接口正文增量无效", reason="invalid_response")
            if delta.get("tool_calls") or delta.get("function_call") or choice.get("finish_reason") in {"tool_calls","function_call"}:
                raise LLMRequestFailed("兼容接口返回未经授权的工具请求", reason="unexpected_tool_call")
            if isinstance(delta.get("content"), str):
                text.append(delta["content"])
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
        raise LLMRequestFailed("兼容接口事件流未正常结束", reason="response_truncated")

    @staticmethod
    def _anthropic_payload(response: Any) -> dict[str, Any]:
        if "text/event-stream" not in getattr(response, "headers", {}).get("content-type", ""):
            return response.json()
        message = None
        blocks: dict[int, dict] = {}
        open_blocks: set[int] = set()
        data_lines: list[str] = []
        for line in (str(response.text).replace("\r\n", "\n") + "\n\n").split("\n"):
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
                continue
            if line or not data_lines:
                continue
            try:
                event = json.loads("\n".join(data_lines))
            except (ValueError, TypeError):
                raise LLMRequestFailed("Claude事件流无法解析", reason="invalid_response") from None
            finally:
                data_lines = []
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if kind == "error":
                raise LLMRequestFailed("Claude服务返回错误事件", reason="request_failed")
            if kind == "message_start":
                if message is not None or not isinstance(event.get("message"), dict):
                    raise LLMRequestFailed("Claude事件流消息头无效", reason="invalid_response")
                message = dict(event["message"])
            elif kind == "content_block_start":
                index, block = event.get("index"), event.get("content_block")
                if message is None or not isinstance(index, int) or index in blocks or not isinstance(block, dict):
                    raise LLMRequestFailed("Claude内容块无效", reason="invalid_response")
                # Never retain thinking text, signatures, or tool arguments.
                blocks[index] = {"type": block.get("type")}
                if block.get("type") == "text":
                    blocks[index]["text"] = block.get("text", "")
                open_blocks.add(index)
            elif kind == "content_block_delta":
                index, delta = event.get("index"), event.get("delta")
                if index not in open_blocks or not isinstance(delta, dict):
                    raise LLMRequestFailed("Claude内容增量缺少有效内容块", reason="invalid_response")
                if delta.get("type") == "text_delta":
                    text = delta.get("text")
                    if blocks[index].get("type") != "text" or not isinstance(text, str):
                        raise LLMRequestFailed("Claude正文增量无效", reason="invalid_response")
                    blocks[index]["text"] += text
            elif kind == "content_block_stop":
                index = event.get("index")
                if index not in open_blocks:
                    raise LLMRequestFailed("Claude内容块结束事件无效", reason="invalid_response")
                open_blocks.remove(index)
            elif kind == "message_delta":
                if message is None or not isinstance(event.get("delta"), dict):
                    raise LLMRequestFailed("Claude消息增量无效", reason="invalid_response")
                if "stop_reason" in event["delta"]:
                    message["stop_reason"] = event["delta"]["stop_reason"]
            elif kind == "message_stop":
                if message is None or open_blocks or not message.get("stop_reason"):
                    raise LLMRequestFailed("Claude事件流未完整结束", reason="response_truncated")
                return {**message, "content": [blocks[i] for i in sorted(blocks)]}
        raise LLMRequestFailed("Claude事件流未正常结束", reason="response_truncated")

    @staticmethod
    def _anthropic_content(payload: Any) -> str:
        if not isinstance(payload, dict) or payload.get("type") == "error" or payload.get("error"):
            raise LLMRequestFailed("Claude返回失败响应", reason="invalid_response")
        stop = payload.get("stop_reason")
        if stop in {"max_tokens", "model_context_window_exceeded"}:
            raise LLMRequestFailed("Claude回答未完整返回", reason="response_truncated")
        content = payload.get("content")
        if not isinstance(content, list):
            raise LLMRequestFailed("Claude响应缺少内容块", reason="invalid_response")
        if stop == "tool_use" or any(isinstance(b, dict) and b.get("type") in {"tool_use", "server_tool_use"} for b in content):
            raise LLMRequestFailed("Claude返回了本通道未授权的工具请求", reason="unexpected_tool_call")
        if stop not in {"end_turn", "stop_sequence"}:
            raise LLMRequestFailed("Claude未正常完成正文回答", reason="request_failed")
        text = "".join(block["text"] for block in content if isinstance(block, dict)
                       and block.get("type") == "text" and isinstance(block.get("text"), str))
        if not text.strip():
            raise LLMRequestFailed("Claude未返回正文回答", reason="empty_response")
        return text

    @staticmethod
    def _payload(response: Any) -> dict[str, Any]:
        """Accept ordinary JSON or a completed Responses SSE stream.

        No text deltas are treated as complete JSON before a terminal event.
        Some relay services require the Responses streaming transport.
        """
        headers = getattr(response, "headers", {})
        if "text/event-stream" not in headers.get("content-type", ""):
            return response.json()
        data_lines: list[str] = []
        for line in (str(response.text).replace("\r\n", "\n") + "\n\n").split("\n"):
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif not line and data_lines:
                data = "\n".join(data_lines); data_lines = []
                if data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except (ValueError, TypeError):
                    raise LLMRequestFailed("模型返回的事件流无法解析", reason="invalid_response") from None
                if not isinstance(event, dict):
                    continue
                if event.get("type") in {"response.completed", "response.incomplete", "response.failed"}:
                    payload = event.get("response")
                    if not isinstance(payload, dict):
                        raise LLMRequestFailed("模型事件缺少响应正文", reason="invalid_response")
                    if event["type"] == "response.incomplete":
                        payload = {**payload, "status": "incomplete"}
                    if event["type"] == "response.failed":
                        raise LLMRequestFailed("模型服务返回失败状态", reason="request_failed")
                    return payload
                if event.get("type") == "error":
                    raise LLMRequestFailed("模型服务返回错误事件", reason="request_failed")
        raise LLMRequestFailed("模型事件流未正常结束", reason="response_truncated")

    @staticmethod
    def _responses_content(payload: Any) -> str:
        if not isinstance(payload, dict):
            raise LLMRequestFailed("模型响应不是对象", reason="invalid_response")
        if payload.get("status") == "incomplete":
            raise LLMRequestFailed("模型输出未完成", reason="response_truncated")
        if payload.get("status") in {"failed", "cancelled"} or payload.get("error"):
            raise LLMRequestFailed("模型服务返回失败状态", reason="request_failed")
        output = payload.get("output")
        chunks = [part["text"] for item in output if isinstance(item, dict) and item.get("type") == "message"
                  for part in (item.get("content") or []) if isinstance(part, dict)
                  and part.get("type") == "output_text" and isinstance(part.get("text"), str)] if isinstance(output, list) else []
        content = "".join(chunks) or payload.get("output_text")
        if not isinstance(content, str) or not content.strip():
            raise LLMRequestFailed("模型未返回正文回答", reason="empty_response")
        return content

    @staticmethod
    def _content(payload: Any) -> str:
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestFailed("语言模型响应缺少 choices[0].message.content",
                                   reason="invalid_response") from exc
        if choice.get("finish_reason") == "length":
            raise LLMRequestFailed("模型输出达到 token 上限，未返回完整回答",
                                   reason="response_truncated")
        if choice.get("finish_reason") in {"tool_calls", "function_call"} or choice["message"].get("tool_calls") or choice["message"].get("function_call"):
            raise LLMRequestFailed("兼容接口返回未经授权的工具请求", reason="unexpected_tool_call")
        if choice.get("finish_reason") not in {None, "stop"}:
            raise LLMRequestFailed("兼容接口未正常完成回答", reason="request_failed")
        if not isinstance(content, str) or not content.strip():
            raise LLMRequestFailed("模型未返回正文回答", reason="empty_response")
        return content


def build_llm_client(settings: LLMSettings,
                     logger: logging.Logger | None = None) -> OpenAICompatibleClient | None:
    """A client when one is configured, else ``None``.

    Returning ``None`` rather than raising is what lets every caller fall back
    to the deterministic path without a special case.
    """
    if not settings.is_available():
        return None
    return OpenAICompatibleClient(settings, logger)

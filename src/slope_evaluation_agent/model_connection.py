"""Local, memory-only model connection settings; no workflow or session writes."""
from __future__ import annotations

import secrets
from typing import Any
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, SecretStr, field_validator

from .config import LLMSettings
from .llm import LLMRequestFailed, OpenAICompatibleClient, api_endpoint, model_headers


class ConnectionRequest(LLMSettings):
    model_config = ConfigDict(extra="forbid")
    base_url: str
    model: str = ""
    retain_key: bool = True

    @field_validator("base_url")
    @classmethod
    def valid_endpoint(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        parts = urlsplit(value)
        if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username
                or parts.password or parts.query or parts.fragment or len(value) > 2048):
            raise ValueError("invalid API base URL")
        if value.endswith(("/chat/completions", "/responses", "/messages", "/models", ":generateContent", ":streamGenerateContent")):
            raise ValueError("use API base URL, not operation URL")
        return value

    @field_validator("model")
    @classmethod
    def valid_model(cls, value: str) -> str:
        value = value.strip()
        if len(value) > 200 or any(ord(c) < 32 for c in value):
            raise ValueError("invalid model ID")
        return value


def public_connection(runtime) -> dict[str, Any]:
    llm = runtime.agent.settings.llm
    observed = getattr(getattr(runtime.agent, "llm", None), "last_response_model", None)
    return {"provider": llm.provider, "base_url": llm.base_url, "model": llm.model,
            "api_format": llm.api_format, "reasoning_effort": llm.reasoning_effort,
            "deepseek_thinking": llm.deepseek_thinking, "temperature": llm.temperature,
            "anthropic_thinking": llm.anthropic_thinking,
            "anthropic_thinking_budget_tokens": llm.anthropic_thinking_budget_tokens,
            "supported_api_formats": ["chat_completions", "responses", "anthropic_messages", "gemini_generate_content"],
            "supported_providers": ["openai-compatible", "deepseek", "anthropic", "grok", "gemini", "qwen"],
            "chat_streaming": llm.chat_streaming,
            "qwen_thinking": llm.qwen_thinking,
            "qwen_thinking_budget_tokens": llm.qwen_thinking_budget_tokens,
            "gemini_thinking_budget_tokens": llm.gemini_thinking_budget_tokens,
            "maximum_tokens": llm.maximum_tokens, "request_timeout_s": llm.request_timeout_s,
            "max_retries": llm.max_retries, "json_mode": llm.json_mode,
            "configured": llm.is_available(), "enabled": llm.enabled,
            "credential_present": bool(llm.api_key and llm.api_key.get_secret_value()),
            "response_model": observed, "display_model": observed or llm.model,
            "model_source": "response" if observed else "configuration"}


def candidate_settings(body: ConnectionRequest, current: LLMSettings) -> LLMSettings:
    data = body.model_dump(exclude={"retain_key", "api_key"})
    key = body.api_key.get_secret_value().strip() if body.api_key else ""
    # Never silently send an existing credential to a newly entered endpoint.
    if key:
        data["api_key"] = SecretStr(key)
    elif body.retain_key and body.base_url == str(current.base_url or "").rstrip("/"):
        data["api_key"] = current.api_key
    else:
        data["api_key"] = None
    data["enabled"] = True
    return LLMSettings.model_validate(data)


def connection_failure_message(reason: str) -> str:
    """Explain the observed failure without guessing or echoing remote bodies."""
    messages = {
        "http_503": "中转站返回HTTP 503：本次生成请求未获服务。可能涉及模型通道、账户分组路由或上游暂时异常；仅凭状态码不能确定原因，请查看中转站该次请求的错误详情。模型列表可读取不代表生成通道可用。",
        "http_401": "接口返回HTTP 401：认证未通过，请核对密钥是否有效及所属服务地址。",
        "http_403": "接口返回HTTP 403：请求被拒绝，请核对账户权限及服务商策略。",
        "http_404": "接口返回HTTP 404：请核对API根地址、接口协议和模型ID，并查看服务商错误详情。",
        "http_429": "接口返回HTTP 429：请求受到限流或额度限制，请查看账户配额和服务商日志。",
        "http_400": "接口返回HTTP 400：请求参数未被接受，请核对模型对应的协议、思考模式、effort及生成预算；不会自动改成其他模型。",
        "unexpected_tool_call": "模型返回了工具请求，本智能体的模型通道只接收JSON意图或回答，未执行该请求。",
        "response_truncated": "模型回答未完整返回，请检查生成token预算和服务商输出限制。",
        "transport_error": "未获得完整接口响应，请检查网络、请求超时和服务状态。",
    }
    return messages.get(reason, f"连接测试未通过（{reason}）。请核对模型权限、协议和请求参数。")


async def check_routing_compatibility(probe, settings) -> list[dict[str, Any]]:
    """Exercise the real router in unsaved sessions, without tools or actions.

    A valid JSON response or a rule fallback cannot demonstrate that a newly
    connected model understands the application's routing contract.
    """
    from .dialogue_router import DialogueRouter, RouteKind
    from .models import AgentSession

    router = DialogueRouter(llm=probe, settings=settings)
    checks = []
    cases = [
        ("greeting", "你好", "greeting"),
        ("engineering_question", "请解释降雨影响边坡稳定性的可能机制，只讨论原理，不启动计算。", None),
    ]
    for name, message, topic in cases:
        session = AgentSession(session_id="session-000000000000", state="new")
        decision = await router.decide(message, session)
        passed = (router.last_source == "model"
                  and decision.route is RouteKind.KNOWLEDGE_CHAT
                  and (topic is None or decision.topic == topic)
                  and not decision.candidates and not decision.missing_fields)
        check = {"name": name, "passed": passed, "route": decision.route.value,
                 "source": router.last_source}
        if not passed:
            check["reason"] = ((router.last_diagnostic or {}).get("reason")
                               or "unexpected_conversation_route")
        checks.append(check)
        if not passed:
            break
    return checks


def install_model_connection(app, runtime) -> None:
    token = secrets.token_urlsafe(32)

    def local_page(request: Request) -> None:
        host = request.url.hostname
        test_client = host == "testserver" and request.client is not None and request.client.host == "testclient"
        if host not in {"127.0.0.1", "localhost", "::1"} and not test_client:
            raise HTTPException(403, "请通过本机地址打开模型设置。")

    async def parse(request: Request) -> ConnectionRequest:
        local_page(request)
        supplied = request.headers.get("x-model-connection-token", "")
        if not secrets.compare_digest(supplied, token):
            raise HTTPException(403, "请从本机模型设置面板操作。")
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            raise HTTPException(403, "模型设置只接受同一页面的操作。")
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, "模型设置只接受同一页面的操作。")
        if not request.headers.get("content-type", "").startswith("application/json"):
            raise HTTPException(415, "请使用模型设置表单提交。")
        # Do not let FastAPI's default validation response echo a password.
        try:
            raw = await request.body()
            if len(raw) > 16384:
                raise ValueError("too large")
            return ConnectionRequest.model_validate_json(raw)
        except Exception:
            raise HTTPException(400, "连接参数无效：请核对API根地址、模型名称、协议与推理选项。") from None

    def require_model(settings: LLMSettings) -> None:
        if not settings.model:
            raise HTTPException(400, "请选择或手动填写模型名称。")

    @app.get("/api/v1/model-connection")
    async def connection_state(request: Request):
        local_page(request)
        return {**public_connection(runtime), "form_token": token}

    @app.post("/api/v1/model-connection/models")
    async def list_models(request: Request):
        import httpx
        body = await parse(request)
        settings = candidate_settings(body, runtime.agent.settings.llm)
        headers = model_headers(settings)
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
                response = await client.get(api_endpoint(settings, "models"), headers=headers)
                if response.status_code >= 400:
                    raise HTTPException(400, f"读取模型列表返回HTTP {response.status_code}，可核对凭据或手动填写模型ID。")
                data = response.json()
                items = data.get("data", data.get("models", [])) if isinstance(data, dict) else []
                ids = set()
                for row in items if isinstance(items, list) else []:
                    value = row.get("id", row.get("name")) if isinstance(row, dict) else None
                    if isinstance(value, str):
                        value = value.strip()
                        if settings.api_format == "gemini_generate_content":
                            value = value.removeprefix("models/")
                        if value and len(value) <= 200 and not any(ord(c) < 32 for c in value):
                            ids.add(value)
                ids = sorted(ids)[:2000]
                if not ids:
                    raise HTTPException(400, "接口未提供可读取的模型列表，请手动填写模型ID。")
                efforts = {}
                # Provider-published metadata is a hint, never a substitute for
                # the account's model list or a successful connection check.
                if urlsplit(str(settings.base_url)).hostname == "4router.net":
                    try:
                        meta = await client.get("https://docs.4router.net/models.json")
                        if meta.status_code == 200:
                            for row in meta.json().get("models", []):
                                if row.get("slug") in ids:
                                    efforts[row["slug"]] = [x['effort'] for x in row.get('supported_reasoning_levels', [])
                                                           if isinstance(x, dict) and x.get('effort') in {"none","minimal","low","medium","high","xhigh","max","ultra"}]
                    except Exception:
                        pass
                return {"models": ids, "reasoning_efforts": efforts}
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(400, "无法读取模型列表，请检查地址和网络，或手动填写模型ID。") from None

    @app.post("/api/v1/model-connection/check")
    async def check_connection(request: Request):
        body = await parse(request)
        settings = candidate_settings(body, runtime.agent.settings.llm)
        require_model(settings)
        probe = OpenAICompatibleClient(settings)
        try:
            result = await probe.complete_json(system='Reply with a JSON object only: {"ok":true}.', user='Return {"ok":true}.')
        except LLMRequestFailed as exc:
            raise HTTPException(400, connection_failure_message(exc.reason)) from None
        except Exception:
            raise HTTPException(400, "连接测试未通过，请检查连接参数。") from None
        if result.get("ok") is not True:
            raise HTTPException(400, "模型已返回，但未通过JSON回答检查。")
        checks = await check_routing_compatibility(probe, runtime.agent.settings.dialogue)
        return {"ok": True, "model": settings.model, "response_model": probe.last_response_model,
                "api_format": settings.api_format, "reasoning_effort": settings.reasoning_effort,
                "workflow_compatible": all(check["passed"] for check in checks),
                "routing_checks": checks, "compatibility_scope": "basic_intent_routing_only"}

    @app.post("/api/v1/model-connection")
    async def apply_connection(request: Request):
        body = await parse(request)
        settings = candidate_settings(body, runtime.agent.settings.llm)
        require_model(settings)
        if any(lock.locked() for lock in runtime.locks.values()):
            raise HTTPException(409, "当前正在处理请求，请完成后再切换模型。")
        client = OpenAICompatibleClient(settings)
        # No await between the busy check and replacement: in-flight agent
        # operations retain one model, while future messages share the new one.
        runtime.settings.llm = settings
        runtime.agent.settings.llm = settings
        runtime.agent.llm = client
        for name in ("router", "parser", "rewriter", "responder", "locator"):
            component = getattr(runtime.agent, name, None)
            if component is not None and hasattr(component, "llm"):
                component.llm = None if name == "responder" and not runtime.settings.dialogue.allow_general_llm else client
        return {**public_connection(runtime), "saved_in_memory_only": True}

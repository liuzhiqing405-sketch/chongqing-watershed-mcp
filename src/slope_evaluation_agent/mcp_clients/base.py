"""Shared machinery for the two MCP clients.

Both clients launch their server over stdio and speak MCP to it. Nothing here
computes a scientific quantity: the clients validate, call, and map failures
onto exceptions that a user can act on.

Every failure mode has its own exception type so the workflow can tell "the
server would not start" from "the server refused this request" from "the answer
did not have the shape we require". Each carries a ``next_step`` phrase that is
safe to show a user.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from contextlib import AsyncExitStack
from typing import Any, Sequence

from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

from ..config import ServerSettings


#: Ids are opaque strings minted by the servers; the agent never builds one.
ID_PATTERN = re.compile(r"^[0-9A-Za-z_\-]{6,120}$")


class AgentMCPError(RuntimeError):
    """Base for every MCP failure the agent turns into a user-facing message."""

    #: A short, actionable phrase; never contains a traceback.
    next_step: str = "请稍后重试。"

    def __init__(self, message: str, *, tool: str | None = None,
                 next_step: str | None = None, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.tool = tool
        self.detail = detail or {}
        if next_step:
            self.next_step = next_step

    def user_message(self) -> str:
        return f"{self}\n下一步：{self.next_step}"


class MCPConnectionError(AgentMCPError):
    """The server could not be started or the transport died."""

    next_step = "确认两个 MCP 服务已安装（pip install -e .），然后重试。"


class MCPProtocolError(AgentMCPError):
    """The server answered, but not in a shape this agent can use."""

    next_step = "这通常表示服务版本与 Agent 不匹配；请检查两个 MCP 的版本后重试。"


class MCPToolRejected(AgentMCPError):
    """The server refused the request: a business rule, not a crash."""

    next_step = "按提示修改请求内容后重试。"


class MCPToolTimeout(AgentMCPError):
    """The call did not answer within the configured timeout."""

    next_step = "调大 mcp.<server>.call_timeout_s，或稍后用同一编号重试。"


class MCPResponseSchemaError(AgentMCPError):
    """The answer parsed, but a required field was missing or malformed."""

    next_step = "请报告该问题；在修复前不要据此下结论。"


def validate_id(value: Any, label: str) -> str:
    """Validate an identifier supplied *to* a client method.

    A malformed argument is a caller error, so this raises ``ValueError``. Ids
    that come back *from* a server are checked with :func:`require_id` instead,
    where a bad value means the response did not have the promised shape.
    """
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError(f"{label}格式无效：只允许字母、数字、下划线和连字符，长度 6 到 120")
    return value


def require_id(value: Any, label: str) -> str:
    """Validate an identifier that came from a server response."""
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise MCPResponseSchemaError(f"{label}不是有效编号：{value!r}")
    return value


#: Keys that identify the hydro service's uniform response envelope.
ENVELOPE_KEYS = frozenset({"ok", "status", "summary", "facts", "data"})


def is_envelope(payload: dict[str, Any]) -> bool:
    """True when a response is the hydro service's uniform envelope.

    The spatial service predates that convention and returns each tool's own
    structure, some of which also carry an ``ok`` field with an unrelated
    meaning (``validate_data_bundle`` reports a validation verdict). Requiring
    the whole key set avoids mistaking one for the other.
    """
    return ENVELOPE_KEYS <= set(payload)


def normalize(payload: dict[str, Any], *, tool: str) -> dict[str, Any]:
    """Put every response into one shape the workflow can reason about.

    Nothing is invented. A response that is not already an envelope is wrapped
    with the server's own payload preserved verbatim under ``data``; its
    ``ok`` verdict (when it has one, as ``validate_data_bundle`` does),
    ``warnings`` and ``next_step`` are lifted to the envelope fields the
    workflow already reads. The workflow therefore never has to branch on
    which service it called.
    """
    if is_envelope(payload):
        return payload
    ok = bool(payload.get("ok", True))
    warnings = payload.get("warnings")
    next_step = payload.get("next_step")
    return {
        "ok": ok,
        "status": "ok" if ok else "not_ok",
        "summary": "",
        "facts": {},
        "warnings": list(warnings) if isinstance(warnings, list) else [],
        "limitations": [],
        "next_actions": [next_step] if isinstance(next_step, str) and next_step else [],
        "data": payload,
    }


#: ``Error executing tool <name>: `` and the SDK's own wrapper, stripped before
#: a server message is shown to a user.
_TOOL_ERROR_PREFIX = re.compile(r"^(?:Tool '[^']+' failed:\s*)?Error executing tool [A-Za-z0-9_]+:\s*")

#: Absolute paths the servers mention in their messages. The plan forbids
#: exposing internal file paths to a user, and the servers embed them (a missing
#: artifact is reported with its full path), so the adapter scrubs them.
_ABSOLUTE_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/](?:[^\\/\s\"']+[\\/])*[^\\/\s\"']*)"   # Windows drive path
    r"|(?:\\\\[^\\/\s]+[\\/][^\\/\s]+(?:[\\/][^\\/\s]+)*)"    # UNC path
    r"|(?:/(?:usr|home|opt|var|etc|tmp|mnt|data)(?:/[^/\s\"']+)+)"  # common POSIX root
)
_PATH_PLACEHOLDER = "<内部路径>"


def sanitize_message(text: str) -> str:
    """Make a server message safe and readable for a user.

    Drops the transport's ``Error executing tool ...`` wrapper and replaces
    absolute machine paths with a placeholder. Everything the server actually
    said about the problem is kept.
    """
    cleaned = _TOOL_ERROR_PREFIX.sub("", str(text).strip())
    cleaned = _ABSOLUTE_PATH.sub(_PATH_PLACEHOLDER, cleaned)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines[0][:300] if lines else ""


#: Environment variables the services read to locate their data, runs and
#: native components. The MCP client only inherits a small allow-list, so the
#: agent has to forward these explicitly: without them a service would fall back
#: to its own defaults and look in a different place than the agent expects.
FORWARDED_ENVIRONMENT_PREFIXES = ("WHS_", "CQ_")


def forwarded_environment() -> dict[str, str]:
    """The service settings the agent passes on to the servers it launches."""
    return {key: value for key, value in os.environ.items()
            if key.startswith(FORWARDED_ENVIRONMENT_PREFIXES)}


class MCPClientBase:
    """Launch one MCP server over stdio and call it by name.

    Subclasses expose one named method per allowed tool. There is deliberately
    no generic ``call_any_tool``: the workflow can only reach the tools this
    class names, so no user text can select a tool or an argument.
    """

    #: Human-readable service label used in messages.
    service_label: str = "MCP 服务"
    #: ``server_info.name`` the server must report.
    expected_name: str = ""
    #: Tools the server must expose for this agent to work.
    required_tools: frozenset[str] = frozenset()
    #: Explicitly supported additions; older servers may omit these.
    optional_tools: frozenset[str] = frozenset()

    def __init__(self, settings: ServerSettings, logger: logging.Logger | None = None) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger("slope_agent")
        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None
        self._tools: frozenset[str] = frozenset()
        self._connect_lock = asyncio.Lock()
        self._stderr_handle = None

    # ------------------------------------------------------------- lifecycle
    @property
    def connected(self) -> bool:
        return self._client is not None

    def _parameters(self) -> StdioServerParameters:
        return StdioServerParameters(
            command=self.settings.resolved_command(),
            args=list(self.settings.args),
            cwd=self.settings.resolved_cwd(),
            # Server messages are re-encoded on the wire; force UTF-8 so Chinese
            # summaries survive regardless of the console code page. The SDK
            # merges this over a small inherited allow-list — which does not
            # include the service roots, so they are forwarded here.
            env={"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                 **forwarded_environment()},
        )

    def _connect_target(self):
        """What to hand to ``mcp.Client``.

        A persistent service is reached over streamable-http, so closing the
        agent closes only the connection and leaves the service's jobs running.
        The stdio path launches a subprocess with an explicitly duplicated OS
        stderr handle: the SDK's default binds ``sys.stderr`` at import time, and
        a framework that replaces it would otherwise break the spawn. Tests
        override this to pass an in-process server.
        """
        if self.settings.transport == "streamable-http":
            return streamable_http_client(str(self.settings.url))
        return stdio_client(self._parameters(), errlog=self.server_stderr())

    def server_stderr(self):
        """A real OS stream for the server's own diagnostics.

        ``os.dup(2)`` is a genuine handle whatever ``sys.stderr`` currently is,
        so connecting works under pytest's capture and under a console wrapper
        that has no ``fileno``.
        """
        if self._stderr_handle is None:
            try:
                self._stderr_handle = os.fdopen(os.dup(2), "w", encoding="utf-8",
                                                errors="replace")
            except OSError:  # no stderr at all (a detached process)
                self._stderr_handle = open(os.devnull, "w", encoding="utf-8")
        return self._stderr_handle

    async def connect(self) -> None:
        """Start the server and verify its identity and tool set."""
        if self._client is not None:
            return
        async with self._connect_lock:
            if self._client is not None:
                return
            await self._connect_with_retries()

    async def _connect_with_retries(self) -> None:
        attempts = self.settings.handshake_retries + 1
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                await self._open()
                return
            except MCPProtocolError:
                raise
            except (AgentMCPError, OSError, MCPError, Exception) as exc:  # noqa: BLE001
                last = exc
                await self._teardown()
                if attempt < attempts:
                    self.logger.warning("%s 连接失败（第 %d 次），重试：%s",
                                        self.service_label, attempt, exc)
                    await asyncio.sleep(min(2.0 * attempt, 5.0))
        raise MCPConnectionError(
            f"{self.service_label}无法启动：{last}",
            next_step="确认命令可执行："
                      f"{self.settings.resolved_command()} {' '.join(self.settings.args)}",
        )

    async def _open(self) -> None:
        """Start the server and verify it, bounded by ``startup_timeout_s``.

        Without the bound a server that accepts the connection and then stalls
        hangs the agent forever; the timeout turns that into an error a user can
        act on.
        """
        try:
            await asyncio.wait_for(self._open_unbounded(),
                                   timeout=self.settings.startup_timeout_s)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            await self._teardown()
            raise MCPConnectionError(
                f"{self.service_label}启动超时（{self.settings.startup_timeout_s:.0f}s 内未完成握手）",
                next_step="确认服务地址或启动命令可达，或调大 mcp.<server>.startup_timeout_s。",
            ) from exc

    async def _open_unbounded(self) -> None:
        stack = AsyncExitStack()
        try:
            client = await stack.enter_async_context(Client(self._connect_target()))
        except BaseException:
            await stack.aclose()
            raise
        info = client.server_info
        if self.expected_name and info.name != self.expected_name:
            await stack.aclose()
            raise MCPProtocolError(
                f"{self.service_label}身份不符：期望 {self.expected_name!r}，实际 {info.name!r}",
                next_step="确认启动命令指向正确的服务，然后重试。",
            )
        listed = await client.list_tools()
        rows = listed.tools if hasattr(listed, "tools") else listed
        names = frozenset(tool.name for tool in rows)
        missing = self.required_tools - names
        if missing:
            await stack.aclose()
            raise MCPProtocolError(
                f"{self.service_label}缺少必需工具：{sorted(missing)}",
                detail={"available": sorted(names)},
            )
        try:
            self.validate_tool_contracts(rows)
        except MCPProtocolError:
            await stack.aclose()
            raise
        self._stack = stack
        self._client = client
        self._tools = names
        self.logger.debug("%s 已连接：%s v%s，工具 %d 个",
                          self.service_label, info.name, info.version, len(names))

    def validate_tool_contracts(self, rows) -> None:
        """Service-specific compatibility checks after tools are listed."""

    async def close(self) -> None:
        await self._teardown()

    async def _teardown(self) -> None:
        stack, self._stack = self._stack, None
        self._client = None
        self._tools = frozenset()
        if stack is not None:
            try:
                await stack.aclose()
            except Exception as exc:  # noqa: BLE001 - closing must never raise
                self.logger.debug("%s 关闭时出现异常：%s", self.service_label, exc)
        if self._stderr_handle is not None:
            try:
                self._stderr_handle.close()
            except Exception:  # noqa: BLE001 - a closed stream is not an error
                pass
            self._stderr_handle = None

    async def __aenter__(self) -> "MCPClientBase":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @property
    def available_tools(self) -> frozenset[str]:
        return self._tools

    # ------------------------------------------------------------------ calls
    async def _call(
        self,
        tool: str,
        arguments: dict[str, Any] | None = None,
        *,
        expect_ok: bool = True,
    ) -> dict[str, Any]:
        """Call one tool and return its structured envelope.

        ``expect_ok=False`` is for tools that legitimately report ``ok: false``
        as a result (a run that finished with ``qa_failed``), where the caller
        must inspect the status rather than treat it as a refusal.
        """
        if tool not in self.required_tools | self.optional_tools:
            # Defensive: a tool the class did not declare can never be reached.
            raise MCPProtocolError(f"{self.service_label}不允许调用工具：{tool}")
        if not self.connected:
            await self.connect()
        assert self._client is not None  # connect() raises otherwise
        if tool in self.optional_tools and tool not in self._tools:
            raise MCPProtocolError(f"{self.service_label}当前版本不支持工具：{tool}")

        try:
            result = await self._client.call_tool(
                tool, arguments or {}, read_timeout_seconds=self.settings.call_timeout_s,
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise MCPToolTimeout(f"{self.service_label}.{tool} 超时",
                                 tool=tool) from exc
        except MCPError as exc:
            self._raise_protocol_error(exc, tool)
        except (BrokenPipeError, ConnectionError, OSError) as exc:
            await self._teardown()
            raise MCPConnectionError(f"{self.service_label}连接中断：{exc}", tool=tool) from exc
        except Exception as exc:  # noqa: BLE001 - last resort, never leak a traceback
            raise MCPConnectionError(f"{self.service_label}.{tool} 调用失败：{exc}",
                                     tool=tool) from exc

        if result.is_error:
            raise MCPToolRejected(
                self._error_text(result, tool), tool=tool,
                detail={"content": self._texts(result)},
            )
        payload = result.structured_content
        if not isinstance(payload, dict):
            raise MCPResponseSchemaError(
                f"{self.service_label}.{tool} 未返回结构化结果",
                tool=tool,
            )
        envelope = normalize(payload, tool=tool)
        if expect_ok and envelope.get("ok") is False:
            raise MCPToolRejected(
                str(envelope.get("summary") or f"{self.service_label}拒绝了该请求"),
                tool=tool,
                next_step=(envelope.get("next_actions") or [None])[0] or MCPToolRejected.next_step,
                detail=envelope,
            )
        return envelope

    async def _read_resource(
        self,
        uri: str,
        *,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
    ) -> dict[str, Any]:
        """Read one resource and return its bytes.

        The message channel answers a resource read in a single response, so the
        whole file arrives at once — which is why only files below a configured
        ceiling travel this way, and why the caller must have a size in hand
        before asking (``expected_size``). Over stdio the payload comes back
        base64-encoded and is decoded here; in memory it arrives as bytes.

        ``expected_sha256`` is checked when given: bytes that do not match the
        digest the service published are not the artifact, and returning them
        would put a stale file behind a registered name.
        """
        if not self.connected:
            await self.connect()
        assert self._client is not None
        try:
            result = await asyncio.wait_for(
                self._client.read_resource(uri),
                timeout=self.settings.call_timeout_s,
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise MCPToolTimeout(f"{self.service_label}读取资源超时：{uri}") from exc
        except MCPError as exc:
            message = str(getattr(exc, "message", None) or exc)
            if "timed out" in message.lower():
                raise MCPToolTimeout(
                    f"{self.service_label}读取资源超时：超过 "
                    f"mcp.<server>.call_timeout_s = {self.settings.call_timeout_s:g} 秒",
                    next_step="请调大 mcp.<server>.call_timeout_s 后重启智能体，再重试下载。",
                ) from exc
            raise MCPToolRejected(f"{self.service_label}拒绝读取该资源：{message}") from exc
        except (BrokenPipeError, ConnectionError, OSError) as exc:
            await self._teardown()
            raise MCPConnectionError(f"{self.service_label}连接中断：{exc}") from exc
        except Exception as exc:  # noqa: BLE001 - never leak a traceback
            raise MCPConnectionError(f"{self.service_label}读取资源失败：{exc}") from exc

        contents = getattr(result, "contents", None) or []
        if not contents:
            raise MCPResponseSchemaError(f"{self.service_label}资源为空：{uri}")
        block = contents[0]
        blob = getattr(block, "blob", None)
        if blob is None:
            raise MCPResponseSchemaError(f"{self.service_label}资源不是二进制内容：{uri}")
        raw = base64.b64decode(blob) if isinstance(blob, str) else bytes(blob)
        if expected_size is not None and len(raw) != expected_size:
            raise MCPResponseSchemaError(
                f"{self.service_label}资源长度与登记不符：{len(raw)} ≠ {expected_size}")
        if expected_sha256:
            import hashlib

            if hashlib.sha256(raw).hexdigest() != expected_sha256:
                raise MCPResponseSchemaError(f"{self.service_label}资源摘要与登记不符：{uri}")
        return {
            "data": raw,
            "mime_type": str(getattr(block, "mime_type", None) or "application/octet-stream"),
            "size_bytes": len(raw),
        }

    def _raise_protocol_error(self, exc: MCPError, tool: str) -> None:
        """Map a transport-level MCP error onto the failure a user can act on.

        The SDK reports a request that ran out of time as a plain protocol
        error (``Request 'tools/call' timed out``). Mapping that to the generic
        protocol failure told the reader to check *versions* — measured on a
        real run: a delineation that legitimately needs 3–5 minutes under a
        180 s timeout was reported as "服务版本与 Agent 不匹配", which sends
        everyone after the wrong problem. Timeouts are recognised by their
        message and reported as timeouts, with the actual setting to raise.
        """
        message = str(getattr(exc, "message", None) or exc)
        if "timed out" in message.lower():
            raise MCPToolTimeout(
                f"{self.service_label}.{tool} 超时：超过 mcp.<server>.call_timeout_s = "
                f"{self.settings.call_timeout_s:g} 秒仍未返回",
                tool=tool,
                next_step=(
                    "这次调用确实需要更久（例如流域划分会自动放大处理范围重试）。"
                    f"请在配置里调大 mcp.<server>.call_timeout_s（当前 "
                    f"{self.settings.call_timeout_s:g} 秒）并重启智能体后重试；"
                    "已经开始的运行不会因此丢失，可用同一编号继续查询。"),
            ) from exc
        raise MCPProtocolError(
            f"{self.service_label}.{tool} 协议错误：{message}", tool=tool,
            next_step=("这通常表示这次请求未能完成（服务被中断或响应不可用）；"
                       "请用 scripts/check_mcp_services.ps1 核对服务状态后重试。"),
        ) from exc

    @staticmethod
    def _texts(result) -> list[str]:
        return [block.text for block in result.content
                if getattr(block, "type", None) == "text"]

    @classmethod
    def _error_text(cls, result, tool: str) -> str:
        texts = cls._texts(result)
        if not texts:
            return f"{tool} 调用被拒绝"
        # The server's own message is user-facing; the transport wrapper and any
        # absolute machine path are not.
        message = sanitize_message(texts[0])
        return message or f"{tool} 调用被拒绝"

    async def _call_image(
        self,
        tool: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a tool that returns an image, keeping the payload opaque.

        The agent never decodes or rewrites image bytes; it carries them to
        whatever is displaying the conversation.
        """
        if tool not in self.required_tools | self.optional_tools:
            raise MCPProtocolError(f"{self.service_label}不允许调用工具：{tool}")
        if not self.connected:
            await self.connect()
        assert self._client is not None
        if tool in self.optional_tools and tool not in self._tools:
            raise MCPProtocolError(f"{self.service_label}当前版本不支持工具：{tool}")
        try:
            result = await self._client.call_tool(
                tool, arguments or {}, read_timeout_seconds=self.settings.call_timeout_s,
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise MCPToolTimeout(f"{self.service_label}.{tool} 超时", tool=tool) from exc
        except MCPError as exc:
            self._raise_protocol_error(exc, tool)
        except (BrokenPipeError, ConnectionError, OSError) as exc:
            await self._teardown()
            raise MCPConnectionError(f"{self.service_label}连接中断：{exc}", tool=tool) from exc
        except Exception as exc:  # noqa: BLE001
            raise MCPConnectionError(f"{self.service_label}.{tool} 调用失败：{exc}",
                                     tool=tool) from exc

        if result.is_error:
            raise MCPToolRejected(self._error_text(result, tool), tool=tool)
        images = [block for block in result.content
                  if getattr(block, "type", None) == "image"]
        if not images:
            raise MCPResponseSchemaError(f"{self.service_label}.{tool} 未返回图像", tool=tool)
        block = images[0]
        return {
            "mime_type": getattr(block, "mime_type", "image/png"),
            "data": getattr(block, "data", ""),
            "texts": self._texts(result),
            "count": len(images),
        }


def envelope_facts(payload: dict[str, Any]) -> dict[str, Any]:
    """The facts block of a tool envelope, always a dict."""
    facts = payload.get("facts")
    return facts if isinstance(facts, dict) else {}


def envelope_data(payload: dict[str, Any]) -> dict[str, Any]:
    """The data block of a tool envelope, always a dict."""
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def envelope_summary(payload: dict[str, Any]) -> str:
    return str(payload.get("summary") or "")


def envelope_status(payload: dict[str, Any]) -> str:
    return str(payload.get("status") or "")


def require_fields(payload: dict[str, Any], fields: Sequence[str], *, tool: str) -> None:
    """Fail with a schema error when a response omits a field the agent needs."""
    missing = [field for field in fields if field not in payload]
    if missing:
        raise MCPResponseSchemaError(f"{tool} 响应缺少字段：{missing}", tool=tool)

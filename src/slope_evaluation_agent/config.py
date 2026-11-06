"""Configuration for the slope evaluation agent.

Resolution order for the config file:

1. ``SLOPE_AGENT_CONFIG`` environment variable;
2. ``config/agent.default.yaml`` in the repository;
3. built-in defaults.

An absent config file is not an error: the agent must start with no
configuration at all and with no LLM key. Everything here is validated, so a
malformed value fails loudly at startup instead of silently changing workflow
behaviour.

The LLM API key is held in a :class:`~pydantic.SecretStr` and is only ever read
from the environment. It is excluded from ``model_dump`` output, so it cannot
reach a session file, an event log or a response.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


#: Repository root, derived from this file's location (``src/<pkg>/config.py``).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_CONFIG = "SLOPE_AGENT_CONFIG"
ENV_LLM_BASE_URL = "SLOPE_AGENT_LLM_BASE_URL"
ENV_LLM_MODEL = "SLOPE_AGENT_LLM_MODEL"
ENV_LLM_API_KEY = "SLOPE_AGENT_LLM_API_KEY"
ENV_LLM_PROVIDER = "SLOPE_AGENT_LLM_PROVIDER"


class ServerSettings(BaseModel):
    """How to reach one MCP server.

    Two transports:

    * ``streamable-http`` — a persistent local service the agent connects to.
      This is what the agent uses for long runs: closing the agent closes only
      the connection, and the service keeps its jobs.
    * ``stdio`` — a subprocess the agent owns. Convenient, and the only option
      without a separately started service, but the job ends with the agent.

    ``command`` and ``cwd`` default to the running interpreter and the project
    root, so no absolute machine path is written into the config.
    """

    transport: Literal["stdio", "streamable-http"] = "stdio"
    #: Required for ``streamable-http``; loopback by default so a research
    #: service is never exposed to the local network.
    url: str | None = Field(default=None, description="例如 http://127.0.0.1:8765/mcp")
    command: str | None = Field(default=None, description="Executable; 默认当前解释器")
    args: list[str] = Field(default_factory=list)
    cwd: str | None = Field(default=None, description="工作目录；默认项目根目录")
    startup_timeout_s: float = Field(default=30.0, gt=0, le=600)
    call_timeout_s: float = Field(default=180.0, gt=0, le=3600)
    #: Long-running tools (start_hydro_run) return immediately, so they use the
    #: normal call timeout; this covers a slow first handshake after a cold start.
    handshake_retries: int = Field(default=2, ge=0, le=10)

    @model_validator(mode="after")
    def _url_required_for_http(self) -> "ServerSettings":
        if self.transport == "streamable-http":
            if not self.url:
                raise ValueError("transport=streamable-http 需要设置 url")
            if not self.url.startswith(("http://", "https://")):
                raise ValueError(f"url 必须是 http(s) 地址：{self.url}")
            host = self.url.split("//", 1)[1].split("/", 1)[0].split(":")[0]
            if host not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError(
                    f"默认只允许本机地址；如需暴露服务请显式修改配置（当前 {host}）")
        return self

    def resolved_command(self) -> str:
        return self.command or sys.executable

    def resolved_cwd(self) -> str:
        return str(Path(self.cwd).expanduser().resolve()) if self.cwd else str(PROJECT_ROOT)


#: Module invoked when a config overrides some other field of a server but
#: leaves ``args`` unset. Without this, overriding one timeout would silently
#: blank the launch arguments and the server would never start.
DEFAULT_SERVER_ARGS = {
    "spatial": ["-m", "chongqing_spatial_mcp.server"],
    "hydro": ["-m", "watershed_hydro_mcp.server"],
}


class MCPSettings(BaseModel):
    spatial: ServerSettings = Field(default_factory=lambda: ServerSettings(
        args=list(DEFAULT_SERVER_ARGS["spatial"])))
    hydro: ServerSettings = Field(default_factory=lambda: ServerSettings(
        args=list(DEFAULT_SERVER_ARGS["hydro"])))

    @model_validator(mode="after")
    def _fill_launch_arguments(self) -> "MCPSettings":
        for name, default in DEFAULT_SERVER_ARGS.items():
            server = getattr(self, name)
            if not server.args:
                setattr(self, name, server.model_copy(update={"args": list(default)}))
        return self


class SessionSettings(BaseModel):
    root: str = Field(default="runs/agent/sessions")
    #: Conversation turns kept verbatim; older turns are dropped but the
    #: structured session state is always kept in full.
    max_conversation_turns: int = Field(default=20, ge=2, le=200)
    max_text_chars_per_turn: int = Field(default=2000, ge=100, le=20000)
    #: Events log is append-only; this caps a single logged payload.
    max_event_payload_chars: int = Field(default=4000, ge=200, le=100000)

    def resolved_root(self) -> Path:
        path = Path(self.root).expanduser()
        return path if path.is_absolute() else (PROJECT_ROOT / path)


class PollingSettings(BaseModel):
    """Polling cadence for long-running hydro jobs and visualizations."""

    initial_interval_s: float = Field(default=5.0, gt=0, le=300)
    maximum_interval_s: float = Field(default=30.0, gt=0, le=1800)
    backoff_factor: float = Field(default=1.5, ge=1.0, le=5.0)
    #: A single blocking wait gives up after this; the job keeps running.
    maximum_wait_s: float = Field(default=1800.0, gt=0, le=86400)

    @model_validator(mode="after")
    def _ordered(self) -> "PollingSettings":
        if self.maximum_interval_s < self.initial_interval_s:
            raise ValueError("polling.maximum_interval_s 不能小于 initial_interval_s")
        return self


class DownloadSettings(BaseModel):
    """Limits on what one conversation may download.

    Two ceilings with different jobs. ``max_file_bytes`` is the policy: a file
    larger than this is refused before anything is read. ``resource_read_limit_
    bytes`` is a transport choice: the MCP resource channel answers in a single
    message, so files above it are streamed from disk by the same-machine
    adapter instead. The measured cost of the resource channel is roughly half a
    second per megabyte over stdio (base64 in one JSON-RPC response), which is
    why the default is small.
    """

    #: The most a single download may be. 1 GiB, per the plan's initial value.
    max_file_bytes: int = Field(default=1024 ** 3, ge=1024 ** 2, le=8 * 1024 ** 3)
    #: At or below this, a file travels through the MCP resource channel.
    resource_read_limit_bytes: int = Field(default=4 * 1024 ** 2,
                                           ge=64 * 1024, le=64 * 1024 * 1024)
    #: Bytes per chunk while streaming to the browser.
    chunk_bytes: int = Field(default=1024 * 1024, ge=32 * 1024, le=16 * 1024 * 1024)

    @model_validator(mode="after")
    def _ordered(self) -> "DownloadSettings":
        if self.resource_read_limit_bytes > self.max_file_bytes:
            raise ValueError("download.resource_read_limit_bytes 不能大于 max_file_bytes")
        return self


class DialogueSettings(BaseModel):
    """How much the conversational layer may do.

    Calculation steps require ``workflow_confidence_threshold``. A model that
    explicitly chose a location task can open a read-only drawing map at
    ``clarification_confidence_threshold``; this cannot submit or confirm a
    boundary. An expressed ambiguity still asks rather than opening the map.
    """

    enabled: bool = True
    #: At or above this, a request with every required entity is executed.
    workflow_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    #: Below this, the model has not understood enough to do anything but ask.
    clarification_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    #: How many recent turns the router is shown. Bounded: a long conversation
    #: must not grow the prompt without limit.
    recent_turns: int = Field(default=16, ge=0, le=50)
    #: Whether ordinary conversation is answered at all. Turning both off leaves
    #: exactly the pre-V2 behaviour: rules, and the deterministic templates.
    allow_general_chat: bool = True
    #: General engineering answers may use the model's knowledge, separately
    #: from the strictly evidence-bound computation reply pipeline.
    allow_general_llm: bool = True
    allow_out_of_scope_reply: bool = True

    @model_validator(mode="after")
    def _ordered(self) -> "DialogueSettings":
        if self.clarification_confidence_threshold > self.workflow_confidence_threshold:
            raise ValueError(
                "dialogue.clarification_confidence_threshold 不能高于 workflow_confidence_threshold")
        return self


class LLMSettings(BaseModel):
    """Optional language model.

    The agent must work fully without this. ``enabled`` defaults to ``None``,
    meaning "on if a base URL and model are configured"; setting it to ``False``
    forces the deterministic path even when a key is present.
    """

    provider: Literal["openai-compatible", "deepseek", "anthropic", "grok", "gemini", "qwen"] = "openai-compatible"
    enabled: bool | None = None
    base_url: str | None = None
    model: str | None = None
    api_format: Literal["chat_completions", "responses", "anthropic_messages", "gemini_generate_content"] = "chat_completions"
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"] | None = None
    api_key: SecretStr | None = Field(default=None, exclude=True)
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    maximum_tokens: int = Field(default=1200, ge=64, le=32000)
    request_timeout_s: float = Field(default=60.0, gt=0, le=600)
    max_retries: int = Field(default=1, ge=0, le=5)
    json_mode: bool = True
    #: DeepSeek enables reasoning by default; short JSON contracts need a
    #: budget for the final answer. Other providers never receive this field.
    deepseek_thinking: Literal["enabled", "disabled"] = "disabled"
    anthropic_thinking: Literal["default", "disabled", "adaptive", "enabled"] = "default"
    anthropic_thinking_budget_tokens: int = Field(default=1024, ge=1024, le=31000)
    chat_streaming: bool = False
    qwen_thinking: Literal["default", "enabled", "disabled"] = "default"
    qwen_thinking_budget_tokens: int | None = Field(default=None, ge=0, le=262144)
    gemini_thinking_budget_tokens: int | None = Field(default=None, ge=-1, le=32768)

    @model_validator(mode="before")
    @classmethod
    def _anthropic_default_protocol(cls, values):
        if isinstance(values, dict):
            values = dict(values)
            if values.get("provider") == "anthropic" and "api_format" not in values:
                values["api_format"] = "anthropic_messages"
            if values.get("provider") == "qwen" and "chat_streaming" not in values:
                values["chat_streaming"] = True
        return values

    @model_validator(mode="after")
    def _additional_provider_options(self):
        if self.provider in {"grok", "gemini", "qwen"}:
            allowed_formats = {"chat_completions", "responses"} if self.provider == "grok" else {"chat_completions"}
            if self.provider == "gemini":
                allowed_formats.add("gemini_generate_content")
            if self.api_format not in allowed_formats:
                raise ValueError("所选模型服务与接口协议不匹配")
        if self.api_format == "gemini_generate_content" and self.provider != "gemini":
            raise ValueError("Gemini原生协议请选择Gemini服务类型")
        if self.provider == "grok" and self.reasoning_effort not in {None, "low", "medium", "high", "xhigh"}:
            raise ValueError("Grok仅接受该模型支持的low、medium、high、xhigh，或默认不指定")
        if self.provider == "gemini":
            if self.reasoning_effort not in {None, "none", "minimal", "low", "medium", "high"}:
                raise ValueError("Gemini推理强度请按模型选择none、minimal、low、medium、high或默认")
            if self.gemini_thinking_budget_tokens is not None and self.reasoning_effort is not None:
                raise ValueError("Gemini思考等级与思考预算不能同时指定")
        if self.provider == "qwen":
            if self.reasoning_effort == "ultra":
                raise ValueError("Qwen不使用ultra强度")
            if self.qwen_thinking_budget_tokens is not None and self.reasoning_effort is not None:
                raise ValueError("Qwen思考预算与effort不能同时指定")
            if self.qwen_thinking == "disabled" and (self.qwen_thinking_budget_tokens is not None or self.reasoning_effort not in {None, "none"}):
                raise ValueError("Qwen关闭思考时不能同时指定思考预算或其他effort")
            if self.qwen_thinking == "enabled" and self.reasoning_effort == "none":
                raise ValueError("Qwen开启思考不能同时指定none强度")
        return self

    @model_validator(mode="after")
    def _anthropic_options(self):
        if self.provider == "anthropic" and self.api_format != "anthropic_messages":
            raise ValueError("Anthropic服务请选择Messages协议")
        if self.api_format != "anthropic_messages":
            return self
        if self.provider == "deepseek":
            raise ValueError("DeepSeek服务不使用Anthropic Messages协议")
        if self.reasoning_effort not in {None, "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Claude effort仅支持low、medium、high、xhigh、max；关闭思考请使用思考模式")
        if self.temperature is not None and self.temperature > 1:
            raise ValueError("Claude温度范围为0到1，或留空使用服务默认值")
        if self.anthropic_thinking in {"adaptive", "enabled"} and self.temperature is not None:
            raise ValueError("Claude开启思考时请选择服务默认、不发送温度")
        if self.anthropic_thinking == "enabled" and self.anthropic_thinking_budget_tokens >= self.maximum_tokens:
            raise ValueError("Claude思考预算必须小于生成token上限，为最终回答保留空间")
        return self

    def is_available(self) -> bool:
        """True only when the model can actually be called."""
        if self.enabled is False:
            return False
        endpoint_ready = bool(self.base_url and self.model)
        if self.provider in {"deepseek", "anthropic", "grok", "gemini", "qwen"} or self.api_format == "anthropic_messages":
            return endpoint_ready and bool(
                self.api_key is not None and self.api_key.get_secret_value())
        return endpoint_ready

    def unavailable_reason(self) -> str | None:
        if self.is_available():
            return None
        if self.enabled is False:
            return "配置中显式关闭了语言模型"
        if self.base_url and self.model and not self.api_key:
            if self.provider == "deepseek":
                return "尚未设置 DeepSeek API Key"
            if self.provider == "anthropic" or self.api_format == "anthropic_messages":
                return "尚未设置 Claude API Key"
            if self.provider in {"grok", "gemini", "qwen"}:
                return "尚未设置所选模型服务的API Key"
        if not self.base_url and not self.model:
            return "未配置语言模型地址与模型名"
        return "语言模型配置不完整"


class AgentSettings(BaseModel):
    """The whole agent configuration."""

    schema_version: int = 1
    language: Literal["zh", "en"] = "zh"
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    sessions: SessionSettings = Field(default_factory=SessionSettings)
    polling: PollingSettings = Field(default_factory=PollingSettings)
    download: DownloadSettings = Field(default_factory=DownloadSettings)
    dialogue: DialogueSettings = Field(default_factory=DialogueSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    #: Where the resolved config came from, for diagnostics. Never contains secrets.
    source: str | None = None

    @field_validator("language")
    @classmethod
    def _known_language(cls, value: str) -> str:
        if value not in {"zh", "en"}:
            raise ValueError(f"不支持的语言：{value}")
        return value

    # ---------------------------------------------------------------- helpers
    def redacted(self) -> dict[str, Any]:
        """A log-safe view: the API key is reported as present or absent."""
        payload = self.model_dump(mode="json")
        key = self.llm.api_key
        payload["llm"]["api_key"] = "set" if key is not None and key.get_secret_value() else "unset"
        payload["llm"]["available"] = self.llm.is_available()
        payload["llm"]["unavailable_reason"] = self.llm.unavailable_reason()
        return payload


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件必须是映射结构：{path}")
    return payload


def _apply_environment(payload: dict[str, Any]) -> dict[str, Any]:
    """Environment variables override the file, and are the only key source."""
    merged = {**payload}
    llm = dict(merged.get("llm") or {})
    if os.environ.get(ENV_LLM_PROVIDER):
        llm["provider"] = os.environ[ENV_LLM_PROVIDER]
    if os.environ.get(ENV_LLM_BASE_URL):
        llm["base_url"] = os.environ[ENV_LLM_BASE_URL]
    if os.environ.get(ENV_LLM_MODEL):
        llm["model"] = os.environ[ENV_LLM_MODEL]
    if os.environ.get(ENV_LLM_API_KEY):
        llm["api_key"] = os.environ[ENV_LLM_API_KEY]
    if llm:
        merged["llm"] = llm
    return merged


def resolve_config_path(explicit: str | Path | None = None) -> Path | None:
    """The config file that will be read, or ``None`` when defaults apply."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"指定的配置文件不存在：{path}")
        return path
    from_env = os.environ.get(ENV_CONFIG)
    if from_env:
        path = Path(from_env).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{ENV_CONFIG} 指向的配置文件不存在：{path}")
        return path
    default = PROJECT_ROOT / "config" / "agent.default.yaml"
    return default if default.is_file() else None


def load_settings(explicit: str | Path | None = None) -> AgentSettings:
    """Load and validate the agent configuration.

    A missing file yields the built-in defaults, which are safe: no LLM, no
    absolute paths, sessions under ``runs/agent/sessions``.
    """
    path = resolve_config_path(explicit)
    payload = _apply_environment(_read_config_file(path)) if path else _apply_environment({})
    # ``source`` is informational; keep it out of the model to avoid it being
    # mistaken for a setting.
    payload.pop("source", None)
    settings = AgentSettings.model_validate(payload)
    return settings.model_copy(update={"source": str(path) if path else "built-in defaults"})


# --------------------------------------------------------------------- logging

_REDACTION = "***REDACTED***"


class _SecretRedactingFilter(logging.Filter):
    """Replace any live secret with a placeholder before a record is emitted."""

    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self._secrets = [value for value in secrets if value and len(value) >= 6]

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - never let logging break the agent
            return True
        redacted = message
        for secret in self._secrets:
            if secret in redacted:
                redacted = redacted.replace(secret, _REDACTION)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def configure_logging(settings: AgentSettings, *, verbose: bool = False,
                      stream: Any = None) -> logging.Logger:
    """Install a logger whose records can never contain the configured key."""
    logger = logging.getLogger("slope_agent")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                           datefmt="%H:%M:%S"))
    secrets = []
    if settings.llm.api_key is not None:
        secrets.append(settings.llm.api_key.get_secret_value())
    handler.addFilter(_SecretRedactingFilter(secrets))
    logger.addHandler(handler)
    logger.propagate = False
    return logger

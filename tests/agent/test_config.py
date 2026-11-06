"""Phase 1 acceptance: configuration resolution, defaults and secret handling."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from slope_evaluation_agent import DISCLAIMER_ZH, SCOPE_STATEMENT, __version__
from slope_evaluation_agent.config import (
    AgentSettings,
    ENV_LLM_API_KEY,
    LLMSettings,
    PROJECT_ROOT,
    PollingSettings,
    configure_logging,
    load_settings,
    resolve_config_path,
)
from slope_evaluation_agent.cli import main as cli_main


class TestRuleResolution:
    def test_missing_file_falls_back_to_built_in_defaults(self, monkeypatch, tmp_path):
        monkeypatch.setattr("slope_evaluation_agent.config.PROJECT_ROOT", tmp_path)
        settings = load_settings()
        assert settings.source == "built-in defaults"
        assert settings.llm.is_available() is False

    def test_environment_variable_selects_the_file(self, monkeypatch, write_config):
        path = write_config({"language": "en"})
        monkeypatch.setenv("SLOPE_AGENT_CONFIG", str(path))
        assert resolve_config_path() == path
        assert load_settings().language == "en"

    def test_explicit_path_wins_over_environment(self, monkeypatch, write_config):
        explicit = write_config({"language": "en"}, "explicit.yaml")
        other = write_config({"language": "zh"}, "other.yaml")
        monkeypatch.setenv("SLOPE_AGENT_CONFIG", str(other))
        assert load_settings(explicit).language == "en"

    def test_missing_explicit_path_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_settings(tmp_path / "absent.yaml")

    def test_missing_environment_path_is_an_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SLOPE_AGENT_CONFIG", str(tmp_path / "absent.yaml"))
        with pytest.raises(FileNotFoundError):
            resolve_config_path()

    def test_repository_default_file_loads(self):
        settings = load_settings()
        assert settings.source is not None
        assert settings.mcp.spatial.args == ["-m", "chongqing_spatial_mcp.server"]
        assert settings.mcp.hydro.args == ["-m", "watershed_hydro_mcp.server"]


class TestSafeDefaults:
    def test_no_absolute_machine_paths_in_defaults(self, monkeypatch, tmp_path):
        monkeypatch.setattr("slope_evaluation_agent.config.PROJECT_ROOT", tmp_path)
        settings = load_settings()
        payload = json.dumps(settings.redacted(), ensure_ascii=False)
        for marker in ("D:\\paper", "Yongchuan", "F:\\QGIS", "anaconda"):
            assert marker not in payload

    def test_interpreter_and_cwd_default_to_this_project(self, monkeypatch, tmp_path):
        monkeypatch.setattr("slope_evaluation_agent.config.PROJECT_ROOT", tmp_path)
        settings = load_settings()
        assert settings.mcp.spatial.resolved_command() == sys.executable
        assert Path(settings.mcp.spatial.resolved_cwd()) == tmp_path

    def test_sessions_default_under_the_project(self, monkeypatch, tmp_path):
        monkeypatch.setattr("slope_evaluation_agent.config.PROJECT_ROOT", tmp_path)
        settings = load_settings()
        assert settings.sessions.resolved_root() == tmp_path / "runs" / "agent" / "sessions"

    def test_relative_session_root_resolves_against_the_project(self, monkeypatch, tmp_path):
        monkeypatch.setattr("slope_evaluation_agent.config.PROJECT_ROOT", tmp_path)
        settings = AgentSettings.model_validate({"sessions": {"root": "custom/sessions"}})
        assert settings.sessions.resolved_root() == tmp_path / "custom" / "sessions"

    def test_absolute_session_root_is_honoured(self, tmp_path):
        settings = AgentSettings.model_validate({"sessions": {"root": str(tmp_path)}})
        assert settings.sessions.resolved_root() == tmp_path

    def test_llm_disabled_by_default_without_configuration(self, monkeypatch, tmp_path):
        monkeypatch.setattr("slope_evaluation_agent.config.PROJECT_ROOT", tmp_path)
        settings = load_settings()
        assert settings.llm.is_available() is False
        assert settings.llm.unavailable_reason() is not None


class TestLLMConfiguration:
    def test_base_url_and_model_enable_the_model(self):
        settings = LLMSettings(base_url="https://example.invalid/v1", model="m")
        assert settings.is_available() is True

    def test_missing_model_keeps_it_unavailable(self):
        settings = LLMSettings(base_url="https://example.invalid/v1")
        assert settings.is_available() is False

    def test_explicit_disable_overrides_a_complete_configuration(self):
        settings = LLMSettings(enabled=False, base_url="https://example.invalid/v1", model="m")
        assert settings.is_available() is False
        assert "关闭" in settings.unavailable_reason()

    def test_environment_variables_fill_the_model_settings(self, monkeypatch, tmp_path):
        monkeypatch.setattr("slope_evaluation_agent.config.PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("SLOPE_AGENT_LLM_BASE_URL", "https://example.invalid/v1")
        monkeypatch.setenv("SLOPE_AGENT_LLM_MODEL", "test-model")
        monkeypatch.setenv(ENV_LLM_API_KEY, "sk-test-key-not-real")
        settings = load_settings()
        assert settings.llm.is_available() is True
        assert settings.llm.model == "test-model"

    def test_temperature_must_be_zero_for_deterministic_parsing(self):
        with pytest.raises(ValueError):
            LLMSettings(temperature=3.0)

    def test_default_temperature_is_zero(self):
        assert LLMSettings().temperature == 0.0


class TestSecretHandling:
    SECRET = "sk-super-secret-value-12345"

    def _settings(self, monkeypatch, tmp_path) -> AgentSettings:
        monkeypatch.setattr("slope_evaluation_agent.config.PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("SLOPE_AGENT_LLM_BASE_URL", "https://example.invalid/v1")
        monkeypatch.setenv("SLOPE_AGENT_LLM_MODEL", "test-model")
        monkeypatch.setenv(ENV_LLM_API_KEY, self.SECRET)
        return load_settings()

    def test_key_never_reaches_a_serialised_dump(self, monkeypatch, tmp_path):
        settings = self._settings(monkeypatch, tmp_path)
        assert self.SECRET not in json.dumps(settings.model_dump(mode="json"), ensure_ascii=False)
        assert self.SECRET not in json.dumps(settings.redacted(), ensure_ascii=False)
        assert self.SECRET not in repr(settings)

    def test_redacted_view_reports_presence_not_value(self, monkeypatch, tmp_path):
        payload = self._settings(monkeypatch, tmp_path).redacted()
        assert payload["llm"]["api_key"] == "set"
        assert payload["llm"]["available"] is True

    def test_redacted_view_reports_absence(self, monkeypatch, tmp_path):
        monkeypatch.setattr("slope_evaluation_agent.config.PROJECT_ROOT", tmp_path)
        payload = load_settings().redacted()
        assert payload["llm"]["api_key"] == "unset"

    def test_log_records_are_scrubbed(self, monkeypatch, tmp_path):
        import io

        settings = self._settings(monkeypatch, tmp_path)
        buffer = io.StringIO()
        logger = configure_logging(settings, stream=buffer)
        logger.info("calling model with key %s", self.SECRET)
        logger.warning("%s", json.dumps(settings.redacted(), ensure_ascii=False))
        emitted = buffer.getvalue()
        assert self.SECRET not in emitted
        assert "***REDACTED***" in emitted

    def test_secret_is_excluded_from_a_plain_model_dump(self):
        settings = LLMSettings(api_key="sk-secret-value-abc123")
        assert "api_key" not in settings.model_dump()
        assert "sk-secret-value-abc123" not in json.dumps(settings.model_dump())
        # The value is still retrievable for the one call that needs it.
        assert settings.api_key.get_secret_value() == "sk-secret-value-abc123"

    def test_deepseek_profile_waits_for_a_key(self):
        settings = LLMSettings(
            provider="deepseek",
            base_url="https://api.deepseek.com",
            model="deepseek-flash",
        )
        assert settings.is_available() is False
        assert "API Key" in settings.unavailable_reason()
        # Model validation is the same path the real configuration loader uses.
        configured = LLMSettings.model_validate(settings.model_dump() | {
            "api_key": "sk-placeholder-not-real"
        })
        assert configured.is_available() is True

    def test_repository_deepseek_profile_is_safe_without_a_key(self, monkeypatch):
        monkeypatch.delenv(ENV_LLM_API_KEY, raising=False)
        profile = Path(__file__).resolve().parents[2] / "config" / "agent.deepseek.yaml"
        settings = load_settings(profile)
        assert settings.llm.provider == "deepseek"
        assert settings.llm.base_url == "https://api.deepseek.com"
        assert settings.llm.is_available() is False
        assert settings.mcp.hydro.transport == "streamable-http"


class TestValidation:
    def test_unknown_language_is_rejected(self):
        with pytest.raises(ValueError):
            AgentSettings.model_validate({"language": "fr"})

    def test_non_mapping_config_is_rejected(self, write_config):
        path = write_config(["not", "a", "mapping"])
        with pytest.raises(ValueError):
            load_settings(path)

    def test_polling_maximum_below_initial_is_rejected(self):
        with pytest.raises(ValueError):
            PollingSettings(initial_interval_s=30, maximum_interval_s=5)

    def test_timeouts_must_be_positive(self):
        with pytest.raises(ValueError):
            AgentSettings.model_validate({"mcp": {"spatial": {"call_timeout_s": 0}}})

    def test_conversation_limit_is_bounded(self):
        with pytest.raises(ValueError):
            AgentSettings.model_validate({"sessions": {"max_conversation_turns": 1}})

    def test_unknown_fields_are_ignored_rather_than_silently_used(self):
        settings = AgentSettings.model_validate({"mcp": {"spatial": {"nonsense": 1}}})
        assert settings.mcp.spatial.args == ["-m", "chongqing_spatial_mcp.server"]

    def test_source_is_not_taken_from_the_file(self, write_config):
        path = write_config({"source": "D:/somewhere/else"})
        assert load_settings(path).source == str(path)


class TestCLI:
    def test_help_succeeds_without_configuration(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr("slope_evaluation_agent.config.PROJECT_ROOT", tmp_path)
        with pytest.raises(SystemExit) as excinfo:
            cli_main(["--help"])
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "slope-evaluation-agent" in out
        assert SCOPE_STATEMENT in out
        assert DISCLAIMER_ZH in out

    def test_no_command_prints_help_and_returns_usage_error(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr("slope_evaluation_agent.config.PROJECT_ROOT", tmp_path)
        assert cli_main([]) == 2
        assert "usage" in capsys.readouterr().out

    def test_config_command_hides_the_key(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr("slope_evaluation_agent.config.PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("SLOPE_AGENT_LLM_BASE_URL", "https://example.invalid/v1")
        monkeypatch.setenv("SLOPE_AGENT_LLM_MODEL", "m")
        monkeypatch.setenv(ENV_LLM_API_KEY, "sk-do-not-print-me")
        assert cli_main(["config"]) == 0
        out = capsys.readouterr().out
        assert "sk-do-not-print-me" not in out
        assert "已设置" in out

    def test_config_json_output_parses(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr("slope_evaluation_agent.config.PROJECT_ROOT", tmp_path)
        assert cli_main(["config", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["language"] == "zh"
        assert payload["llm"]["available"] is False

    def test_broken_config_is_reported_without_a_traceback(self, tmp_path, capsys):
        path = tmp_path / "broken.yaml"
        path.write_text("language: [not, a, scalar]\n", encoding="utf-8")
        code = cli_main(["--config", str(path), "config"])
        assert code == 2
        captured = capsys.readouterr()
        assert "Traceback" not in captured.err
        assert "配置错误" in captured.err

    def test_module_entry_point_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "slope_evaluation_agent.cli", "--help"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "--version" in result.stdout

    def test_version_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "slope_evaluation_agent.cli", "--version"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert __version__ in result.stdout

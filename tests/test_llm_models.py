"""Offline tests for calle.llm_models (CALLE_LLM_MODEL presets)."""

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.groq.llm import GroqLLMService

from calle.config import ConfigError
from calle.llm_models import DEFAULT_LLM, LLM_ENV_VAR, LLM_PRESETS, resolve_llm


def test_default_is_gpt_oss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LLM_ENV_VAR, raising=False)
    assert resolve_llm() == LLM_PRESETS[DEFAULT_LLM]
    assert resolve_llm().model == "openai/gpt-oss-20b"


@pytest.mark.parametrize("value", ["qwen", "qwen/qwen3.8-27b", " qwen "])
def test_env_selects_qwen_by_key_or_model_id(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(LLM_ENV_VAR, value)
    preset = resolve_llm()
    assert preset.model == "qwen/qwen3.8-27b"
    assert preset.reasoning_effort == "none" and not preset.supports_developer_role


def test_unknown_model_is_a_clear_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LLM_ENV_VAR, "gpt-5")
    with pytest.raises(ConfigError, match="qwen") as error:
        resolve_llm()
    assert "gpt-oss" in str(error.value)


@pytest.mark.parametrize("key, expected_role", [("gpt-oss", "developer"), ("qwen", "user")])
def test_preset_controls_how_pipecat_sends_the_greeting(key: str, expected_role: str) -> None:
    # bot.py sets llm.supports_developer_role from the preset; Pipecat's
    # adapter then converts "developer" messages to "user" for Qwen.
    preset = LLM_PRESETS[key]
    llm = GroqLLMService(
        api_key="test",
        settings=GroqLLMService.Settings(model=preset.model, system_instruction="sys"),
    )
    llm.supports_developer_role = preset.supports_developer_role
    context = LLMContext()
    context.add_message({"role": "developer", "content": "Greet the caller."})
    params = llm.get_llm_adapter().get_llm_invocation_params(
        context,
        system_instruction="sys",
        convert_developer_to_user=not llm.supports_developer_role,
    )
    assert [m["role"] for m in params["messages"]] == ["system", expected_role]

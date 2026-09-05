"""Regression tests for the Rigel OpenCode session-affinity compatibility hotfix."""

from types import SimpleNamespace

from agent import auxiliary_client as aux
from agent import chat_completion_helpers as cch
from agent.opencode_affinity import (
    OPENCODE_SESSION_HEADER,
    is_opencode_target,
    merge_opencode_session_headers,
    opencode_session_headers,
)
from gateway import session_context


def test_opencode_provider_uses_stable_explicit_session_id():
    first = opencode_session_headers(
        "opencode-go", "https://opencode.ai/zen/go/v1", "sess-one"
    )
    second = opencode_session_headers(
        "opencode-go", "https://opencode.ai/zen/go/v1", "sess-one"
    )
    other = opencode_session_headers(
        "opencode-go", "https://opencode.ai/zen/go/v1", "sess-two"
    )

    assert first[OPENCODE_SESSION_HEADER] == "sess-one"
    assert second == first
    assert other[OPENCODE_SESSION_HEADER] == "sess-two"


def test_opencode_target_detection_supports_go_zen_and_url_only():
    assert is_opencode_target("opencode-go", None)
    assert is_opencode_target("opencode-zen", None)
    assert is_opencode_target("custom", "https://opencode.ai/zen/go/v1")
    assert not is_opencode_target("custom", "https://not-opencode.example/v1")
    assert not is_opencode_target("custom", "https://opencode.ai.evil.example/v1")


def test_non_opencode_request_is_untouched():
    kwargs = {"extra_headers": {"x-existing": "yes"}}
    result = merge_opencode_session_headers(
        kwargs, "openrouter", "https://openrouter.ai/api/v1", "sess-one"
    )
    assert result == {"extra_headers": {"x-existing": "yes"}}


def test_existing_caller_pinned_opencode_header_wins():
    kwargs = {"extra_headers": {OPENCODE_SESSION_HEADER: "caller-pinned"}}
    result = merge_opencode_session_headers(
        kwargs, "opencode-go", "https://opencode.ai/zen/go/v1", "sess-one"
    )
    assert result["extra_headers"][OPENCODE_SESSION_HEADER] == "caller-pinned"


def test_main_api_builder_adds_header_after_transport_builder(monkeypatch):
    monkeypatch.setattr(
        cch,
        "_build_api_kwargs_for_mode",
        lambda agent, messages: {
            "model": agent.model,
            "messages": messages,
            "extra_headers": {"x-existing": "yes"},
        },
    )
    agent = SimpleNamespace(
        provider="opencode-go",
        base_url="https://opencode.ai/zen/go/v1",
        session_id="sess-main",
        model="deepseek-v4-flash",
    )

    result = cch.build_api_kwargs(agent, [{"role": "user", "content": "hi"}])
    assert result["extra_headers"]["x-existing"] == "yes"
    assert result["extra_headers"][OPENCODE_SESSION_HEADER] == "sess-main"


def test_auxiliary_call_uses_ambient_session_id(monkeypatch):
    session_context.reset_session_vars()
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-aux")

    result = aux._build_call_kwargs(
        "opencode-go",
        "deepseek-v4-flash",
        [{"role": "user", "content": "hi"}],
        base_url="https://opencode.ai/zen/go/v1",
    )
    assert result["extra_headers"][OPENCODE_SESSION_HEADER] == "sess-aux"


def test_auxiliary_non_opencode_call_has_no_affinity_header(monkeypatch):
    session_context.reset_session_vars()
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-aux")

    result = aux._build_call_kwargs(
        "openrouter",
        "some-model",
        [{"role": "user", "content": "hi"}],
        base_url="https://openrouter.ai/api/v1",
    )
    assert OPENCODE_SESSION_HEADER not in (result.get("extra_headers") or {})

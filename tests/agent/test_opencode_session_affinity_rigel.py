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



def test_codex_aux_adapter_forwards_extra_headers_to_responses():
    from agent.auxiliary_client import _CodexCompletionsAdapter

    message_item = SimpleNamespace(
        type="message",
        role="assistant",
        status="completed",
        content=[SimpleNamespace(type="output_text", text="ok")],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.output_item.done", item=message_item),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                status="completed",
                id="resp_test",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
            ),
        ),
    ]

    class _Stream:
        def __iter__(self):
            return iter(events)

        def close(self):
            pass

    captured = {}

    def _create(**kwargs):
        captured.update(kwargs)
        return _Stream()

    client = SimpleNamespace(responses=SimpleNamespace(create=_create))
    adapter = _CodexCompletionsAdapter(client, "gpt-5.5")
    adapter.create(
        messages=[{"role": "user", "content": "hi"}],
        extra_headers={OPENCODE_SESSION_HEADER: "sess-codex"},
    )

    assert captured["extra_headers"][OPENCODE_SESSION_HEADER] == "sess-codex"


def test_anthropic_aux_adapter_forwards_extra_headers_to_messages(monkeypatch):
    from agent import anthropic_adapter
    from agent import transports
    from agent.auxiliary_client import _AnthropicCompletionsAdapter

    captured = {}

    def _create_message(_client, api_kwargs, **_kwargs):
        captured.update(api_kwargs)
        return SimpleNamespace(usage=None)

    class _Transport:
        @staticmethod
        def normalize_response(_response, **_kwargs):
            return SimpleNamespace(
                content="ok",
                tool_calls=None,
                reasoning=None,
                finish_reason="stop",
            )

    monkeypatch.setattr(anthropic_adapter, "create_anthropic_message", _create_message)
    monkeypatch.setattr(transports, "get_transport", lambda _mode: _Transport())

    adapter = _AnthropicCompletionsAdapter(SimpleNamespace(), "claude-sonnet-4-5")
    adapter.create(
        messages=[{"role": "user", "content": "hi"}],
        extra_headers={OPENCODE_SESSION_HEADER: "sess-anthropic"},
    )

    assert captured["extra_headers"][OPENCODE_SESSION_HEADER] == "sess-anthropic"

"""AnthropicAdapter and AnthropicWebSearchBackend through the installed SDK.

The other Anthropic tests use stand-in clients, so they cannot notice when an
``anthropic`` release changes what the package sends or reads. These tests
drive the real SDK over a mock HTTP transport, with no network and no key:
anthropic 2.x runs on ``httpx2`` and targets API version 2026-10-21, which
has no ``temperature``; older releases run on ``httpx`` and still take it.
"""

from __future__ import annotations

import inspect
import json

import pytest

anthropic = pytest.importorskip("anthropic")

from metacouplingllm.knowledge.websearch import AnthropicWebSearchBackend  # noqa: E402
from metacouplingllm.llm.client import AnthropicAdapter, Message  # noqa: E402

if int(str(anthropic.__version__).split(".")[0]) >= 2:
    httpx = pytest.importorskip("httpx2")
else:
    httpx = pytest.importorskip("httpx")

_MESSAGES = [Message(role="system", content="S"), Message(role="user", content="U")]
_WEB_RESULTS = [
    {"type": "web_search_result", "url": "https://example.org/a",
     "title": "Source A", "encrypted_content": "e", "page_age": "2026-01-01"},
    {"type": "web_search_result", "url": "https://example.org/b",
     "title": "Source B", "encrypted_content": "e", "page_age": None},
]


def _sse(model: str, blocks: list[dict], stop: str) -> bytes:
    """The server-sent events the Messages API streams for *blocks*."""
    events = [("message_start", {"type": "message_start", "message": {
        "id": "msg_01", "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 1}}})]
    for i, block in enumerate(blocks):
        if block["type"] == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        elif block["type"] in ("tool_use", "server_tool_use"):
            start = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        else:
            start, delta = block, None
        events.append(("content_block_start", {
            "type": "content_block_start", "index": i, "content_block": start}))
        if delta is not None:
            events.append(("content_block_delta", {
                "type": "content_block_delta", "index": i, "delta": delta}))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    events.append(("message_delta", {
        "type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": 34}}))
    events.append(("message_stop", {"type": "message_stop"}))
    return "".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events
    ).encode()


class _Server:
    """Answers Messages API requests as the API does, plain or streamed."""

    def __init__(self) -> None:
        self.submit_results = True
        self.requests: list[dict] = []

    def _blocks(self, request: dict) -> list[dict]:
        tools = request.get("tools") or []
        if any(str(t.get("type", "")).startswith("web_search") for t in tools):
            blocks = [
                {"type": "server_tool_use", "id": "srvtoolu_01",
                 "name": "web_search", "input": {"query": "q"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_01",
                 "content": _WEB_RESULTS},
                {"type": "text", "text": "Found two sources."},
            ]
            if self.submit_results:
                blocks.append({
                    "type": "tool_use", "id": "toolu_02", "name": "submit_results",
                    "input": {"results": [
                        {"title": r["title"], "url": r["url"],
                         "model_summary": f"About {r['title']}."}
                        for r in _WEB_RESULTS
                    ]},
                })
            return blocks
        if tools:
            return [{"type": "tool_use", "id": "toolu_01",
                     "name": tools[0]["name"], "input": {"x": "y"}}]
        return [{"type": "text", "text": "Hello"}]

    def __call__(self, http_request):
        request = json.loads(http_request.content)
        self.requests.append(request)
        blocks = self._blocks(request)
        stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        if request.get("stream"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"},
                content=_sse(request["model"], blocks, stop),
            )
        return httpx.Response(200, json={
            "id": "msg_01", "type": "message", "role": "assistant",
            "model": request["model"], "content": blocks, "stop_reason": stop,
            "stop_sequence": None, "usage": {"input_tokens": 12, "output_tokens": 34},
        })


@pytest.fixture
def server():
    return _Server()


@pytest.fixture
def client(server):
    return anthropic.Anthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
    )


def _sdk_takes_temperature(client) -> bool:
    return "temperature" in inspect.signature(client.messages.create).parameters


def test_create_sends_temperature_only_when_the_sdk_takes_it(client, server):
    result = AnthropicAdapter(client, model="claude-sonnet-4-6").chat(
        _MESSAGES, temperature=0.3,
    )
    assert result.content == "Hello"
    assert result.usage == {"input_tokens": 12, "output_tokens": 34}
    request = server.requests[-1]
    assert request["system"] == "S"
    assert request["messages"] == [{"role": "user", "content": "U"}]
    assert request["max_tokens"] == 4096
    assert not request.get("stream")
    if _sdk_takes_temperature(client):
        assert request["temperature"] == 0.3
    else:
        assert "temperature" not in request


def test_default_model_is_sent_without_temperature(client, server):
    AnthropicAdapter(client).chat(_MESSAGES, temperature=0.3)
    request = server.requests[-1]
    assert request["model"] == "claude-sonnet-5-5"
    assert "temperature" not in request


def test_tool_use_is_read(client):
    tool = {
        "name": "submit", "description": "d",
        "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
    }
    result = AnthropicAdapter(client, model="claude-sonnet-4-6").chat(
        _MESSAGES, tools=[tool], tool_choice={"type": "tool", "name": "submit"},
    )
    assert result.tool_uses == [{"id": "toolu_01", "name": "submit", "input": {"x": "y"}}]


def test_large_max_tokens_streams(client, server):
    result = AnthropicAdapter(client, model="claude-sonnet-4-6").chat(
        _MESSAGES, temperature=0.3, max_tokens=20000,
    )
    assert result.content == "Hello"
    request = server.requests[-1]
    assert request["stream"] is True
    assert request["max_tokens"] == 20000
    assert ("temperature" in request) == _sdk_takes_temperature(client)


def test_web_search_reads_submit_results(client, server):
    results = AnthropicWebSearchBackend(client=client, model="claude-sonnet-4-6").search(
        "q", max_results=2,
    )
    assert [r["url"] for r in results] == ["https://example.org/a", "https://example.org/b"]
    assert [r["title"] for r in results] == ["Source A", "Source B"]
    request = server.requests[-1]
    assert request["stream"] is True
    assert "temperature" not in request


def test_web_search_falls_back_to_the_search_results(client, server):
    server.submit_results = False
    results = AnthropicWebSearchBackend(client=client, model="claude-sonnet-4-6").search(
        "q", max_results=2,
    )
    assert {r["url"] for r in results} == {"https://example.org/a", "https://example.org/b"}

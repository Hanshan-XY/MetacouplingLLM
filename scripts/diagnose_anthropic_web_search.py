#!/usr/bin/env python3
"""Bisect the Anthropic web_search streaming failure.

The PR #28 live trace hit "peer closed connection without sending
complete message body (incomplete chunked read)" mid-stream when
AnthropicWebSearchBackend.search() called Claude with the full
PR #28 config (web_search_20260209 + code_execution + submit_results +
max_tokens=25000 + max_uses=5).

This script runs 5 progressively-complex probes against Anthropic
directly to isolate which combination of features causes the
streaming connection to drop.  Each probe is small (~$0.05) and
quick (~30-60s).

Usage:
    python scripts/diagnose_anthropic_web_search.py

Edit API_KEY_PATH if needed.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

API_KEY_PATH = Path(
    r"D:\Onedrive\OneDrive - Michigan State University\Desktop\Api_Anthropic.env"
)
MODEL = "claude-sonnet-4-6"
QUERY = "Mexico avocado exports destinations 2024"


def load_api_key() -> None:
    for raw in API_KEY_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ[key.strip()] = value.strip()


def run_probe(label: str, client: Any, **request_kwargs: Any) -> None:
    """Run a single web_search probe and report success/failure."""
    print(f"\n{'=' * 70}")
    print(f"PROBE: {label}")
    print("=" * 70)
    print(f"  tools: {[t.get('name') or t.get('type') for t in request_kwargs.get('tools', [])]}")
    print(f"  tool_choice: {request_kwargs.get('tool_choice')}")
    print(f"  max_tokens: {request_kwargs.get('max_tokens')}")
    t0 = time.perf_counter()
    try:
        with client.messages.stream(**request_kwargs) as stream:
            response = stream.get_final_message()
        duration = time.perf_counter() - t0
        n_blocks = len(response.content)
        block_types = [getattr(b, "type", "?") for b in response.content]
        tool_use_names = [
            getattr(b, "name", "?")
            for b in response.content
            if getattr(b, "type", None) == "tool_use"
        ]
        print(f"  [OK] SUCCESS in {duration:.1f}s")
        print(f"    blocks: {n_blocks} ({block_types})")
        print(f"    tool_use names: {tool_use_names}")
        usage = response.usage
        print(f"    usage: input={usage.input_tokens}, output={usage.output_tokens}")
    except Exception as exc:
        duration = time.perf_counter() - t0
        print(f"  [FAIL] FAILED in {duration:.1f}s")
        print(f"    {type(exc).__name__}: {exc}")
        # Print a brief traceback for context.
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        print("    " + "".join(tb_lines[-3:]).rstrip().replace("\n", "\n    "))


def main() -> int:
    load_api_key()
    import anthropic
    client = anthropic.Anthropic()

    # Shared user message body
    user_msg = {"role": "user", "content": f"Search the web for: {QUERY}"}

    # Submit-results user-defined tool (matches PR #28 shape)
    submit_tool = {
        "name": "submit_results",
        "description": (
            "Submit the final list of web search results.  Call this "
            "tool AFTER you have completed all web_search invocations."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "model_summary": {"type": "string"},
                        },
                        "required": ["title", "url", "model_summary"],
                    },
                },
            },
            "required": ["results"],
        },
        "strict": True,
    }

    # ------------------------------------------------------------------
    # Probe 1: minimal -- basic web_search, no submit_results, no
    # code_execution, low max_tokens, max_uses=2.  Establishes a
    # baseline that web_search itself works.
    # ------------------------------------------------------------------
    run_probe(
        "1. basic web_search_20250305, no extras",
        client,
        model=MODEL,
        max_tokens=4000,
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 2,
        }],
        tool_choice={"type": "tool", "name": "web_search"},
        messages=[user_msg],
    )

    # ------------------------------------------------------------------
    # Probe 2: basic web_search + submit_results.  Isolates whether the
    # user-defined submit_results tool destabilises the stream.
    # ------------------------------------------------------------------
    run_probe(
        "2. basic web_search_20250305 + submit_results tool",
        client,
        model=MODEL,
        max_tokens=4000,
        tools=[
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 2,
            },
            submit_tool,
        ],
        tool_choice={"type": "tool", "name": "web_search"},
        messages=[user_msg],
    )

    # ------------------------------------------------------------------
    # Probe 3: dynamic-filtering web_search + code_execution, no
    # submit_results.  Isolates the code_execution interaction.
    # ------------------------------------------------------------------
    run_probe(
        "3. web_search_20260209 + code_execution (no submit_results)",
        client,
        model=MODEL,
        max_tokens=4000,
        tools=[
            {
                "type": "web_search_20260209",
                "name": "web_search",
                "max_uses": 2,
            },
            {
                "type": "code_execution_20260120",
                "name": "code_execution",
            },
        ],
        tool_choice={"type": "tool", "name": "code_execution"},
        messages=[user_msg],
    )

    # ------------------------------------------------------------------
    # Probe 4: full PR #28 combo + tight max_tokens.  Tests whether
    # the issue is the three-tool stack itself, independent of token
    # budget.
    # ------------------------------------------------------------------
    run_probe(
        "4. full PR #28 stack with low max_tokens=4000",
        client,
        model=MODEL,
        max_tokens=4000,
        tools=[
            {
                "type": "web_search_20260209",
                "name": "web_search",
                "max_uses": 2,
            },
            submit_tool,
            {
                "type": "code_execution_20260120",
                "name": "code_execution",
            },
        ],
        tool_choice={"type": "tool", "name": "code_execution"},
        messages=[user_msg],
    )

    # ------------------------------------------------------------------
    # Probe 5: replicate the live-trace failure exactly --
    # max_tokens=25000, max_uses=5, full stack.
    # ------------------------------------------------------------------
    run_probe(
        "5. live-trace replica: max_tokens=25000, max_uses=5",
        client,
        model=MODEL,
        max_tokens=25000,
        tools=[
            {
                "type": "web_search_20260209",
                "name": "web_search",
                "max_uses": 5,
            },
            submit_tool,
            {
                "type": "code_execution_20260120",
                "name": "code_execution",
            },
        ],
        tool_choice={"type": "tool", "name": "code_execution"},
        messages=[user_msg],
    )

    print(f"\n{'=' * 70}")
    print("Bisect complete.  The first FAILING probe identifies the issue.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

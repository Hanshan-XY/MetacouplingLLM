"""
Citation sanitization for the turn-scoped RAG pipeline.

In multi-turn conversations, each user message may include a
``<retrieved_literature turn="k">`` block whose passages are labeled
with ``id="N"``. The LLM is instructed (via the CITATION_RULES_LAYER
prompt) to cite them inline as ``[Tk:N]`` for literature and
``[Tk:Wn]`` for web sources. The ``Tk:`` prefix is the **turn index**
so that once a citation is emitted it remains unambiguous forever —
even when conversation history is read across many turns.

This module provides:

- :data:`TURN_CITATION_PATTERN` — the canonical regex matching
  ``[Tk:N]`` (literature) and ``[Tk:Wn]`` (web).
- :func:`sanitize_turn_citations` — drops any token whose ``Tk:N``
  pair is invalid given the recorded passage / web counts per turn.
  Bare ``[N]`` / ``[W1]`` tokens (the pre-v0.1.3 grammar, which the
  LLM should no longer emit) are silently stripped as a defensive
  measure against LLM slips — they are NOT supported as input.
- :func:`extract_turn_cited_ids` — returns the set of citation tuples
  ``(k, kind, N)`` used in a block of text, where ``kind`` is ``""``
  for literature and ``"W"`` for web.

A short, well-cited turn is better than a long, weakly-supported one.

.. note::

   The pre-v0.1.3 public API (``sanitize_citations``,
   ``extract_cited_ids``, ``CITATION_PATTERN``) was removed in v0.1.3.
   Callers should migrate to :func:`sanitize_turn_citations` and
   :func:`extract_turn_cited_ids`.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Canonical turn-scoped citation pattern. Matches:
#   [T1:1], [T2:42]        → group(1)="1"/"2", group(2)="",  group(3)="1"/"42"
#   [T1:W1], [T3:W2]       → group(1)="1"/"3", group(2)="W", group(3)="1"/"2"
# Does NOT match bare [1] or [W1] — those are silently stripped inside
# the sanitizer as a defensive measure but are not a supported input
# form.
TURN_CITATION_PATTERN = re.compile(r"\[T(\d+):(W?)(\d+)\]")


# Internal regexes used by the defensive bare-token strip in
# sanitize_turn_citations. Kept module-private — not exported.
_BARE_LITERATURE_TOKEN_RE = re.compile(r"\[(\d+)\]")
_BARE_WEB_TOKEN_RE = re.compile(r"\[W(\d+)\]")


def _cleanup_whitespace(text: str) -> str:
    """Idempotent cleanup of leftover whitespace and punctuation spacing.

    Run after citation stripping so that an input like
    ``"claim [T1:99] and more"`` does not leave a double space, and
    ``"claim [T1:99]."`` does not leave an orphan space before the
    period.

    Only horizontal whitespace and punctuation-adjacent spaces are
    touched — newlines are preserved exactly so list/paragraph
    structure survives.
    """
    # Collapse runs of horizontal whitespace (tabs + spaces); keep newlines
    text = re.sub(r"[ \t]+", " ", text)
    # Remove any space immediately before sentence punctuation
    text = re.sub(r" +([.,;:!?)])", r"\1", text)
    # Strip trailing horizontal whitespace on every line
    text = re.sub(r" +$", "", text, flags=re.MULTILINE)
    return text


def sanitize_turn_citations(
    text: str,
    turn_passage_counts: dict[int, int],
    turn_web_counts: dict[int, int],
    current_turn: int,
) -> tuple[str, set[tuple[int, str, int]]]:
    """Strip turn-scoped citations whose ``(k, N)`` is out of range.

    Parameters
    ----------
    text:
        The LLM-generated text to sanitize.
    turn_passage_counts:
        Mapping ``k -> number of literature passages shown in turn k``.
        Used to validate ``[Tk:N]`` tokens.
    turn_web_counts:
        Mapping ``k -> number of web sources shown in turn k``. Used
        to validate ``[Tk:Wn]`` tokens. Empty dict if web search was
        never enabled.
    current_turn:
        The 1-indexed turn that is currently being emitted. Tokens
        with ``k > current_turn`` are forward references (impossible)
        and get stripped. ``k = 0`` is also invalid (turns are
        1-indexed).

    Returns
    -------
    A tuple ``(sanitized_text, invalid_tokens)``:

    - ``sanitized_text`` has invalid tokens removed and whitespace /
      punctuation normalized.
    - ``invalid_tokens`` is a set of ``(k, kind, N)`` tuples that were
      stripped, where ``kind`` is ``""`` (literature) or ``"W"`` (web).

    A WARNING-level log message naming the stripped tokens is emitted
    when at least one invalid token is found. Bare ``[N]`` and ``[W1]``
    tokens (legacy pre-v0.1.3 grammar) are stripped silently as a
    defensive measure — they are not a supported input form.
    """
    invalid: set[tuple[int, str, int]] = set()

    def _replacer(match: re.Match[str]) -> str:
        k = int(match.group(1))
        kind = match.group(2)  # "" for literature, "W" for web
        n = int(match.group(3))
        if k < 1 or k > current_turn:
            invalid.add((k, kind, n))
            return ""
        if kind == "W":
            valid_n = turn_web_counts.get(k, 0)
        else:
            valid_n = turn_passage_counts.get(k, 0)
        if not (1 <= n <= valid_n):
            invalid.add((k, kind, n))
            return ""
        return match.group(0)

    sanitized = TURN_CITATION_PATTERN.sub(_replacer, text)

    # Defensive: strip any bare-numeric / bare-web tokens the LLM may
    # have slipped despite the prompt rules. These are not a supported
    # input form; this is just garbage cleanup so they don't pollute
    # the user's output. Silent — no warning per occurrence.
    sanitized = _BARE_LITERATURE_TOKEN_RE.sub("", sanitized)
    sanitized = _BARE_WEB_TOKEN_RE.sub("", sanitized)

    if invalid:
        formatted = sorted(
            f"[T{k}:{kind}{n}]" for k, kind, n in invalid
        )
        logger.warning(
            "Stripped %d invalid turn-scoped citation(s): %s "
            "(current turn: %d; turn counts: lit=%s web=%s)",
            len(invalid),
            formatted,
            current_turn,
            dict(sorted(turn_passage_counts.items())),
            dict(sorted(turn_web_counts.items())),
        )

    sanitized = _cleanup_whitespace(sanitized)
    return sanitized, invalid


def extract_turn_cited_ids(text: str) -> set[tuple[int, str, int]]:
    """Return the set of turn-scoped citation tuples used in ``text``.

    Each tuple is ``(k, kind, N)`` where ``kind`` is ``""`` for
    literature citations or ``"W"`` for web citations.

    Useful for downstream validation and tests. Bare ``[N]`` / ``[W1]``
    tokens (legacy pre-v0.1.3 grammar) are NOT recognised.
    """
    return {
        (int(m.group(1)), m.group(2), int(m.group(3)))
        for m in TURN_CITATION_PATTERN.finditer(text)
    }

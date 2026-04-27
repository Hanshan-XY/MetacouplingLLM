"""
Citation sanitization utilities for the pre-retrieval RAG pipeline.

When the LLM is given a `<retrieved_literature>` block with N passages
labeled `[1]..[N]` and asked to cite them inline, it may occasionally
emit out-of-range tokens like `[99]` (a hallucinated citation) or copy a
stale `[1]` from a previous turn that no longer corresponds to the same
paper. This module provides:

- :func:`sanitize_citations` strips bracket-number tokens whose ID falls
  outside ``[1, n_valid]``, logs a warning naming the stripped IDs, and
  runs an idempotent whitespace/punctuation cleanup pass so the result
  reads naturally.
- :func:`extract_cited_ids` returns the set of citation numbers used in a
  block of text — handy for tests and downstream validation.

Both functions only touch numeric bracket tokens like ``[1]`` or ``[42]``;
they leave alphanumeric tokens like ``[W1]`` (web search citations) alone.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Match [N] where N is one or more digits.
# Does NOT match [W1] (web citations) or other alphanumeric tokens.
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _cleanup_whitespace(text: str) -> str:
    """Idempotent cleanup of leftover whitespace and punctuation spacing.

    Run after citation stripping so that an input like
    ``"claim [99] and more"`` does not leave a double space, and
    ``"claim [99]."`` does not leave an orphan space before the period.

    Only horizontal whitespace and punctuation-adjacent spaces are
    touched — newlines are preserved exactly so list/paragraph structure
    survives.
    """
    # Collapse runs of horizontal whitespace (tabs + spaces); keep newlines
    text = re.sub(r"[ \t]+", " ", text)
    # Remove any space immediately before sentence punctuation
    text = re.sub(r" +([.,;:!?)])", r"\1", text)
    # Strip trailing horizontal whitespace on every line
    text = re.sub(r" +$", "", text, flags=re.MULTILINE)
    return text


def sanitize_citations(
    text: str, n_valid: int
) -> tuple[str, set[int]]:
    """Strip numeric citation tokens whose ID is outside ``[1, n_valid]``.

    Parameters
    ----------
    text:
        The LLM-generated text to sanitize. Typically the raw response
        ``content`` before any structured parsing.
    n_valid:
        The number of valid citation labels in the most recent
        ``<retrieved_literature>`` block. Tokens ``[1]..[n_valid]`` are
        kept; everything else (e.g., ``[0]``, ``[99]``) is stripped.

    Returns
    -------
    A tuple ``(sanitized_text, invalid_ids)``:

    - ``sanitized_text`` is the cleaned text with invalid tokens removed
      and whitespace/punctuation normalized.
    - ``invalid_ids`` is the set of out-of-range integers that were
      found and stripped (empty if input was already clean).

    A WARNING-level log message naming the stripped IDs is emitted via
    the module logger when at least one invalid token is found.
    """
    invalid_ids: set[int] = set()

    def _replacer(match: re.Match[str]) -> str:
        cid = int(match.group(1))
        if 1 <= cid <= n_valid:
            return match.group(0)
        invalid_ids.add(cid)
        return ""

    sanitized = CITATION_PATTERN.sub(_replacer, text)
    if invalid_ids:
        logger.warning(
            "Stripped %d invalid citation ID(s): %s (valid range: 1..%d)",
            len(invalid_ids),
            sorted(invalid_ids),
            n_valid,
        )
    # Always run cleanup — it is idempotent on already-clean input and
    # only matters when something was actually stripped.
    sanitized = _cleanup_whitespace(sanitized)
    return sanitized, invalid_ids


def extract_cited_ids(text: str) -> set[int]:
    """Return the set of numeric citation IDs used in ``text``.

    Useful for downstream validation and tests. Web citations like
    ``[W1]`` are NOT included — only purely numeric brackets.
    """
    return {int(m) for m in CITATION_PATTERN.findall(text)}

"""Regression tests for the fast ADM1 name matching.

The substring strategies of ``resolve_adm1_code`` and the country-name scan of
``resolve_country_code`` test hundreds to thousands of names per query.  They
check each name with ``str.find`` and explicit boundary tests, and the
substring strategies test only the names that share letter runs with the
query, instead of compiling a regular expression per name.  These tests pin
the results to outputs recorded from the regular-expression implementation
(main at cf919fc), held in ``data/adm1_resolve_golden.json``:

* ``resolve_adm1_code`` for every (query, country hint) that
  ``MetacouplingAssistant._extract_mentioned_adm1_from_text`` generates for
  the committed avocado run's main analysis when no candidate resolves (a
  superset of the calls a real run makes), for every input the test suite
  passed the resolver, and for hand-picked edge cases;
* the extractor's ADM1 code set for that analysis.

``data/avocado_main_analysis_response.md`` is a verbatim copy of the
``## Response`` section of
``runs/avocado_2026-06-16_gpt-5.5_builtin-trace/turn1/05_llm_call_main_analysis.md``,
copied so that re-running the trace harness, which writes to that folder,
cannot change this test's input.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
import time
from pathlib import Path

from metacouplingllm.core import MetacouplingAssistant
from metacouplingllm.knowledge import adm1_pericoupling as adm1
from metacouplingllm.knowledge.countries import _contains_standalone_country_term
from metacouplingllm.llm.parser import parse_analysis

_DATA = Path(__file__).resolve().parent / "data"
_GOLDEN = json.loads(
    (_DATA / "adm1_resolve_golden.json").read_text(encoding="utf-8")
)


def _regex_phrase(haystack: str, needle: str) -> bool:
    """The regular expression ``_contains_phrase`` stands for."""
    pattern = rf"(?<![a-z]){re.escape(needle)}(?![a-z])"
    return re.search(pattern, haystack) is not None


def _regex_country_term(text: str, term: str) -> bool:
    """The regular expression ``_contains_standalone_country_term`` stands for."""
    pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _string_pairs() -> list[tuple[str, str]]:
    """Every haystack of up to 4 and needle of up to 2 characters over an
    alphabet with letters, a character just past ``z``, a digit, an accented
    letter and a space; then seeded random pairs over wider alphabets, half
    with the needle cut from the haystack so that most of them occur."""
    alphabet = "az{0é "
    pairs = [
        ("".join(h), "".join(n))
        for h_len in range(5)
        for h in itertools.product(alphabet, repeat=h_len)
        for n_len in range(3)
        for n in itertools.product(alphabet, repeat=n_len)
    ]
    rng = random.Random(20261007)
    wide = ["ab -", "az`{09/:", "aé\u0301ñ'-b", "Aa İı1 9", "東京a b"]
    for _ in range(3000):
        chars = rng.choice(wide)
        haystack = "".join(rng.choice(chars) for _ in range(rng.randint(0, 14)))
        if haystack and rng.random() < 0.5:
            i = rng.randint(0, len(haystack))
            needle = haystack[i:rng.randint(i, len(haystack))]
        else:
            needle = "".join(rng.choice(chars) for _ in range(rng.randint(0, 4)))
        pairs.append((haystack, needle))
    return pairs


class TestRecordedOutputs:
    """Results identical to those of the regular-expression implementation."""

    def test_resolve_adm1_code_unchanged(self):
        rows = _GOLDEN["resolve_adm1_code"]
        changed = [
            (query, country, expected, got)
            for query, country, expected in rows
            if (got := adm1.resolve_adm1_code(query, country=country))
            != expected
        ]
        assert not changed, (
            f"{len(changed)} of {len(rows)} results changed: {changed[:10]}"
        )

    def test_extractor_codes_unchanged_on_avocado_analysis(self):
        response = (_DATA / "avocado_main_analysis_response.md").read_text(
            encoding="utf-8"
        )
        digest = hashlib.sha256(response.encode("utf-8")).hexdigest()
        assert digest == _GOLDEN["avocado_response_sha256"]
        codes = MetacouplingAssistant._extract_mentioned_adm1_from_text(
            parse_analysis(response)
        )
        assert sorted(codes) == _GOLDEN["avocado_extractor_codes"]


class TestMatcherExactness:
    """The ``str.find`` matchers and the letter-run pre-filter change no
    individual decision."""

    def test_contains_phrase_matches_its_regex(self):
        for haystack, needle in _string_pairs():
            assert adm1._contains_phrase(haystack, needle) == _regex_phrase(
                haystack, needle
            ), (haystack, needle)

    def test_country_term_check_matches_its_regex(self):
        for text, term in _string_pairs():
            assert _contains_standalone_country_term(
                text, term
            ) == _regex_country_term(text, term), (text, term)

    def test_substring_candidates_hold_every_match(self):
        """For each query, the keys ``_substring_match`` accepts among the
        candidates are exactly those it accepts in a scan of the whole
        index, for the accented and the folded index alike."""
        index = adm1._get_adm1_name_index()
        folded_index = adm1._get_adm1_folded_name_index()
        queries = [row[0] for row in _GOLDEN["resolve_adm1_code"][::40]]
        for key in sorted(index)[::25]:
            queries += [key, f"state of {key}", f"{key} province"]
        for query in queries:
            text = query.strip().lower()
            for q, idx, folded in (
                (text, index, False),
                (adm1._fold_diacritics(text), folded_index, True),
            ):
                full = {
                    k for k in idx
                    if len(k) >= 4 and adm1._substring_match(q, k)
                }
                fast = {
                    k for k, _ in adm1._substring_candidates(q, folded)
                    if len(k) >= 4 and adm1._substring_match(q, k)
                }
                assert fast == full, (query, folded, full ^ fast)


def test_unresolvable_queries_stay_fast():
    """A query no strategy resolves runs both substring strategies.  With a
    regular expression compiled per name, these 100 calls took about 50 s;
    they now take about 10 ms."""
    adm1.resolve_adm1_code("warm up the indexes")
    start = time.perf_counter()
    for i in range(100):
        assert adm1.resolve_adm1_code(f"avocado orchard cluster {i}") is None
    assert time.perf_counter() - start < 5.0

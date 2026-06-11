"""Doc-example execution CI guard (follow-up to PR #46 / PR #55).

PR #55 manually fixed ~7 fenced ``python`` examples in MANUAL.md (§7
structured-data access, §16 worked example, §17 end-to-end workflow)
that had drifted from the real API because nothing ever executed them.
This module turns that one-off accuracy pass into a build break, in
the spirit of ``tests/test_docs_capabilities.py``:

1. ``test_python_block_compiles`` — every fenced ```python block in
   MANUAL.md must at least be valid Python syntax (IPython ``!pip`` /
   ``%pip`` magic lines are masked first).  This covers blocks that
   can never run in CI — e.g. the §17 end-to-end workflow, which
   constructs ``OpenAI(api_key=...)`` — at the syntax level.

2. ``test_doc_section_executes`` — an allowlist of ``###`` section
   headings whose blocks are fully deterministic (no API keys, no
   network, no LLM calls) is *executed*, with assertions that the
   documented behaviour and outputs still match the real package:

   - §7  "Accessing structured data programmatically", run against a
     populated ``ParsedAnalysis`` built with
     ``tests/_helpers.make_parsed_analysis``.
   - §7  "Formatting options" — every ``format_component`` name in the
     doc must be recognised (no ``(Unknown component`` in output).
   - §8  standalone pericoupling lookups, adjacency standards, and
     country-name resolution — documented results must match the DB.
   - §16 Brazil-soybean worked example including the
     ``group_cols=["flow_type"]`` continuation — MFE ≈ 0.73,
     IFCI == 1.0, and the other documented indicator values.

Blocks NOT on the allowlist are skipped from execution by design —
anything needing an API key, the network, or an LLM must never run
here.  ``NEEDS_LLM_OR_NETWORK`` double-checks that allowlisted blocks
stay LLM-free, so a future doc edit that adds an ``advisor.analyze``
call to an executed section fails loudly instead of hitting the wire.

Adding a new runnable example to MANUAL.md: add an ``ExecSpec`` to
``EXECUTABLE_SECTIONS`` below, keyed by the new ``###`` heading text.
Renaming an allowlisted heading breaks its ``ExecSpec`` loudly.
"""
from __future__ import annotations

import io
import re
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from ._helpers import make_parsed_analysis

REPO_ROOT = Path(__file__).parent.parent
MANUAL_MD = REPO_ROOT / "MANUAL.md"


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DocBlock:
    """One fenced ```python block from MANUAL.md."""

    section: str        # nearest preceding "## " heading
    subsection: str     # nearest preceding "### " heading ("" if none)
    start_line: int     # line number of the opening fence
    code: str

    @property
    def location(self) -> str:
        heading = self.subsection or self.section
        return f"MANUAL.md:{self.start_line} ({heading})"


def _parse_manual_python_blocks() -> list[DocBlock]:
    """Extract every ```python fenced block, tagged with its headings."""
    blocks: list[DocBlock] = []
    section = subsection = ""
    in_block = False
    lang = ""
    start = 0
    buf: list[str] = []
    text = MANUAL_MD.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        if in_block:
            if line.strip() == "```":
                in_block = False
                if lang == "python":
                    blocks.append(
                        DocBlock(section, subsection, start, "\n".join(buf) + "\n")
                    )
            else:
                buf.append(line)
        elif line.startswith("### "):
            subsection = line[4:].strip()
        elif line.startswith("## "):
            section = line[3:].strip()
            subsection = ""
        elif line.startswith("```"):
            in_block = True
            lang = line[3:].strip()
            start = lineno
            buf = []
    return blocks


PYTHON_BLOCKS = _parse_manual_python_blocks()


def _block_id(block: DocBlock) -> str:
    slug = re.sub(r"\W+", "-", (block.subsection or block.section).lower())
    return f"L{block.start_line}-{slug.strip('-')[:48]}"


def test_extractor_finds_python_blocks():
    """Guard the extractor itself — if the fence regex rots, the
    parametrized tests below would silently collect nothing."""
    assert len(PYTHON_BLOCKS) >= 40, (
        f"Only {len(PYTHON_BLOCKS)} ```python blocks extracted from "
        "MANUAL.md (expected ~48). Either large doc sections were "
        "deleted or _parse_manual_python_blocks() no longer matches "
        "the fence style."
    )


# ---------------------------------------------------------------------------
# Tier 1 — every python block must be valid Python syntax
# ---------------------------------------------------------------------------

_IPYTHON_MAGIC = re.compile(r"^\s*[!%]")


def _mask_ipython_magics(code: str) -> str:
    """Comment out ``!pip`` / ``%pip`` notebook-magic lines so the
    §1 / §13 install snippets (valid in Jupyter, not in CPython)
    still pass the compile check without special-casing whole blocks."""
    return "\n".join(
        "# " + line if _IPYTHON_MAGIC.match(line) else line
        for line in code.splitlines()
    )


@pytest.mark.parametrize("block", PYTHON_BLOCKS, ids=_block_id)
def test_python_block_compiles(block: DocBlock):
    """Every ```python block in MANUAL.md must be syntactically valid —
    including the LLM-dependent ones (§2 Quick Start, §17 end-to-end
    workflow) that the execution tier below can never run."""
    code = _mask_ipython_magics(block.code)
    try:
        compile(code, f"{MANUAL_MD}:{block.start_line}", "exec")
    except SyntaxError as exc:
        pytest.fail(
            f"Doc example at {block.location} is not valid Python "
            f"(line {exc.lineno} of the block): {exc.msg}"
        )


# ---------------------------------------------------------------------------
# Tier 2 — deterministic sections are executed and their documented
# behaviour asserted
# ---------------------------------------------------------------------------

# Constructs that require an API key, the network, or an LLM round-trip.
# Allowlisted blocks must never contain these — this is the safety net
# that keeps the execution tier deterministic if a doc section is later
# edited to add an LLM call.
NEEDS_LLM_OR_NETWORK = re.compile(
    r"OpenAI\(|api_key|advisor\s*=|advisor\.|\.analyze\(|llm_client|"
    r"read_csv|RAGEngine|search_web|recommend_papers|"
    r"plot_focal_country_map|plot_analysis_map|plot_focal_adm1_map|"
    r"define_study\(|check_inputs\(|interpret_results\(|write_methods\(|"
    r"classify_ambiguous_edges\(|to_docx|to_markdown"
)


@dataclass(frozen=True)
class ExecSpec:
    """One deterministic MANUAL.md region to execute.

    All ```python blocks under ``subsections`` (in document order)
    run sequentially in a single shared namespace — continuation
    blocks like §16's ``group_cols=["flow_type"]`` example reuse
    variables from the preceding block, exactly as a reader would.
    """

    id: str
    subsections: tuple[str, ...]
    setup: Callable[[], dict[str, Any]]
    check: Callable[[dict[str, Any], str], None]
    requires: tuple[str, ...] = ()


# --- §7 Accessing structured data programmatically -------------------------

def _setup_structured_data() -> dict[str, Any]:
    """The §7 snippet starts from ``result.parsed`` — stand in for a
    real ``AnalysisResult`` with a populated ``ParsedAnalysis`` whose
    content mirrors the §7 formatted-report example (Brazil→China
    soybean telecoupling)."""
    parsed = make_parsed_analysis(
        coupling_classification=(
            "Telecoupling (primary) — the study involves interactions "
            "between Brazil (sending) and China (receiving) through "
            "soybean trade."
        ),
        systems={
            "sending": {
                "name": "Brazil soybean production regions",
                "geographic_scope": "Mato Grosso, Goiás, Bahia",
            },
            "receiving": {"name": "China consumer markets"},
        },
        flows=[
            {
                "category": "Matter",
                "direction": "Brazil → China",
                "description": "Soybeans exported",
            },
            {
                "category": "Capital",
                "direction": "China → Brazil",
                "description": "Payment for soybean imports",
            },
        ],
        agents=[
            {
                "level": "Governments / Policymakers",
                "name": "Chinese Ministry of Commerce",
            },
        ],
        causes={"Economic": ["Growing demand for animal feed in China"]},
        effects={
            "Environmental": ["Deforestation of Amazon and Cerrado biomes"],
        },
        suggestions=["Quantify carbon footprint of transportation flows"],
        country_pericoupling_info={
            "pair_results": "Brazil (BRA) ↔ China (CHN): TELECOUPLED",
            "note": (
                "LLM classification is consistent with the coupling "
                "database."
            ),
        },
        pericoupling_info={
            "pair_results": "Mato Grosso (BRA014) — subnational pairs",
            "note": "Subnational validation note.",
        },
    )
    return {"result": SimpleNamespace(parsed=parsed)}


def _check_structured_data(ns: dict[str, Any], out: str) -> None:
    # get_system_detail with a sub_field returns the raw field value.
    assert ns["sending"] == "Brazil soybean production regions"
    assert ns["scope"] == "Mato Grosso, Goiás, Bahia"
    # Each iter_* loop printed the fixture content it walked.
    assert "Matter: Brazil → China — Soybeans exported" in out
    assert "Capital: China → Brazil — Payment for soybean imports" in out
    assert "[Governments / Policymakers] Chinese Ministry of Commerce" in out
    assert (
        "[telecoupling] Economic: Growing demand for animal feed in China"
        in out
    )
    assert (
        "[telecoupling] Environmental: Deforestation of Amazon and "
        "Cerrado biomes" in out
    )
    assert "- Quantify carbon footprint of transportation flows" in out
    # Both validation fields were truthy, so both branches printed.
    assert "Brazil (BRA) ↔ China (CHN): TELECOUPLED" in out
    assert "LLM classification is consistent" in out
    assert "Mato Grosso (BRA014)" in out
    assert "Subnational validation note." in out


# --- §7 Formatting options --------------------------------------------------

def _setup_formatting_options() -> dict[str, Any]:
    """The snippet references ``parsed`` plus ``parsed1``/``2``/``3``
    for the comparison call.  Give each one enough content that the
    component renders are non-degenerate."""
    parsed = _setup_structured_data()["result"].parsed
    parsed1 = make_parsed_analysis(
        coupling_classification="Telecoupling (primary)",
        systems={"sending": "Brazil", "receiving": "China"},
        flows=[{"category": "Matter", "direction": "Brazil → China",
                "description": "Soybeans"}],
        suggestions=["Quantify spillover effects"],
    )
    parsed2 = make_parsed_analysis(
        coupling_classification="Pericoupling (primary)",
        systems={"sending": "Mexico", "receiving": "United States"},
        flows=[{"category": "People", "direction": "Mexico → USA",
                "description": "Seasonal labor migration"}],
        coupling_type="pericoupling",
    )
    parsed3 = make_parsed_analysis(
        coupling_classification="Intracoupling (primary)",
        systems={"sending": "China urban", "receiving": "China rural"},
        flows=[{"category": "Capital", "direction": "Urban → rural",
                "description": "Domestic remittances"}],
        coupling_type="intracoupling",
    )
    return {
        "parsed": parsed,
        "parsed1": parsed1,
        "parsed2": parsed2,
        "parsed3": parsed3,
    }


def _check_formatting_options(ns: dict[str, Any], out: str) -> None:
    # The headline guarantee: every component name used in the doc is
    # recognised by AnalysisFormatter.format_component.
    assert "(Unknown component" not in out, (
        "MANUAL.md §7 'Formatting options' passes a component name "
        "that AnalysisFormatter.format_component no longer recognises:"
        f"\n{out}"
    )
    # And none of the documented calls hit an empty-data fallback —
    # the fixture populates every component the snippet formats.
    assert "(No classification data)" not in out
    assert "(No telecoupling data)" not in out
    assert "(No research gap data)" not in out
    # format_comparison with three analyses renders the side-by-side view.
    assert "COMPARATIVE ANALYSIS" in out


# --- §8 Pericoupling database (offline, bundled DB) -------------------------

def _check_pericoupling_standalone(ns: dict[str, Any], out: str) -> None:
    # `result` is the lookup_pericoupling("Mexico", "United States")
    # object; the doc comments document each attribute.
    result = ns["result"]
    assert str(result.pair_type).endswith("PERICOUPLED")
    assert result.sending_code == "MEX"
    assert result.receiving_code == "USA"
    assert result.confidence == "database"

    is_pericoupled = ns["is_pericoupled"]
    assert is_pericoupled("Mexico", "USA") is True
    assert is_pericoupled("Mexico", "Canada") is False
    assert is_pericoupled("Brazil", "China") is False
    assert is_pericoupled("Atlantis", "USA") is None

    # Last assignment in the block: Mexico's pericoupled neighbors.
    assert ns["neighbors"] == {"USA", "GTM", "BLZ"}

    # "Adjacency standards" continuation block — documented values for
    # the coupling_standard toggle (PR #53).
    assert is_pericoupled("Romania", "Moldova") is True
    assert (
        is_pericoupled("Romania", "Moldova", coupling_standard="stringent")
        is False
    )
    assert is_pericoupled("COD", "CAF") is False
    assert is_pericoupled("COD", "CAF", coupling_standard="lenient") is True


def _check_country_resolution(ns: dict[str, Any], out: str) -> None:
    resolve = ns["resolve_country_code"]
    documented = {
        "USA": "USA",
        "usa": "USA",
        "Mexico": "MEX",
        "United States of America": "USA",
        "UK": "GBR",
        "South Korea": "KOR",
        "Russia": "RUS",
        "Mexican": "MEX",
        "Brazilian": "BRA",
        "Chinese": "CHN",
        "Ethiopian coffee regions": "ETH",
    }
    for query, expected in documented.items():
        assert resolve(query) == expected, (
            f"MANUAL.md §8 documents resolve_country_code({query!r}) "
            f"== {expected!r}, got {resolve(query)!r}"
        )
    get_name = ns["get_country_name"]
    assert get_name("MEX") == "Mexico"
    assert get_name("GBR") == "United Kingdom"


# --- §16 Brazil-soybean worked example --------------------------------------

def _check_brazil_soybean(ns: dict[str, Any], out: str) -> None:
    summary = ns["summary"]
    assert list(summary["focal_system_id"]) == ["Brazil"]
    row = summary.iloc[0]
    # Documented printed table:
    #   focal_system_id  IFS  PFS  TFS   MFE  IFCI  PFCI  TFCI
    # 0          Brazil  0.1  0.2  0.7  0.73   1.0  0.25  0.33
    assert row["IFS"] == pytest.approx(0.1)
    assert row["PFS"] == pytest.approx(0.2)
    assert row["TFS"] == pytest.approx(0.7)
    assert row["MFE"] == pytest.approx(0.73, abs=0.005)
    # Single-partner convention: the intracoupling self-loop gives
    # IFCI exactly 1 (maximal concentration), as §16 calls out.
    assert row["IFCI"] == 1.0
    assert row["PFCI"] == pytest.approx(0.25, abs=0.005)
    assert row["TFCI"] == pytest.approx(0.33, abs=0.005)
    # The snippet's print(summary[cols].round(2)) shows the doc values.
    assert "Brazil" in out
    assert "0.73" in out

    # "Orthogonal flow_type × coupling_type" continuation block —
    # the single matter-flow group must reproduce the same indicators.
    by_flow = ns["summary_by_flow"]
    assert "flow_type" in by_flow.columns
    assert list(by_flow["flow_type"]) == ["matter"]
    assert by_flow.iloc[0]["MFE"] == pytest.approx(row["MFE"])
    assert by_flow.iloc[0]["IFCI"] == 1.0


EXECUTABLE_SECTIONS: list[ExecSpec] = [
    ExecSpec(
        id="s7-structured-data",
        subsections=("Accessing structured data programmatically",),
        setup=_setup_structured_data,
        check=_check_structured_data,
    ),
    ExecSpec(
        id="s7-formatting-options",
        subsections=("Formatting options",),
        setup=_setup_formatting_options,
        check=_check_formatting_options,
    ),
    ExecSpec(
        id="s8-pericoupling-standalone",
        subsections=(
            "Standalone usage (no LLM needed)",
            "Adjacency standards (disputed borders & water-separated pairs)",
        ),
        setup=dict,
        check=_check_pericoupling_standalone,
    ),
    ExecSpec(
        id="s8-country-name-resolution",
        subsections=("Country name resolution",),
        setup=dict,
        check=_check_country_resolution,
    ),
    ExecSpec(
        id="s16-brazil-soybean",
        subsections=(
            "Brazil-soybean worked example",
            "Orthogonal flow_type × coupling_type analysis",
        ),
        setup=dict,
        check=_check_brazil_soybean,
        requires=("pandas",),
    ),
]


@pytest.mark.parametrize("spec", EXECUTABLE_SECTIONS, ids=lambda s: s.id)
def test_doc_section_executes(spec: ExecSpec):
    """Execute the allowlisted MANUAL.md sections and assert the
    documented outputs still match the real package (PR #55 guard)."""
    for module in spec.requires:
        pytest.importorskip(module)

    blocks: list[DocBlock] = []
    for heading in spec.subsections:
        matching = [b for b in PYTHON_BLOCKS if b.subsection == heading]
        assert matching, (
            f"MANUAL.md no longer has a ```python block under "
            f"'### {heading}'. If the heading was renamed, update the "
            f"'{spec.id}' ExecSpec in tests/test_docs_examples.py."
        )
        blocks.extend(matching)
    blocks.sort(key=lambda b: b.start_line)

    namespace: dict[str, Any] = {"__name__": "manual_doc_example"}
    namespace.update(spec.setup())
    captured = io.StringIO()
    for block in blocks:
        match = NEEDS_LLM_OR_NETWORK.search(block.code)
        assert match is None, (
            f"Doc example at {block.location} is on the execution "
            f"allowlist but now contains {match.group(0)!r}, which "
            "needs an API key / network / LLM. Either revert the doc "
            "edit or move the section off EXECUTABLE_SECTIONS."
        )
        code = compile(block.code, f"{MANUAL_MD}:{block.start_line}", "exec")
        with warnings.catch_warnings():
            # §16 intentionally emits UserWarnings (n=1 MFCI
            # convention); keep the doc-exec tier quiet either way.
            warnings.simplefilter("ignore")
            with redirect_stdout(captured):
                exec(code, namespace)  # noqa: S102 — doc snippets from this repo

    spec.check(namespace, captured.getvalue())


def test_llm_sections_stay_off_the_allowlist():
    """Sanity-check the split: LLM/network-dependent examples exist in
    MANUAL.md (Quick Start, §17 end-to-end workflow, ...) and none of
    them sit under an allowlisted heading."""
    llm_blocks = [
        b for b in PYTHON_BLOCKS if NEEDS_LLM_OR_NETWORK.search(b.code)
    ]
    assert llm_blocks, (
        "Expected MANUAL.md to contain LLM-dependent python examples; "
        "if they were all removed, NEEDS_LLM_OR_NETWORK and this test "
        "can be simplified."
    )
    allowlisted = {h for s in EXECUTABLE_SECTIONS for h in s.subsections}
    offenders = [
        b.location for b in llm_blocks if b.subsection in allowlisted
    ]
    assert not offenders, (
        "These allowlisted doc sections contain LLM/network "
        f"constructs and must not be executed: {offenders}"
    )

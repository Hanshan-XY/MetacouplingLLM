"""Prompt templates for the LLM-assisted helpers (PR #36).

System + user prompt pairs per spec §15.2.  Versioned via
``*_PROMPT_VERSION`` constants so the reproducibility trace
(``LLMTrace.prompt_version``) records exactly which prompt fired.
Bump the version string when changing a prompt's wording so old
traces stay attributable.

Templates use ``str.format(...)``-style placeholders; no f-string
interpolation at module load time.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Shared guardrail block prepended to every system prompt -- per spec §15.4
# ---------------------------------------------------------------------------

_GUARDRAILS = (
    "GUARDRAILS:\n"
    "- You MUST NOT invent numerical flow values, adjacency facts, "
    "or distances not present in the user's input.\n"
    "- You MUST NOT calculate final indicator values; that's done "
    "by deterministic code.\n"
    "- When information is insufficient, say so explicitly rather "
    "than guessing.\n"
    "- Return strictly structured output conforming to the provided "
    "schema (no commentary, no markdown fences around the JSON).\n"
)


# ---------------------------------------------------------------------------
# define_study()
# ---------------------------------------------------------------------------

DEFINE_STUDY_PROMPT_VERSION = "define_study_v1"

DEFINE_STUDY_SYSTEM = (
    "You convert a researcher's plain-language description of a "
    "metacoupling study into a structured configuration object.\n\n"
    + _GUARDRAILS +
    "\nYour role: extract the focal system, flow type, flow unit, "
    "and infer reasonable classification rules for intra-, peri-, "
    "and telecoupling based on the user's framing.  Do NOT invent "
    "specific countries / partners not mentioned by the user."
)

DEFINE_STUDY_USER = (
    "Convert the following study description into a structured "
    "configuration object that downstream code can use to set up "
    "indicator calculations.\n\n"
    "Study description:\n{description}"
)


# ---------------------------------------------------------------------------
# check_inputs()
# ---------------------------------------------------------------------------

CHECK_INPUTS_PROMPT_VERSION = "check_inputs_v1"

CHECK_INPUTS_SYSTEM = (
    "You inspect a tabular dataset summary and report whether it "
    "can support each of the three metacoupling indicator families:"
    " Metacoupled Flow Shares, Metacoupled Flow Evenness, and "
    "Metacoupled Flow Concentration Index.\n\n"
    + _GUARDRAILS +
    "\nRules:\n"
    "- Flow Shares + MFE need totals by coupling type per focal "
    "system (F_I, F_P, F_T).\n"
    "- MFCI needs PARTNER-LEVEL flows within each coupling type "
    "(not just totals).\n"
    "- If intracoupling is represented only as a single self-loop "
    "row (e.g., Brazil -> Brazil), surface that in warnings -- "
    "IFCI will not be substantively meaningful.\n"
    "- If units appear mixed in one column (e.g., tons and USD), "
    "surface that in warnings.\n"
    "- Adjacency information is REQUIRED only if coupling_type is "
    "not pre-classified."
)

CHECK_INPUTS_USER = (
    "Inspect this data summary and assess what indicators can be "
    "computed.\n\n"
    "Column names: {columns}\n"
    "Row count: {row_count}\n"
    "Detected coupling_type column: {has_coupling_col}\n"
    "Sample rows (first {sample_n}):\n{sample_rows}\n\n"
    "Return a structured assessment per the schema."
)


# ---------------------------------------------------------------------------
# classify_ambiguous_edges()
# ---------------------------------------------------------------------------

CLASSIFY_EDGES_PROMPT_VERSION = "classify_edges_v1"

CLASSIFY_EDGES_SYSTEM = (
    "You classify flow edges in a metacoupling network as "
    "intracoupling (I), pericoupling (P), telecoupling (T), or "
    "unknown when the user-provided context is insufficient.\n\n"
    + _GUARDRAILS +
    "\nDefinitions (Liu 2017):\n"
    "- I = flow occurs within the focal system itself\n"
    "- P = flow connects the focal system to an adjacent / "
    "neighbouring system\n"
    "- T = flow connects the focal system to a distant / "
    "non-adjacent system\n"
    "\nRules:\n"
    "- Use the study_config's pericoupling_rule and "
    "telecoupling_rule as your primary guide.  When the rule is "
    "ambiguous for a specific edge (e.g., supranational unions, "
    "archipelago neighbours, edges spanning multiple administrative "
    "levels), use your geographic / domain knowledge.\n"
    "- Return 'unknown' when you genuinely don't have enough "
    "information.  DO NOT invent adjacency facts.\n"
    "- Set needs_user_confirmation=true whenever confidence is "
    "'low' or the case is borderline."
)

CLASSIFY_EDGES_USER = (
    "Classify these edges using the study configuration below.  "
    "Return one entry per edge in the classifications array, "
    "preserving the input order so the caller can merge back by "
    "index.\n\n"
    "Study configuration:\n{study_config_json}\n\n"
    "Edges to classify (origin, destination):\n{edges_list}"
)


# ---------------------------------------------------------------------------
# interpret_results()
# ---------------------------------------------------------------------------

INTERPRET_RESULTS_PROMPT_VERSION = "interpret_results_v1"

_INTERPRET_AUDIENCES = {
    "academic": (
        "Write in formal academic prose suitable for a journal "
        "paper's Results section.  Use measured language; favour "
        "precise numerical claims grounded in the table.  Avoid "
        "narrative storytelling."
    ),
    "general": (
        "Write in plain language for a general policy audience.  "
        "Avoid jargon; explain framework terms briefly on first "
        "use.  Keep sentences short."
    ),
    "policy": (
        "Write in concise briefing-style language for a policy or "
        "decision-maker audience.  Lead with the headline finding, "
        "then evidence.  Avoid academic hedging."
    ),
}

INTERPRET_RESULTS_SYSTEM = (
    "You interpret a computed metacoupling indicator table and "
    "explain what it means.  You DO NOT recalculate or alter any "
    "numerical values.\n\n"
    + _GUARDRAILS +
    "\nAudience-specific style:\n{audience_style}\n"
    "\nStructure your output as one or two short paragraphs "
    "(not a bulleted list) covering:\n"
    "1. Relative size of intra-, peri-, and telecoupled flows "
    "(from IFS/PFS/TFS).\n"
    "2. Whether the three coupling types are balanced or "
    "dominated by one (from MFE).\n"
    "3. Within each coupling type, whether flows are concentrated "
    "in few partners or diversified (from IFCI/PFCI/TFCI).\n"
)

INTERPRET_RESULTS_USER = (
    "Interpret these indicator values for the audience described in "
    "the system prompt.  Use the numbers verbatim.\n\n"
    "Indicator table (one row per focal system / time group):\n"
    "{results_table}"
)


# ---------------------------------------------------------------------------
# write_methods()
# ---------------------------------------------------------------------------

WRITE_METHODS_PROMPT_VERSION = "write_methods_v1"

WRITE_METHODS_SYSTEM = (
    "You write a paper's Methods subsection describing the "
    "metacoupling indicators that were calculated.  Output is "
    "intended for direct paste into a manuscript.\n\n"
    + _GUARDRAILS +
    "\nRules:\n"
    "- Include the formulas for IFS/PFS/TFS, MFE, and MFCI.\n"
    "- Cite the established sources: Shannon (1948), "
    "Hirschman (1945), Hannah & Kay (1977), "
    "Laakso & Taagepera (1979).\n"
    "- Mention the metacoupling framework citation: Liu (2017).\n"
    "- Mention edge-case conventions: 0*ln(0)=0 for MFE; "
    "MFCI = NaN when F_ic = 0; MFCI = 1 by convention when "
    "n_ic = 1.\n"
    "- Keep it concise (3-5 sentences per indicator family).\n"
    "- Use formal academic prose, no bullet points."
)

WRITE_METHODS_USER = (
    "Write the Methods subsection describing the indicators "
    "specified below.  Format: a single coherent prose section, "
    "not a list.\n\n"
    "Indicator specification:\n{indicator_spec_json}"
)

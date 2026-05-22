"""Scholar-facing Markdown + Word exports for ``AnalysisResult``.

PR #31: turns the existing ``AnalysisResult`` (rich Python object with
``parsed``, ``formatted`` text, optional ``map`` Figure) into formats
scholars can drop straight into a paper draft:

- ``render_markdown(result, path=None)`` -- single Markdown string.
  When ``path`` is given, also writes the .md file and saves the map
  as ``<stem>_map.png`` alongside.
- ``render_docx(result, path=None)`` -- Word .docx file with headings,
  flow / evidence tables, and the map embedded.  Requires the optional
  ``python-docx`` dependency (``pip install metacouplingllm[export]``).

Both renderers read from the SAME structured intermediate
(``_build_sections``) so the two formats stay in sync.  Sections is
internal; not exposed on ``AnalysisResult`` to keep the public API
small per the PR #31 design discussion ("user-friendly way to show
the output, most extra work is not necessary").
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type-only
    from metacouplingllm.core import AnalysisResult


# ---------------------------------------------------------------------------
# Internal intermediate -- structured sections built from ParsedAnalysis
# ---------------------------------------------------------------------------


def _build_sections(result: "AnalysisResult") -> dict[str, Any]:
    """Pull structured pieces of the analysis into one dict for both
    renderers.  Reads ``result.parsed`` (ParsedAnalysis) directly rather
    than re-parsing ``result.formatted`` -- the data is already
    structured.

    Returns
    -------
    Dict shape:

    ```
    {
        "abstract": str,
        "coupling_classification": str,
        "intracoupling": dict | None,    # CouplingSection as dict
        "pericoupling": dict | None,
        "telecoupling": dict | None,
        "cross_coupling_interactions": list[str],
        "research_gaps": list[str],
        "evidence_coverage": str,
        "flows": list[dict],             # from map_data or fallback
        "web_sources": list[dict],       # [{id, title, url, model_summary}]
        "focal_country": str | None,
        "topic": str,                    # best-effort topic label
    }
    ```
    """
    parsed = result.parsed

    def _section_to_dict(section: Any) -> dict[str, Any] | None:
        if section is None or getattr(section, "is_empty", True):
            return None
        return {
            "systems": list(getattr(section, "systems", []) or []),
            "flows": list(getattr(section, "flows", []) or []),
            "agents": list(getattr(section, "agents", []) or []),
            "causes": dict(getattr(section, "causes", {}) or {}),
            "effects": dict(getattr(section, "effects", {}) or {}),
        }

    # Flows table: prefer Stage-3 structured map_data; fall back to
    # iterating coupling-section flow prose entries.
    flows: list[dict[str, Any]] = []
    map_data = getattr(parsed, "map_data", None)
    if isinstance(map_data, dict):
        raw_flows = map_data.get("flows") or []
        if isinstance(raw_flows, list):
            for flow in raw_flows:
                if not isinstance(flow, dict):
                    continue
                flows.append({
                    "category": str(flow.get("category", "")).strip(),
                    "source": str(flow.get("source", "")).strip(),
                    "target": str(flow.get("target", "")).strip(),
                    "bidirectional": bool(flow.get("bidirectional", False)),
                    "description": str(flow.get("description", "")).strip(),
                })
    if not flows:
        # Fallback: walk the parsed CouplingSection prose flows.  These
        # don't have ISO codes, just direction strings, but at least
        # something renders in the table.
        for section_name in ("intracoupling", "pericoupling", "telecoupling"):
            section = getattr(parsed, section_name, None)
            if section is None:
                continue
            for flow in getattr(section, "flows", []) or []:
                if not isinstance(flow, dict):
                    continue
                flows.append({
                    "category": str(flow.get("category", "")).strip(),
                    "source": "",
                    "target": "",
                    "bidirectional": False,
                    "description": str(flow.get("direction", "")).strip()
                    + (
                        (" — " + str(flow.get("description", "")).strip())
                        if flow.get("description")
                        else ""
                    ),
                })

    # Web sources: pull from result.web_map_signals or the assistant's
    # _last_web_results if surfaced.  Use the raw evidence list when
    # available; the IDs (W1, W2, ...) match the markers in formatted.
    web_sources: list[dict[str, str]] = []
    raw_web = None
    if isinstance(result.web_map_signals, dict):
        raw_web = result.web_map_signals.get("evidence")
    # Some assistants expose the raw list on the result via a
    # separate attribute populated by the pipeline; check defensively.
    if not raw_web:
        raw_web = getattr(result, "_web_sources_for_export", None)
    if isinstance(raw_web, list):
        for idx, src in enumerate(raw_web, 1):
            if not isinstance(src, dict):
                continue
            web_sources.append({
                "id": f"W{idx}",
                "title": str(src.get("title", "")).strip(),
                "url": str(src.get("url", "")).strip(),
                "model_summary": str(
                    src.get("model_summary", src.get("snippet", ""))
                ).strip(),
            })

    focal_country: str | None = None
    if isinstance(map_data, dict):
        fc = map_data.get("focal_country")
        if isinstance(fc, str) and fc.strip():
            focal_country = fc.strip()

    # Topic heuristic: first 80 chars of coupling_classification, or
    # fallback to a generic label.  Used only in the document title.
    topic = ""
    if parsed.coupling_classification:
        first = parsed.coupling_classification.split("\n")[0].strip()
        topic = first[:80] + ("…" if len(first) > 80 else "")
    if not topic:
        topic = "metacoupling analysis"

    return {
        "abstract": getattr(result, "abstract", "") or "",
        "coupling_classification": parsed.coupling_classification or "",
        "intracoupling": _section_to_dict(
            getattr(parsed, "intracoupling", None)
        ),
        "pericoupling": _section_to_dict(
            getattr(parsed, "pericoupling", None)
        ),
        "telecoupling": _section_to_dict(
            getattr(parsed, "telecoupling", None)
        ),
        "cross_coupling_interactions": list(
            getattr(parsed, "cross_coupling_interactions", []) or []
        ),
        "research_gaps": list(
            getattr(parsed, "research_gaps", []) or []
        ),
        "evidence_coverage": (
            getattr(parsed, "evidence_coverage_note", "") or ""
        ),
        "flows": flows,
        "web_sources": web_sources,
        "focal_country": focal_country,
        "topic": topic,
    }


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _md_coupling_section(
    parts: list[str], number: int, title: str, section: dict[str, Any] | None,
) -> None:
    """Render one coupling section as Markdown subsections."""
    if section is None:
        return
    parts.append(f"## {number}. {title}")
    parts.append("")

    if section["systems"]:
        parts.append(f"### {number}.1 Systems")
        for entry in section["systems"]:
            role = str(entry.get("role", "system")).title()
            name = str(entry.get("name", "")).strip()
            parts.append(f"- **{role}**" + (f": {name}" if name else ""))
            for key in (
                "human_subsystem", "natural_subsystem",
                "geographic_scope", "description",
            ):
                val = str(entry.get(key, "")).strip()
                if val:
                    label = key.replace("_", " ").capitalize()
                    parts.append(f"  - *{label}*: {val}")
        parts.append("")

    if section["flows"]:
        parts.append(f"### {number}.2 Flows")
        for idx, flow in enumerate(section["flows"], 1):
            category = str(flow.get("category", "unspecified")).title()
            direction = str(flow.get("direction", "")).strip()
            description = str(flow.get("description", "")).strip()
            line = f"{idx}. **[{category}]**"
            if direction:
                line += f" {direction}"
            if description:
                line += f" — {description}"
            parts.append(line)
        parts.append("")

    if section["agents"]:
        parts.append(f"### {number}.3 Agents")
        for agent in section["agents"]:
            level = str(agent.get("level", "")).title()
            name = str(agent.get("name", "")).strip()
            desc = str(agent.get("description", "")).strip()
            prefix = f"*{level}* " if level else ""
            suffix = f" — {desc}" if desc else ""
            parts.append(f"- {prefix}{name}{suffix}")
        parts.append("")

    for sub_idx, key, label in (
        (4, "causes", "Causes"),
        (5, "effects", "Effects"),
    ):
        items = section[key]
        if not items:
            continue
        parts.append(f"### {number}.{sub_idx} {label}")
        for category, entries in items.items():
            parts.append(f"- **{category.title()}**:")
            for entry in entries:
                parts.append(f"  - {entry}")
        parts.append("")


def render_markdown(
    result: "AnalysisResult", path: str | Path | None = None,
) -> str:
    """Convert ``result`` into a Markdown string.  Writes a file +
    saves the map as a sibling PNG when ``path`` is given."""
    s = _build_sections(result)
    parts: list[str] = []

    # Title
    focal = s["focal_country"] or "Case"
    parts.append(f"# Metacoupling Analysis — {focal}: {s['topic']}")
    parts.append("")

    # Abstract
    if s["abstract"]:
        parts.append("## Abstract")
        parts.append("")
        parts.append(s["abstract"])
        parts.append("")

    # §1 Classification
    if s["coupling_classification"]:
        parts.append("## 1. Coupling Classification")
        parts.append("")
        parts.append(s["coupling_classification"])
        parts.append("")

    # §2-§4 Coupling sections
    _md_coupling_section(parts, 2, "Intracoupling Analysis", s["intracoupling"])
    _md_coupling_section(parts, 3, "Pericoupling Analysis", s["pericoupling"])
    _md_coupling_section(parts, 4, "Telecoupling Analysis", s["telecoupling"])

    # §5 Cross-coupling
    if s["cross_coupling_interactions"]:
        parts.append("## 5. Cross-Coupling Interactions")
        parts.append("")
        for item in s["cross_coupling_interactions"]:
            parts.append(f"- {item}")
        parts.append("")

    # §6 Research gaps
    if s["research_gaps"]:
        parts.append("## 6. Research Gaps")
        parts.append("")
        for gap in s["research_gaps"]:
            parts.append(f"- {gap}")
        parts.append("")

    # §7 Evidence coverage
    if s["evidence_coverage"]:
        parts.append("## 7. Evidence Coverage")
        parts.append("")
        parts.append(s["evidence_coverage"])
        parts.append("")

    # Flows table
    if s["flows"]:
        parts.append("## Flows")
        parts.append("")
        parts.append("| Category | Source | Target | Bidirectional | Description |")
        parts.append("|---|---|---|---|---|")
        for flow in s["flows"]:
            # Title-case the category to match the existing
            # AnalysisFormatter convention (so "matter" → "Matter"
            # consistent with the text formatter PR #25 / PR #26 use).
            cat = (flow["category"] or "—").title()
            src = flow["source"] or "—"
            tgt = flow["target"] or "—"
            bi = "Yes" if flow["bidirectional"] else "No"
            desc = flow["description"] or ""
            # Escape pipes inside cells to keep the table valid.
            desc = desc.replace("|", "\\|")
            parts.append(f"| {cat} | {src} | {tgt} | {bi} | {desc} |")
        parts.append("")

    # Web sources
    if s["web_sources"]:
        parts.append("## Web Sources")
        parts.append("")
        for src in s["web_sources"]:
            line = f"- **[{src['id']}]** {src['title']}"
            if src["url"]:
                line += f" — <{src['url']}>"
            parts.append(line)
        parts.append("")

    # Map embed (only when we have a file path to write the PNG to).
    map_rel_name: str | None = None
    if path is not None and result.map is not None:
        path = Path(path)
        map_rel_name = f"{path.stem}_map.png"
        try:
            result.map.savefig(
                path.parent / map_rel_name,
                dpi=150,
                bbox_inches="tight",
            )
            parts.append("## Map")
            parts.append("")
            parts.append(f"![map]({map_rel_name})")
            parts.append("")
        except Exception as exc:  # pragma: no cover - matplotlib edge case
            parts.append(f"<!-- Map render failed: {exc} -->")
            parts.append("")

    text = "\n".join(parts).rstrip() + "\n"

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    return text


# ---------------------------------------------------------------------------
# Word (.docx) renderer -- requires python-docx
# ---------------------------------------------------------------------------


def render_docx(
    result: "AnalysisResult", path: str | Path | None = None,
) -> Path:
    """Convert ``result`` into a Word document.  Returns the
    ``pathlib.Path`` written.  Raises ``ImportError`` when
    ``python-docx`` is not installed."""
    try:
        from docx import Document
        from docx.shared import Inches, Pt
    except ImportError as exc:  # pragma: no cover - exercised in tests
        raise ImportError(
            "render_docx requires the optional ``python-docx`` "
            "dependency.  Install it with "
            "``pip install metacouplingllm[export]``."
        ) from exc

    s = _build_sections(result)
    out_path = Path(path) if path is not None else Path("metacoupling_report.docx")

    doc = Document()

    # Title
    focal = s["focal_country"] or "Case"
    doc.add_heading(
        f"Metacoupling Analysis — {focal}: {s['topic']}", level=0,
    )

    # Abstract
    if s["abstract"]:
        doc.add_heading("Abstract", level=1)
        doc.add_paragraph(s["abstract"])

    # §1 Classification
    if s["coupling_classification"]:
        doc.add_heading("1. Coupling Classification", level=1)
        doc.add_paragraph(s["coupling_classification"])

    # §2-§4 Coupling sections
    for number, title, section in (
        (2, "Intracoupling Analysis", s["intracoupling"]),
        (3, "Pericoupling Analysis", s["pericoupling"]),
        (4, "Telecoupling Analysis", s["telecoupling"]),
    ):
        if section is None:
            continue
        doc.add_heading(f"{number}. {title}", level=1)
        if section["systems"]:
            doc.add_heading(f"{number}.1 Systems", level=2)
            for entry in section["systems"]:
                role = str(entry.get("role", "system")).title()
                name = str(entry.get("name", "")).strip()
                p = doc.add_paragraph()
                run = p.add_run(role)
                run.bold = True
                if name:
                    p.add_run(f": {name}")
                for key in (
                    "human_subsystem", "natural_subsystem",
                    "geographic_scope", "description",
                ):
                    val = str(entry.get(key, "")).strip()
                    if val:
                        label = key.replace("_", " ").capitalize()
                        doc.add_paragraph(
                            f"{label}: {val}", style="List Bullet",
                        )
        if section["flows"]:
            doc.add_heading(f"{number}.2 Flows", level=2)
            for flow in section["flows"]:
                category = str(flow.get("category", "unspecified")).title()
                direction = str(flow.get("direction", "")).strip()
                description = str(flow.get("description", "")).strip()
                text = f"[{category}]"
                if direction:
                    text += f" {direction}"
                if description:
                    text += f" — {description}"
                doc.add_paragraph(text, style="List Number")
        if section["agents"]:
            doc.add_heading(f"{number}.3 Agents", level=2)
            for agent in section["agents"]:
                level = str(agent.get("level", "")).title()
                name = str(agent.get("name", "")).strip()
                desc = str(agent.get("description", "")).strip()
                text = f"{level + ' ' if level else ''}{name}"
                if desc:
                    text += f" — {desc}"
                doc.add_paragraph(text, style="List Bullet")
        for sub_idx, key, label in (
            (4, "causes", "Causes"),
            (5, "effects", "Effects"),
        ):
            items = section[key]
            if not items:
                continue
            doc.add_heading(f"{number}.{sub_idx} {label}", level=2)
            for category, entries in items.items():
                p = doc.add_paragraph()
                run = p.add_run(f"{category.title()}: ")
                run.bold = True
                p.add_run("; ".join(entries))

    # §5 Cross-coupling
    if s["cross_coupling_interactions"]:
        doc.add_heading("5. Cross-Coupling Interactions", level=1)
        for item in s["cross_coupling_interactions"]:
            doc.add_paragraph(item, style="List Bullet")

    # §6 Research gaps
    if s["research_gaps"]:
        doc.add_heading("6. Research Gaps", level=1)
        for gap in s["research_gaps"]:
            doc.add_paragraph(gap, style="List Bullet")

    # §7 Evidence coverage
    if s["evidence_coverage"]:
        doc.add_heading("7. Evidence Coverage", level=1)
        doc.add_paragraph(s["evidence_coverage"])

    # Flows table
    if s["flows"]:
        doc.add_heading("Flows", level=1)
        table = doc.add_table(rows=1, cols=5)
        table.style = "Light Grid Accent 1"
        hdr = table.rows[0].cells
        hdr[0].text = "Category"
        hdr[1].text = "Source"
        hdr[2].text = "Target"
        hdr[3].text = "Bidirectional"
        hdr[4].text = "Description"
        for flow in s["flows"]:
            row = table.add_row().cells
            row[0].text = (flow["category"] or "—").title()
            row[1].text = flow["source"] or "—"
            row[2].text = flow["target"] or "—"
            row[3].text = "Yes" if flow["bidirectional"] else "No"
            row[4].text = flow["description"] or ""

    # Web sources table
    if s["web_sources"]:
        doc.add_heading("Web Sources", level=1)
        table = doc.add_table(rows=1, cols=3)
        table.style = "Light Grid Accent 1"
        hdr = table.rows[0].cells
        hdr[0].text = "ID"
        hdr[1].text = "Title"
        hdr[2].text = "URL"
        for src in s["web_sources"]:
            row = table.add_row().cells
            row[0].text = src["id"]
            row[1].text = src["title"]
            row[2].text = src["url"]

    # Map figure
    if result.map is not None:
        doc.add_heading("Map", level=1)
        buf = BytesIO()
        try:
            result.map.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            doc.add_picture(buf, width=Inches(6.0))
        except Exception as exc:  # pragma: no cover - matplotlib edge case
            doc.add_paragraph(f"[Map render failed: {exc}]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    return out_path

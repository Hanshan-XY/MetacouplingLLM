"""
Human-readable formatting of coupling-first analysis results.
"""

from __future__ import annotations

from metacouplingllm.llm.parser import CouplingSection, ParsedAnalysis


_SEPARATOR = "=" * 72
_SUB_SEPARATOR = "-" * 40


class AnalysisFormatter:
    """Formats :class:`ParsedAnalysis` objects into readable text."""

    @staticmethod
    def _format_system_label(entry: dict[str, str]) -> str:
        """Return a display label for a parsed system role."""
        role = entry.get("role", "system").strip().lower()
        if role in {"focal", "adjacent", "sending", "receiving", "spillover"}:
            label = f"{role.title()} System"
        else:
            label = role.replace("_", " ").title()

        scope = entry.get("system_scope", "").strip().lower()
        if scope:
            label = f"{label} ({scope})"
        return label

    @staticmethod
    def _format_systems(section: CouplingSection) -> list[str]:
        lines: list[str] = []
        for entry in section.systems:
            lines.append(f"  [{AnalysisFormatter._format_system_label(entry)}]")
            if entry.get("name"):
                lines.append(f"    {entry['name']}")
            if entry.get("human_subsystem"):
                lines.append(f"    Human subsystem: {entry['human_subsystem']}")
            if entry.get("natural_subsystem"):
                lines.append(f"    Natural subsystem: {entry['natural_subsystem']}")
            if entry.get("geographic_scope"):
                lines.append(f"    Geographic scope: {entry['geographic_scope']}")
            if entry.get("description"):
                lines.append(f"    {entry['description']}")
            known = {
                "role",
                "name",
                "human_subsystem",
                "natural_subsystem",
                "geographic_scope",
                "description",
                "system_scope",
            }
            for key, value in entry.items():
                if key not in known and value:
                    lines.append(f"    {key.replace('_', ' ').title()}: {value}")
        return lines

    @staticmethod
    def _format_flows(section: CouplingSection) -> list[str]:
        lines: list[str] = []
        for idx, flow in enumerate(section.flows, 1):
            category = flow.get("category", "unspecified").title()
            direction = flow.get("direction", "Unspecified")
            description = flow.get("description", "")
            lines.append(f"  {idx}. [{category}] {direction}")
            if description:
                lines.append(f"     {description}")
        return lines

    @staticmethod
    def _format_agents(section: CouplingSection) -> list[str]:
        lines: list[str] = []
        for agent in section.agents:
            level = agent.get("level", "").title()
            name = agent.get("name", "")
            description = agent.get("description", "")
            prefix = f"[{level}] " if level else ""
            suffix = f" - {description}" if description else ""
            lines.append(f"  - {prefix}{name}{suffix}")
        return lines

    @staticmethod
    def _format_grouped_items(
        grouped: dict[str, list[str]],
    ) -> list[str]:
        lines: list[str] = []
        for category, items in grouped.items():
            lines.append(f"  {category.title()}:")
            for item in items:
                lines.append(f"    - {item}")
        return lines

    @staticmethod
    def _append_coupling_section(
        parts: list[str],
        number: int,
        title: str,
        section: CouplingSection | None,
    ) -> None:
        if section is None or section.is_empty:
            return
        parts.append(f"{number}. {title}")
        parts.append(_SUB_SEPARATOR)

        if section.systems:
            parts.append(f"  {number}.1 Systems Identification")
            parts.extend(AnalysisFormatter._format_systems(section))
            parts.append("")
        if section.flows:
            parts.append(f"  {number}.2 Flows Analysis")
            parts.extend(AnalysisFormatter._format_flows(section))
            parts.append("")
        if section.agents:
            parts.append(f"  {number}.3 Agents")
            parts.extend(AnalysisFormatter._format_agents(section))
            parts.append("")
        if section.causes:
            parts.append(f"  {number}.4 Causes")
            parts.extend(AnalysisFormatter._format_grouped_items(section.causes))
            parts.append("")
        if section.effects:
            parts.append(f"  {number}.5 Effects")
            parts.extend(AnalysisFormatter._format_grouped_items(section.effects))
            parts.append("")

    @staticmethod
    def format_full(analysis: ParsedAnalysis) -> str:
        """Produce a complete human-readable report."""
        if not analysis.is_parsed:
            return analysis.raw_text

        parts: list[str] = [
            _SEPARATOR,
            "METACOUPLING FRAMEWORK ANALYSIS",
            _SEPARATOR,
            "",
        ]

        if analysis.coupling_classification:
            parts.append("1. Coupling Classification")
            parts.append(_SUB_SEPARATOR)
            parts.append(analysis.coupling_classification)
            parts.append("")

        AnalysisFormatter._append_coupling_section(
            parts,
            2,
            "Intracoupling Analysis",
            analysis.intracoupling,
        )
        AnalysisFormatter._append_coupling_section(
            parts,
            3,
            "Pericoupling Analysis",
            analysis.pericoupling,
        )
        AnalysisFormatter._append_coupling_section(
            parts,
            4,
            "Telecoupling Analysis",
            analysis.telecoupling,
        )

        if analysis.cross_coupling_interactions:
            parts.append("5. Cross-coupling Interactions")
            parts.append(_SUB_SEPARATOR)
            for item in analysis.cross_coupling_interactions:
                parts.append(f"  - {item}")
            parts.append("")

        if analysis.research_gaps:
            parts.append("6. Research Gaps and Suggestions")
            parts.append(_SUB_SEPARATOR)
            for gap in analysis.research_gaps:
                parts.append(f"  - {gap}")
            parts.append("")

        if analysis.pericoupling_info:
            info = analysis.pericoupling_info
            if info.get("level") == "adm1":
                parts.append("PERICOUPLING DATABASE VALIDATION (SUBNATIONAL)")
            else:
                parts.append("PERICOUPLING DATABASE VALIDATION")
            parts.append(_SUB_SEPARATOR)
            for key, value in info.items():
                if key == "level":
                    continue
                if key == "pair_results":
                    for pair_result in str(value).split("; "):
                        parts.append(f"  {pair_result}")
                    continue
                label = key.replace("_", " ").title()
                parts.append(f"  {label}: {value}")
            parts.append("")

        parts.append(_SEPARATOR)
        return "\n".join(parts)

    @staticmethod
    def format_summary(analysis: ParsedAnalysis) -> str:
        """Produce a brief overview highlighting key findings."""
        if not analysis.is_parsed:
            raw = analysis.raw_text.strip()
            return raw[:500] + "..." if len(raw) > 500 else raw

        lines: list[str] = ["ANALYSIS SUMMARY", _SUB_SEPARATOR]
        if analysis.coupling_classification:
            first_line = analysis.coupling_classification.split("\n")[0].strip()
            lines.append(f"Classification: {first_line}")

        active = analysis.active_coupling_types()
        if active:
            lines.append(
                "Active coupling types: " + ", ".join(name.title() for name in active)
            )

        flow_count = sum(1 for _ in analysis.iter_flow_entries())
        if flow_count:
            lines.append(f"Number of flows: {flow_count}")

        gap_count = len(analysis.research_gaps)
        if gap_count:
            lines.append(f"Research gaps: {gap_count} items")

        return "\n".join(lines)

    @staticmethod
    def format_component(analysis: ParsedAnalysis, component: str) -> str:
        """Render a single component section."""
        component_lower = component.lower().strip()
        if component_lower in ("classification", "coupling_classification", "coupling"):
            return analysis.coupling_classification or "(No classification data)"
        if component_lower in ("intracoupling", "pericoupling", "telecoupling"):
            section = analysis.get_coupling_section(component_lower)
            if section is None or section.is_empty:
                return f"(No {component_lower} data)"
            pseudo = ParsedAnalysis(raw_text="")
            setattr(pseudo, component_lower, section)
            return AnalysisFormatter.format_full(pseudo)
        if component_lower in ("cross_coupling", "cross-coupling", "interactions"):
            if not analysis.cross_coupling_interactions:
                return "(No cross-coupling interaction data)"
            return "\n".join(f"- {item}" for item in analysis.cross_coupling_interactions)
        if component_lower in ("research gaps", "gaps", "suggestions", "research_gaps"):
            if not analysis.research_gaps:
                return "(No research gap data)"
            return "\n".join(f"- {item}" for item in analysis.research_gaps)
        return f"(Unknown component: {component})"

    @staticmethod
    def format_comparison(analyses: list[ParsedAnalysis]) -> str:
        """Produce a side-by-side comparison of multiple analyses."""
        if not analyses:
            return "(No analyses to compare)"
        if len(analyses) == 1:
            return AnalysisFormatter.format_full(analyses[0])

        lines: list[str] = [_SEPARATOR, "COMPARATIVE ANALYSIS", _SEPARATOR, ""]
        for idx, analysis in enumerate(analyses, 1):
            lines.append(f"--- Analysis {idx} ---")
            if analysis.coupling_classification:
                first = analysis.coupling_classification.split("\n")[0].strip()
                lines.append(f"  Classification: {first}")
            active = analysis.active_coupling_types()
            if active:
                lines.append(
                    "  Active coupling types: "
                    + ", ".join(name.title() for name in active)
                )
            flow_count = sum(1 for _ in analysis.iter_flow_entries())
            lines.append(f"  Flows: {flow_count}")
            lines.append(f"  Research gaps: {len(analysis.research_gaps)}")
            lines.append("")
        return "\n".join(lines)

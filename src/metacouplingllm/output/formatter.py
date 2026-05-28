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

        # §7 Evidence Coverage — model self-assessment block.  Skipped
        # entirely when the LLM omitted the section (preserves backward
        # compatibility with responses generated before the §7 prompt
        # instruction shipped).  ``suggested_followup_queries`` is
        # appended by ``MetacouplingAssistant._build_result`` (where the
        # web_map_signals are accessible); the formatter only renders
        # what's on the ParsedAnalysis itself.
        if analysis.evidence_coverage_note:
            parts.append("7. Evidence Coverage")
            parts.append(_SUB_SEPARATOR)
            parts.append(analysis.evidence_coverage_note)
            parts.append("")

        # PR #27 + PR #44 (v3): single COUPLING DATABASE VALIDATION
        # block with a flat layout — Focal (+ optional Core
        # subnational regions sub-line) + grouped Pericoupled list
        # + grouped Telecoupled list + Note.  Spillover-roled
        # countries are filtered out at the validator level (they
        # remain in parsed.systems["spillover"] for the framework
        # systems analysis above).
        #
        # History:
        # - PR #27 made the two validators independent (both
        #   populate their own ParsedAnalysis fields, no early
        #   return).
        # - PR #44 v1 renamed + reordered the two separate blocks.
        # - PR #44 v2 merged into one block with two sub-sections.
        # - PR #44 v3 (current): replaces sub-sections with a flat
        #   structure grouping pairs by verdict.  ADM1 validator
        #   only runs when the user's query named a subnational
        #   region; otherwise the country validator's
        #   core_subnational_regions key carries LLM-mentioned
        #   state names without DB-derived neighbour lists.
        #
        # Schema unchanged: both pericoupling_info and
        # country_pericoupling_info remain separate dataclass
        # fields, populated independently.
        country_info = analysis.country_pericoupling_info
        adm1_info = analysis.pericoupling_info
        if country_info or adm1_info:
            parts.append("COUPLING DATABASE VALIDATION")
            parts.append(_SUB_SEPARATOR)

            # PR #44 (v3.1): mode-aware intro + group labels.
            # When the user's query was country-scope
            # (national mode), the country validator sets
            # info["mode"] = "national" and the ADM1 sub-validator
            # is skipped.  When the user named a subnational
            # region, both validators run and pair_results can mix
            # country + ADM1 partners.  Use the country_info mode
            # signal; fall back to "subnational" when only ADM1
            # populated (rare edge case).
            mode = (country_info or {}).get("mode") or "subnational"
            if mode == "national":
                intro = (
                    "  This block cross-checks the LLM's coupled-system\n"
                    "  claims against the bundled coupling databases\n"
                    "  (country adjacency).  Pairs are labeled PERICOUPLED\n"
                    "  (adjacent) countries or TELECOUPLED (distant) countries\n"
                    "  per the database.  Core subnational regions are\n"
                    "  subnational regions identified based on the analysis\n"
                    "  results and are provided for reference only."
                )
                peri_label = "Pericoupled Countries:"
                tele_label = "Telecoupled Countries:"
            else:
                intro = (
                    "  This block cross-checks the LLM's coupled-system\n"
                    "  claims against the bundled coupling databases\n"
                    "  (country adjacency + ADM1 adjacency).  Pairs are\n"
                    "  labeled PERICOUPLED (adjacent) countries/subnational\n"
                    "  regions or TELECOUPLED (distant) countries/subnational\n"
                    "  regions per the database."
                )
                peri_label = "Pericoupled Countries/Subnational Regions:"
                tele_label = "Telecoupled Countries/Subnational Regions:"
            parts.append(intro)
            parts.append("")

            # Unified Focal System line.  Combines ADM1
            # focal_region (when present) and the country focal.
            focal_segments: list[str] = []
            if adm1_info and adm1_info.get("focal_region"):
                focal_segments.append(adm1_info["focal_region"])
            focal_country = (
                (adm1_info or {}).get("focal_country")
                or (country_info or {}).get("focal_country")
            )
            if focal_country:
                focal_segments.append(focal_country)
            if focal_segments:
                parts.append(f"  Focal System: {', '.join(focal_segments)}")
                # Core subnational regions: national-mode-only
                # sub-line listing LLM-mentioned states in the
                # focal country.
                csr = (country_info or {}).get("core_subnational_regions")
                if csr:
                    parts.append(f"    Core subnational regions: {csr}")
                parts.append("")

            # Bucket pair_results by verdict across both info
            # dicts.  Strip the trailing ": VERDICT" since the
            # group label conveys the verdict.
            pericoupled: list[str] = []
            telecoupled: list[str] = []
            for info_dict in (country_info, adm1_info):
                if not info_dict or not info_dict.get("pair_results"):
                    continue
                for line in str(info_dict["pair_results"]).split("; "):
                    line = line.strip()
                    if line.endswith(": PERICOUPLED"):
                        pericoupled.append(line[: -len(": PERICOUPLED")])
                    elif line.endswith(": TELECOUPLED"):
                        telecoupled.append(line[: -len(": TELECOUPLED")])

            if pericoupled:
                parts.append(f"  {peri_label}")
                for pair in pericoupled:
                    parts.append(f"    {pair}")
                parts.append("")
            if telecoupled:
                parts.append(f"  {tele_label}")
                for pair in telecoupled:
                    parts.append(f"    {pair}")
                parts.append("")

            # Combined Note at the bottom.  When both validators'
            # notes agree on consistency, collapse to one summary
            # line; otherwise surface each note separately so
            # disagreement signals aren't hidden by the merge.
            notes: list[tuple[str, str]] = []
            if country_info and country_info.get("note"):
                notes.append(("country", country_info["note"]))
            if adm1_info and adm1_info.get("note"):
                notes.append(("subnational", adm1_info["note"]))
            if notes:
                all_consistent = all(
                    "consider revising" not in n.lower()
                    for _, n in notes
                )
                if all_consistent and len(notes) > 1:
                    parts.append(
                        "  Note: LLM classification is consistent with the "
                        "coupling databases (country adjacency + ADM1 "
                        "adjacency)."
                    )
                elif all_consistent:
                    parts.append(f"  Note: {notes[0][1]}")
                else:
                    for scope, note_text in notes:
                        parts.append(f"  Note ({scope}): {note_text}")
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

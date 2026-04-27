"""
Response parser for coupling-first metacoupling analyses.

The parser expects top-level sections organised by coupling type:

- Coupling Classification
- Intracoupling Analysis
- Pericoupling Analysis
- Telecoupling Analysis
- Cross-coupling Interactions
- Research Gaps and Suggestions

Each active coupling section is parsed into a structured
``CouplingSection`` with internal subsections for systems, flows,
agents, causes, and effects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class CouplingSection:
    """Structured content for one coupling type."""

    systems: list[dict[str, str]] = field(default_factory=list)
    flows: list[dict[str, str]] = field(default_factory=list)
    agents: list[dict[str, str]] = field(default_factory=list)
    causes: dict[str, list[str]] = field(default_factory=dict)
    effects: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Return ``True`` when the section has no parsed content."""
        return not (
            self.systems
            or self.flows
            or self.agents
            or self.causes
            or self.effects
        )


def _normalize_system_role(role: str) -> str:
    """Normalize system-role names across coupling sections."""
    normalized = re.sub(r"\s*\([^)]*\)", "", role.strip().lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    aliases = {
        "focal": "focal",
        "focal system": "focal",
        "adjacent": "adjacent",
        "adjacent system": "adjacent",
        "sending": "sending",
        "sending system": "sending",
        "receiving": "receiving",
        "receiving system": "receiving",
        "spillover": "spillover",
        "spillover system": "spillover",
    }
    return aliases.get(normalized, normalized)


def _normalize_system_scope(scope: str | None) -> str:
    """Normalize optional system-scope qualifiers such as adjacent/distant."""
    if not scope:
        return ""
    normalized = scope.strip().lower().strip("() ")
    normalized = re.sub(r"\s+", " ", normalized)
    aliases = {
        "adjacent": "adjacent",
        "distant": "distant",
    }
    return aliases.get(normalized, normalized)


@dataclass
class ParsedAnalysis:
    """Structured representation of a coupling-first analysis response."""

    coupling_classification: str = ""
    intracoupling: CouplingSection | None = None
    pericoupling: CouplingSection | None = None
    telecoupling: CouplingSection | None = None
    cross_coupling_interactions: list[str] = field(default_factory=list)
    research_gaps: list[str] = field(default_factory=list)
    raw_text: str = ""
    pericoupling_info: dict[str, str] | None = None
    map_data: dict[str, object] | None = None

    @property
    def is_parsed(self) -> bool:
        """Return ``True`` if at least some structured data was extracted."""
        return bool(
            self.coupling_classification
            or self.cross_coupling_interactions
            or self.research_gaps
            or any(section and not section.is_empty for _, section in self.iter_coupling_sections())
        )

    def get_coupling_section(self, name: str) -> CouplingSection | None:
        """Return the named coupling section if present."""
        normalized = name.strip().lower()
        if normalized == "intracoupling":
            return self.intracoupling
        if normalized == "pericoupling":
            return self.pericoupling
        if normalized == "telecoupling":
            return self.telecoupling
        return None

    def iter_coupling_sections(
        self,
    ) -> Iterator[tuple[str, CouplingSection]]:
        """Yield active coupling sections in canonical order."""
        for name, section in (
            ("intracoupling", self.intracoupling),
            ("pericoupling", self.pericoupling),
            ("telecoupling", self.telecoupling),
        ):
            if section is not None:
                yield name, section

    def active_coupling_types(self) -> list[str]:
        """Return the active coupling-type names in canonical order."""
        return [name for name, section in self.iter_coupling_sections() if not section.is_empty]

    def iter_system_entries(
        self,
        *,
        role: str | None = None,
        coupling_type: str | None = None,
    ) -> Iterator[dict[str, str]]:
        """Yield parsed system entries across active coupling sections."""
        normalized_role = _normalize_system_role(role) if role else None
        for section_name, section in self.iter_coupling_sections():
            if coupling_type and section_name != coupling_type.lower():
                continue
            for entry in section.systems:
                entry_role = _normalize_system_role(entry.get("role", ""))
                if normalized_role and entry_role != normalized_role:
                    continue
                yield entry

    def get_system_entries(
        self,
        role: str,
        *,
        coupling_type: str | None = None,
    ) -> list[dict[str, str]]:
        """Return all system entries for a role."""
        return list(
            self.iter_system_entries(role=role, coupling_type=coupling_type)
        )

    def get_first_system_entry(
        self,
        role: str,
        *,
        coupling_type: str | None = None,
    ) -> dict[str, str] | None:
        """Return the first system entry for a role, if any."""
        return next(
            iter(self.get_system_entries(role, coupling_type=coupling_type)),
            None,
        )

    def get_system_detail(
        self,
        role: str,
        sub_field: str | None = None,
        *,
        coupling_type: str | None = None,
    ) -> str:
        """Return a system detail from the first matching system entry."""
        entry = self.get_first_system_entry(role, coupling_type=coupling_type)
        if entry is None:
            return ""
        if sub_field:
            return entry.get(sub_field, "")
        parts: list[str] = []
        if entry.get("name"):
            parts.append(entry["name"])
        if entry.get("human_subsystem"):
            parts.append(f"Human subsystem: {entry['human_subsystem']}")
        if entry.get("natural_subsystem"):
            parts.append(f"Natural subsystem: {entry['natural_subsystem']}")
        if entry.get("geographic_scope"):
            parts.append(f"Geographic scope: {entry['geographic_scope']}")
        if entry.get("description"):
            parts.append(entry["description"])
        return "; ".join(parts)

    def iter_flow_entries(
        self,
        *,
        coupling_type: str | None = None,
    ) -> Iterator[dict[str, str]]:
        """Yield parsed flow entries across active coupling sections."""
        for section_name, section in self.iter_coupling_sections():
            if coupling_type and section_name != coupling_type.lower():
                continue
            yield from section.flows

    def iter_agent_entries(
        self,
        *,
        coupling_type: str | None = None,
    ) -> Iterator[dict[str, str]]:
        """Yield parsed agent entries across active coupling sections."""
        for section_name, section in self.iter_coupling_sections():
            if coupling_type and section_name != coupling_type.lower():
                continue
            yield from section.agents

    def iter_category_items(
        self,
        kind: str,
        *,
        coupling_type: str | None = None,
    ) -> Iterator[tuple[str, str, str]]:
        """Yield ``(coupling_type, category, item)`` triples."""
        attr = kind.lower()
        if attr not in {"causes", "effects"}:
            return
        for section_name, section in self.iter_coupling_sections():
            if coupling_type and section_name != coupling_type.lower():
                continue
            grouped = getattr(section, attr)
            for category, items in grouped.items():
                for item in items:
                    yield section_name, category, item

    def iter_text_fragments(self) -> Iterator[str]:
        """Yield all substantive text fragments in the parsed analysis."""
        if self.coupling_classification:
            yield self.coupling_classification
        for _, section in self.iter_coupling_sections():
            for entry in section.systems:
                for value in entry.values():
                    if isinstance(value, str) and value:
                        yield value
            for flow in section.flows:
                for value in flow.values():
                    if isinstance(value, str) and value:
                        yield value
            for agent in section.agents:
                for value in agent.values():
                    if isinstance(value, str) and value:
                        yield value
            for grouped in (section.causes, section.effects):
                for items in grouped.values():
                    for item in items:
                        if item:
                            yield item
        for item in self.cross_coupling_interactions:
            if item:
                yield item
        for gap in self.research_gaps:
            if gap:
                yield gap


def _extract_sections(
    text: str,
    patterns: dict[str, re.Pattern[str]],
) -> dict[str, str]:
    """Split *text* into named sections using the supplied patterns."""
    found: list[tuple[int, str]] = []
    for name, pattern in patterns.items():
        match = pattern.search(text)
        if match:
            found.append((match.start(), name))

    if not found:
        return {}

    found.sort(key=lambda item: item[0])
    sections: dict[str, str] = {}
    for idx, (start, name) in enumerate(found):
        header_end = text.index("\n", start) + 1 if "\n" in text[start:] else len(text)
        body_end = found[idx + 1][0] if idx + 1 < len(found) else len(text)
        sections[name] = text[header_end:body_end].strip()
    return sections


_TOP_SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "coupling_classification": re.compile(
        r"^#+\s*\d*\.?\s*coupling\s+classification",
        re.IGNORECASE | re.MULTILINE,
    ),
    "intracoupling": re.compile(
        r"^#+\s*\d*\.?\s*intracoupling\s+analysis",
        re.IGNORECASE | re.MULTILINE,
    ),
    "pericoupling": re.compile(
        r"^#+\s*\d*\.?\s*pericoupling\s+analysis",
        re.IGNORECASE | re.MULTILINE,
    ),
    "telecoupling": re.compile(
        r"^#+\s*\d*\.?\s*telecoupling\s+analysis",
        re.IGNORECASE | re.MULTILINE,
    ),
    "cross_coupling_interactions": re.compile(
        r"^#+\s*\d*\.?\s*cross[-\s]+coupling\s+interactions?",
        re.IGNORECASE | re.MULTILINE,
    ),
    "research_gaps": re.compile(
        r"^#+\s*\d*\.?\s*(research\s+gaps?(?:\s*&\s*suggestions|\s+and\s+suggestions)?)",
        re.IGNORECASE | re.MULTILINE,
    ),
}

_COUPLING_SUBSECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "systems": re.compile(
        r"^#{1,6}\s*\d+(?:\.\d+)?\.?\s*systems?\s+identification",
        re.IGNORECASE | re.MULTILINE,
    ),
    "flows": re.compile(
        r"^#{1,6}\s*\d+(?:\.\d+)?\.?\s*flows?\s+analysis",
        re.IGNORECASE | re.MULTILINE,
    ),
    "agents": re.compile(
        r"^#{1,6}\s*\d+(?:\.\d+)?\.?\s*agents?",
        re.IGNORECASE | re.MULTILINE,
    ),
    "causes": re.compile(
        r"^#{1,6}\s*\d+(?:\.\d+)?\.?\s*causes?",
        re.IGNORECASE | re.MULTILINE,
    ),
    "effects": re.compile(
        r"^#{1,6}\s*\d+(?:\.\d+)?\.?\s*effects?",
        re.IGNORECASE | re.MULTILINE,
    ),
}


_BULLET_RE = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)

_CAUSE_EFFECT_CATEGORY_ALIASES: dict[str, str] = {
    "economic": "economic",
    "economics": "economic",
    "socioeconomic": "economic",
    "socio-economic": "economic",
    "financial": "economic",
    "market": "economic",
    "political": "political / institutional",
    "institutional": "political / institutional",
    "political / institutional": "political / institutional",
    "political/institutional": "political / institutional",
    "governance": "political / institutional",
    "ecological": "ecological / biological",
    "biological": "ecological / biological",
    "ecological / biological": "ecological / biological",
    "ecological/biological": "ecological / biological",
    "environmental": "ecological / biological",
    "biodiversity": "ecological / biological",
    "technological": "technological / infrastructural",
    "infrastructural": "technological / infrastructural",
    "technological / infrastructural": "technological / infrastructural",
    "technological/infrastructural": "technological / infrastructural",
    "infrastructure": "technological / infrastructural",
    "cultural": "cultural / social / demographic",
    "social": "cultural / social / demographic",
    "demographic": "cultural / social / demographic",
    "cultural / social / demographic": "cultural / social / demographic",
    "cultural/social/demographic": "cultural / social / demographic",
    "cultural / demographic / social": "cultural / social / demographic",
    "hydrological": "hydrological",
    "water": "hydrological",
    "climatic": "climatic / atmospheric",
    "atmospheric": "climatic / atmospheric",
    "climatic / atmospheric": "climatic / atmospheric",
    "climatic/atmospheric": "climatic / atmospheric",
    "climate": "climatic / atmospheric",
    "biogeochemical": "climatic / atmospheric",
    "geological": "geological / geomorphological",
    "geomorphological": "geological / geomorphological",
    "geological / geomorphological": "geological / geomorphological",
    "geological/geomorphological": "geological / geomorphological",
}


def _normalize_cause_effect_category(category: str) -> str:
    """Normalize cause/effect headings to the fixed category vocabulary."""
    normalized = category.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return _CAUSE_EFFECT_CATEGORY_ALIASES.get(normalized, normalized)


def _extract_bullets(text: str) -> list[str]:
    """Extract bullet items from text."""
    items: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if _BULLET_RE.match(line):
            clean = _BULLET_RE.sub("", line).strip()
            if clean:
                items.append(clean)
    return items


def _extract_categorized_bullets(text: str) -> dict[str, list[str]]:
    """Extract bullets grouped under bold headings."""
    result: dict[str, list[str]] = {}
    current_category = "general"
    for line in text.splitlines():
        stripped = line.strip()
        heading_match = re.match(
            r"\*{2}([^*]+)\*{2}\s*(?::|[-—–].*)?\s*$",
            stripped,
        )
        if heading_match:
            current_category = _normalize_cause_effect_category(
                heading_match.group(1)
            )
            continue
        if _BULLET_RE.match(stripped):
            clean = _BULLET_RE.sub("", stripped).strip()
            if clean:
                result.setdefault(current_category, []).append(clean)
    return result


_SYSTEM_ROLE_ALIASES: dict[str, str] = {
    "focal": "focal",
    "focal system": "focal",
    "adjacent": "adjacent",
    "adjacent system": "adjacent",
    "sending": "sending",
    "sending system": "sending",
    "receiving": "receiving",
    "receiving system": "receiving",
    "spillover": "spillover",
    "spillover system": "spillover",
}

_SYSTEM_SUBFIELD_ALIASES: dict[str, str] = {
    "human subsystem": "human_subsystem",
    "human subsystems": "human_subsystem",
    "human component": "human_subsystem",
    "human components": "human_subsystem",
    "natural subsystem": "natural_subsystem",
    "natural subsystems": "natural_subsystem",
    "natural component": "natural_subsystem",
    "natural components": "natural_subsystem",
    "geographic scope": "geographic_scope",
    "geography": "geographic_scope",
    "location": "geographic_scope",
}


def _parse_systems(text: str) -> list[dict[str, str]]:
    """Parse a systems-identification block into system-entry dicts."""
    system_heading_re = re.compile(
        r"(?:#{1,6}\s*)?\*{0,2}\s*"
        r"(Focal|Adjacent|Sending|Receiving|Spillover)\s*(?:System)?"
        r"(?:\s*\(([^)]+)\))?"
        r"\s*\*{0,2}\s*:\s*(.*)$",
        re.IGNORECASE,
    )

    systems: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    def _flush() -> None:
        nonlocal current
        if current:
            systems.append(dict(current))
        current = None

    found_heading = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        heading_match = system_heading_re.match(stripped)
        if heading_match:
            _flush()
            found_heading = True
            role = _normalize_system_role(heading_match.group(1))
            current = {"role": role}
            scope = _normalize_system_scope(heading_match.group(2))
            if scope:
                current["system_scope"] = scope
            name = heading_match.group(3).strip().rstrip("*").strip()
            if name:
                current["name"] = name
            continue

        if current is not None:
            clean = _BULLET_RE.sub("", stripped).strip()
            if not clean:
                continue
            subfield_match = re.match(r"\*{0,2}([^:*]+)\*{0,2}\s*:\s*(.*)", clean)
            if subfield_match:
                label = subfield_match.group(1).strip().lower()
                value = subfield_match.group(2).strip()
                normalized = _SYSTEM_SUBFIELD_ALIASES.get(label)
                if normalized:
                    current[normalized] = value
                else:
                    current[label.replace(" ", "_")] = value
            else:
                existing = current.get("description", "")
                current["description"] = (
                    f"{existing} {clean}".strip() if existing else clean
                )

    _flush()
    if found_heading and systems:
        return systems

    # Fallback: flat bullets such as "- Focal: Michigan"
    for item in _extract_bullets(text):
        match = re.match(r"\*{0,2}([^:*]+)\*{0,2}\s*:\s*(.*)", item)
        if not match:
            continue
        role = _normalize_system_role(match.group(1))
        if role in {"focal", "adjacent", "sending", "receiving", "spillover"}:
            entry = {"role": role, "name": match.group(2).strip()}
            scope_match = re.search(r"\(([^)]+)\)", match.group(1))
            scope = _normalize_system_scope(scope_match.group(1) if scope_match else None)
            if scope:
                entry["system_scope"] = scope
            systems.append(entry)
    return systems


_AGENT_LEVEL_ALIASES: dict[str, str] = {
    "individuals / households": "individuals / households",
    "individuals/households": "individuals / households",
    "individual": "individuals / households",
    "individuals": "individuals / households",
    "person": "individuals / households",
    "people": "individuals / households",
    "household": "individuals / households",
    "households": "individuals / households",
    "family": "individuals / households",
    "families": "individuals / households",
    "resident": "individuals / households",
    "residents": "individuals / households",
    "farmer": "individuals / households",
    "farmers": "individuals / households",
    "consumer": "individuals / households",
    "consumers": "individuals / households",
    "tourist": "individuals / households",
    "tourists": "individuals / households",
    "firms / traders / corporations": "firms / traders / corporations",
    "firms/traders/corporations": "firms / traders / corporations",
    "firm": "firms / traders / corporations",
    "firms": "firms / traders / corporations",
    "trader": "firms / traders / corporations",
    "traders": "firms / traders / corporations",
    "company": "firms / traders / corporations",
    "companies": "firms / traders / corporations",
    "corporation": "firms / traders / corporations",
    "corporations": "firms / traders / corporations",
    "business": "firms / traders / corporations",
    "businesses": "firms / traders / corporations",
    "exporter": "firms / traders / corporations",
    "exporters": "firms / traders / corporations",
    "importer": "firms / traders / corporations",
    "importers": "firms / traders / corporations",
    "governments / policymakers": "governments / policymakers",
    "governments/policymakers": "governments / policymakers",
    "government": "governments / policymakers",
    "governments": "governments / policymakers",
    "policymaker": "governments / policymakers",
    "policymakers": "governments / policymakers",
    "policy maker": "governments / policymakers",
    "policy makers": "governments / policymakers",
    "regulator": "governments / policymakers",
    "regulators": "governments / policymakers",
    "agency": "governments / policymakers",
    "agencies": "governments / policymakers",
    "organization": "organizations / NGOs",
    "organizations": "organizations / NGOs",
    "organisations": "organizations / NGOs",
    "organisation": "organizations / NGOs",
    "organizations / ngos": "organizations / NGOs",
    "organizations/ngos": "organizations / NGOs",
    "ngo": "organizations / NGOs",
    "ngos": "organizations / NGOs",
    "nonprofit": "organizations / NGOs",
    "non-profit": "organizations / NGOs",
    "university": "organizations / NGOs",
    "universities": "organizations / NGOs",
    "research institution": "organizations / NGOs",
    "research institutions": "organizations / NGOs",
    "non-human agents": "non-human agents",
    "nonhuman agents": "non-human agents",
    "non-human": "non-human agents",
    "nonhuman": "non-human agents",
    "animal": "non-human agents",
    "animals": "non-human agents",
    "species": "non-human agents",
    "organism": "non-human agents",
    "organisms": "non-human agents",
    "pathogen": "non-human agents",
    "pathogens": "non-human agents",
    "pest": "non-human agents",
    "pests": "non-human agents",
    "invasive species": "non-human agents",
    "livestock": "non-human agents",
    "crop": "non-human agents",
    "crops": "non-human agents",
}


def _detect_agent_level(text: str) -> str:
    """Detect an agent level in a text string."""
    bracket_match = re.search(r"\[([^\]]+)\]", text)
    if bracket_match:
        label = re.sub(r"\s+", " ", bracket_match.group(1).strip().lower())
        if label in _AGENT_LEVEL_ALIASES:
            return _AGENT_LEVEL_ALIASES[label]
    lowered = text.lower()
    for label, normalized in _AGENT_LEVEL_ALIASES.items():
        if label in lowered:
            return normalized
    return ""


def _parse_agents(text: str) -> list[dict[str, str]]:
    """Parse agent bullets into structured dicts."""
    agents: list[dict[str, str]] = []
    for item in _extract_bullets(text):
        agent: dict[str, str] = {}
        bracket_match = re.match(r"\[([^\]]+)\]\s*(.*)", item)
        if bracket_match:
            label = re.sub(r"\s+", " ", bracket_match.group(1).strip().lower())
            level = _AGENT_LEVEL_ALIASES.get(label, "")
            rest = bracket_match.group(2).strip()
            if level:
                agent["level"] = level
            colon_match = re.match(r"([^:]+):\s*(.*)", rest)
            if colon_match:
                agent["name"] = colon_match.group(1).strip()
                agent["description"] = colon_match.group(2).strip()
            else:
                agent["name"] = rest
            agents.append(agent)
            continue

        colon_match = re.match(r"([^:]+):\s*(.*)", item)
        if colon_match:
            agent["name"] = colon_match.group(1).strip()
            agent["description"] = colon_match.group(2).strip()
        else:
            agent["name"] = item

        detected_level = _detect_agent_level(item)
        if detected_level:
            agent["level"] = detected_level
        agents.append(agent)
    return agents


_FLOW_CATEGORIES = {
    "capital", "energy", "information", "matter", "organisms", "people",
}

_FLOW_CATEGORY_ALIASES: dict[str, str] = {
    "material": "matter",
    "materials": "matter",
    "goods": "matter",
    "commodity": "matter",
    "commodities": "matter",
    "financial": "capital",
    "finance": "capital",
    "monetary": "capital",
    "money": "capital",
    "economic": "capital",
    "knowledge": "information",
    "data": "information",
    "technology": "information",
    "human": "people",
    "labor": "people",
    "labour": "people",
    "migration": "people",
    "organism": "organisms",
    "biological": "organisms",
    "species": "organisms",
    "wildlife": "organisms",
    "power": "energy",
    "electricity": "energy",
}


def _detect_flow_category(text: str) -> str:
    """Detect a canonical flow category in a string."""
    lowered = text.lower()
    for category in _FLOW_CATEGORIES:
        if category in lowered:
            return category
    for alias, canonical in _FLOW_CATEGORY_ALIASES.items():
        if alias in lowered:
            return canonical
    return ""


def _detect_direction(text: str) -> str:
    """Detect a direction string containing an arrow."""
    arrow_match = re.search(
        r"([A-Za-z][\w\s/,()'-]*?)\s*(?:→|->|=>)\s*([A-Za-z][\w\s/,()'-]*)",
        text,
    )
    if arrow_match:
        return f"{arrow_match.group(1).strip()} → {arrow_match.group(2).strip()}"
    bidirectional_match = re.search(
        r"([A-Za-z][\w\s/,()'-]*?)\s*(?:↔|<->|<=>)\s*([A-Za-z][\w\s/,()'-]*)",
        text,
    )
    if bidirectional_match:
        return (
            f"Bidirectional ({bidirectional_match.group(1).strip()} ↔ "
            f"{bidirectional_match.group(2).strip()})"
        )
    if "bidirectional" in text.lower():
        return "Bidirectional"
    return ""


def _parse_multiline_flows(text: str) -> list[dict[str, str]]:
    """Parse multi-line flow blocks under bold category headings."""
    flow_heading_re = re.compile(
        r"(?:\d+\.?\s*)?\*{2}\s*(?:\d+\.?\s*)?([\w\s/]+?flows?)\s*\*{2}\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    headings = list(flow_heading_re.finditer(text))
    if not headings:
        return []

    flows: list[dict[str, str]] = []
    for idx, match in enumerate(headings):
        block_start = match.end()
        block_end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
        block = text[block_start:block_end].strip()
        category = _detect_flow_category(match.group(1))
        direction = ""
        description_parts: list[str] = []
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            clean = _BULLET_RE.sub("", stripped).strip()
            if not clean:
                continue
            direction_match = re.match(r"\*{0,2}direction\*{0,2}\s*:\s*(.*)", clean, re.IGNORECASE)
            if direction_match:
                direction = direction_match.group(1).strip()
                continue
            description_match = re.match(r"\*{0,2}description\*{0,2}\s*:\s*(.*)", clean, re.IGNORECASE)
            if description_match:
                description_parts.append(description_match.group(1).strip())
                continue
            if not direction:
                detected_direction = _detect_direction(clean)
                if detected_direction:
                    direction = detected_direction
                    continue
            description_parts.append(clean)

        flow: dict[str, str] = {}
        if category:
            flow["category"] = category
        if direction:
            flow["direction"] = direction
        if description_parts:
            flow["description"] = " ".join(description_parts)
        if flow:
            flows.append(flow)
    return flows


def _parse_flows(text: str) -> list[dict[str, str]]:
    """Parse a flow-analysis block into flow-entry dicts."""
    multiline = _parse_multiline_flows(text)
    if multiline:
        return multiline

    flows: list[dict[str, str]] = []
    for item in _extract_bullets(text):
        flow: dict[str, str] = {}
        bracket_match = re.match(r"\[([^\]]+)\]\s*([^:]+):\s*(.*)", item)
        if bracket_match:
            flow["category"] = _detect_flow_category(bracket_match.group(1)) or bracket_match.group(1).strip().lower()
            flow["direction"] = bracket_match.group(2).strip()
            flow["description"] = bracket_match.group(3).strip()
            flows.append(flow)
            continue

        category_match = re.search(r"\*{0,2}category\*{0,2}\s*:\s*([^|*]+)", item, re.IGNORECASE)
        direction_match = re.search(r"\*{0,2}direction\*{0,2}\s*:\s*([^|*]+)", item, re.IGNORECASE)
        description_match = re.search(r"\*{0,2}description\*{0,2}\s*:\s*(.*)", item, re.IGNORECASE)
        if category_match:
            flow["category"] = _detect_flow_category(category_match.group(1)) or category_match.group(1).strip().lower()
        if direction_match:
            flow["direction"] = direction_match.group(1).strip()
        if description_match:
            flow["description"] = description_match.group(1).strip()
        if flow:
            flows.append(flow)
            continue

        detected_direction = _detect_direction(item)
        detected_category = _detect_flow_category(item)
        if detected_category:
            flow["category"] = detected_category
        if detected_direction:
            flow["direction"] = detected_direction
        if item:
            flow.setdefault("description", item)
        if flow:
            flows.append(flow)
    return flows


def _parse_coupling_section(text: str) -> CouplingSection | None:
    """Parse one coupling-type top-level section."""
    subsections = _extract_sections(text, _COUPLING_SUBSECTION_PATTERNS)
    if not subsections:
        return None

    section = CouplingSection()
    if "systems" in subsections:
        section.systems = _parse_systems(subsections["systems"])
    if "flows" in subsections:
        section.flows = _parse_flows(subsections["flows"])
    if "agents" in subsections:
        section.agents = _parse_agents(subsections["agents"])
    if "causes" in subsections:
        section.causes = _extract_categorized_bullets(subsections["causes"])
    if "effects" in subsections:
        section.effects = _extract_categorized_bullets(subsections["effects"])
    return None if section.is_empty else section


def parse_analysis(response_text: str) -> ParsedAnalysis:
    """Parse a coupling-first LLM response into ``ParsedAnalysis``."""
    result = ParsedAnalysis(raw_text=response_text)
    sections = _extract_sections(response_text, _TOP_SECTION_PATTERNS)
    if not sections:
        return result

    result.coupling_classification = sections.get("coupling_classification", "")
    if "intracoupling" in sections:
        result.intracoupling = _parse_coupling_section(sections["intracoupling"])
    if "pericoupling" in sections:
        result.pericoupling = _parse_coupling_section(sections["pericoupling"])
    if "telecoupling" in sections:
        result.telecoupling = _parse_coupling_section(sections["telecoupling"])
    if "cross_coupling_interactions" in sections:
        interactions = _extract_bullets(sections["cross_coupling_interactions"])
        result.cross_coupling_interactions = interactions or [
            line.strip()
            for line in sections["cross_coupling_interactions"].splitlines()
            if line.strip()
        ]
    if "research_gaps" in sections:
        result.research_gaps = _extract_bullets(sections["research_gaps"])

    return result

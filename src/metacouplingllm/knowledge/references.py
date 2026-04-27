"""
Academic references for the metacoupling and telecoupling frameworks.

These references are injected into prompts so the LLM can cite authoritative
sources when advising researchers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    """An academic reference used as a knowledge source."""

    authors: str
    year: int
    title: str
    journal: str
    key_contribution: str


CORE_REFERENCES: list[Reference] = [
    Reference(
        authors="Liu, J., Hull, V., Batistella, M., DeFries, R., Dietz, T., Fu, F., ... & Zhu, C.",
        year=2013,
        title="Framing sustainability in a telecoupled world",
        journal="Ecology and Society, 18(2), 26",
        key_contribution=(
            "Introduced the telecoupling framework with five components "
            "(systems, flows, agents, causes, effects) and 14 telecoupling "
            "categories. Provided foundational examples including the "
            "Brazil-China soybean trade and red imported fire ant invasion."
        ),
    ),
    Reference(
        authors="Liu, J.",
        year=2017,
        title="Integration across a metacoupled world",
        journal="Ecology and Society, 22(4), 29",
        key_contribution=(
            "Proposed the metacoupling framework as an umbrella concept "
            "encompassing intracoupling (within systems), pericoupling "
            "(between adjacent systems), and telecoupling (between distant "
            "systems). Provided formal comparison table of the three coupling "
            "types (Table 1), a six-phase operationalization procedure, and "
            "detailed Wolong Nature Reserve example with panda loans, tourism, "
            "and marriage immigration. Discussed system boundaries, governance, "
            "and agent-based modeling for metacoupled systems."
        ),
    ),
    Reference(
        authors="Deines, J. M., Liu, X., & Liu, J.",
        year=2016,
        title="Telecoupling in urban water systems: An examination of Beijing's imported water supply",
        journal="Water International, 41(2), 251-270",
        key_contribution=(
            "Applied the telecoupling framework to urban water supply, "
            "identifying three telecouplings in Beijing's water system: "
            "inter-basin water transfer (SNWTP), virtual water imports, "
            "and payments for ecosystem services (PES)."
        ),
    ),
    Reference(
        authors="Zhao, Z., Liu, J., Xia, W., & Yang, H.",
        year=2021,
        title=(
            "Metacoupling of sustainable development goals: "
            "Synergies and tradeoffs across panda reserves"
        ),
        journal="Global Environmental Change, Under review (based on provided manuscript)",
        key_contribution=(
            "Demonstrated how metacoupling analysis across 67 giant panda "
            "nature reserves reveals synergies and tradeoffs among SDGs. "
            "Showed that tourism and panda loan telecouplings generate "
            "cross-system interactions affecting multiple SDGs simultaneously."
        ),
    ),
    Reference(
        authors="Liu, J.",
        year=2023,
        title="Leveraging the metacoupling framework for sustainability science and global sustainable development",
        journal="National Science Review, 10(7), nwad090",
        key_contribution=(
            "Comprehensive review of metacoupling advances. Revealed effects "
            "of metacoupling on SDG performance across borders; introduced "
            "coupling transformation stages (noncoupling, coupling, decoupling, "
            "recoupling); documented cascading interactions across space; "
            "re-examined Tobler's First Law showing telecouplings often stronger "
            "than pericouplings; expanded nexus approach across space; proposed "
            "flow-based governance shifting from place-based governance."
        ),
    ),
]


def format_references() -> str:
    """Format all core references as a text block for prompt injection.

    Returns
    -------
    A formatted string listing each reference with its key contribution.
    """
    lines: list[str] = ["## KEY REFERENCES\n"]
    for ref in CORE_REFERENCES:
        lines.append(
            f"- {ref.authors} ({ref.year}). {ref.title}. "
            f"*{ref.journal}*."
        )
        lines.append(f"  Key contribution: {ref.key_contribution}\n")
    return "\n".join(lines)

"""
Map visualization for metacoupling analysis.

Requires optional dependencies::

    pip install metacouplingllm[viz]

Three map functions are provided:

- :func:`plot_focal_country_map` — color all countries by coupling type
  relative to a single focal country (database-only, no LLM needed).
- :func:`plot_analysis_map` — color countries based on an LLM analysis
  result (:class:`~metacouplingllm.llm.parser.ParsedAnalysis`).
- :func:`plot_focal_adm1_map` — color subnational (ADM1) regions by
  coupling type relative to a focal ADM1 region.

Each takes the same arguments as the function it wraps in
:mod:`~metacouplingllm.visualization.worldmap` or
:mod:`~metacouplingllm.visualization.adm1_map`, which it imports only
when called.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure

    from metacouplingllm.llm.parser import ParsedAnalysis
    from metacouplingllm.visualization.adm1_map import Adm1CouplingColors
    from metacouplingllm.visualization.worldmap import CouplingColors


def plot_focal_country_map(
    focal_country: str,
    *,
    title: str | None = None,
    colors: CouplingColors | None = None,
    figsize: tuple[float, float] = (15, 8),
    custom_shapefile: str | Path | None = None,
    adm0_shapefile: str | Path | None = None,
    mentioned_countries: set[str] | None = None,
    de_facto_borders: bool = True,
    coupling_standard: str = "moderate",
) -> matplotlib.figure.Figure:
    """Generate a world map colored by coupling type relative to a focal country.

    Requires: ``pip install metacouplingllm[viz]``

    See :func:`metacouplingllm.visualization.worldmap.plot_focal_country_map`
    for full documentation.
    """
    from metacouplingllm.visualization.worldmap import (
        plot_focal_country_map as _impl,
    )

    return _impl(
        focal_country,
        title=title,
        colors=colors,
        figsize=figsize,
        custom_shapefile=custom_shapefile,
        adm0_shapefile=adm0_shapefile,
        mentioned_countries=mentioned_countries,
        de_facto_borders=de_facto_borders,
        coupling_standard=coupling_standard,
    )


def plot_analysis_map(
    parsed_analysis: ParsedAnalysis,
    *,
    focal_role: str = "sending",
    title: str | None = None,
    colors: CouplingColors | None = None,
    figsize: tuple[float, float] = (15, 8),
    custom_shapefile: str | Path | None = None,
    adm0_shapefile: str | Path | None = None,
    extra_mentioned_countries: set[str] | None = None,
    mentioned_countries_override: set[str] | None = None,
    focal_code_override: str | None = None,
    flows: list[dict[str, str]] | None = None,
    de_facto_borders: bool = True,
    coupling_standard: str = "moderate",
) -> matplotlib.figure.Figure:
    """Generate a world map from a parsed LLM analysis result.

    Requires: ``pip install metacouplingllm[viz]``

    See :func:`metacouplingllm.visualization.worldmap.plot_analysis_map`
    for full documentation.
    """
    from metacouplingllm.visualization.worldmap import plot_analysis_map as _impl

    return _impl(
        parsed_analysis,
        focal_role=focal_role,
        title=title,
        colors=colors,
        figsize=figsize,
        custom_shapefile=custom_shapefile,
        adm0_shapefile=adm0_shapefile,
        extra_mentioned_countries=extra_mentioned_countries,
        mentioned_countries_override=mentioned_countries_override,
        focal_code_override=focal_code_override,
        flows=flows,
        de_facto_borders=de_facto_borders,
        coupling_standard=coupling_standard,
    )


def plot_focal_adm1_map(
    focal_adm1: str,
    *,
    shapefile: str | Path | None = None,
    title: str | None = None,
    colors: Adm1CouplingColors | None = None,
    figsize: tuple[float, float] = (15, 8),
    zoom_to_focal: bool = False,
    mentioned_countries: set[str] | None = None,
    mentioned_adm1_codes: set[str] | None = None,
    flows: list[dict[str, str]] | None = None,
    de_facto_borders: bool = True,
    coupling_standard: str = "moderate",
) -> matplotlib.figure.Figure:
    """Generate a subnational (ADM1) map colored by coupling type.

    Requires: ``pip install metacouplingllm[viz]``

    See :func:`metacouplingllm.visualization.adm1_map.plot_focal_adm1_map`
    for full documentation.
    """
    from metacouplingllm.visualization.adm1_map import (
        plot_focal_adm1_map as _impl,
    )

    return _impl(
        focal_adm1,
        shapefile=shapefile,
        title=title,
        colors=colors,
        figsize=figsize,
        zoom_to_focal=zoom_to_focal,
        mentioned_countries=mentioned_countries,
        mentioned_adm1_codes=mentioned_adm1_codes,
        flows=flows,
        de_facto_borders=de_facto_borders,
        coupling_standard=coupling_standard,
    )

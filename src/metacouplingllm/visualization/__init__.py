"""
Map visualization for metacoupling analysis.

Requires optional dependencies::

    pip install metacoupling[viz]

Three map functions are provided:

- :func:`plot_focal_country_map` — color all countries by coupling type
  relative to a single focal country (database-only, no LLM needed).
- :func:`plot_analysis_map` — color countries based on an LLM analysis
  result (:class:`~metacouplingllm.llm.parser.ParsedAnalysis`).
- :func:`plot_focal_adm1_map` — color subnational (ADM1) regions by
  coupling type relative to a focal ADM1 region.
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
) -> matplotlib.figure.Figure:
    """Generate a world map colored by coupling type relative to a focal country.

    Requires: ``pip install metacoupling[viz]``

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
    )


def plot_analysis_map(
    parsed_analysis: ParsedAnalysis,
    *,
    focal_role: str = "sending",
    title: str | None = None,
    colors: CouplingColors | None = None,
    figsize: tuple[float, float] = (15, 8),
    custom_shapefile: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """Generate a world map from a parsed LLM analysis result.

    Requires: ``pip install metacoupling[viz]``

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
    )


def plot_focal_adm1_map(
    focal_adm1: str,
    *,
    shapefile: str | Path | None = None,
    title: str | None = None,
    colors: Adm1CouplingColors | None = None,
    figsize: tuple[float, float] = (15, 8),
    zoom_to_focal: bool = False,
) -> matplotlib.figure.Figure:
    """Generate a subnational (ADM1) map colored by coupling type.

    Requires: ``pip install metacoupling[viz]``

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
    )

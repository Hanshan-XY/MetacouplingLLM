"""
Subnational (ADM1) map rendering for metacoupling framework analysis.

Generates choropleth maps where first-level administrative divisions (ADM1)
are color-coded by their coupling type (intracoupling, pericoupling,
telecoupling) relative to a focal ADM1 region.

Uses **geopandas** and **matplotlib** (optional dependencies).  Install
with ``pip install metacoupling[viz]``.

The base map (World Bank Admin 1 GeoPackage) can be provided by the user
or auto-downloaded on first use.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from metacouplingllm.knowledge.adm1_pericoupling import (
    get_adm1_country,
    get_adm1_info,
    get_adm1_neighbors,
)

if TYPE_CHECKING:
    import geopandas as gpd
    import matplotlib.figure
    import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------


def _check_dependencies() -> None:
    """Raise :exc:`ImportError` with install instructions if deps are missing."""
    missing: list[str] = []
    try:
        import geopandas  # noqa: F401
    except ImportError:
        missing.append("geopandas")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append("matplotlib")
    if missing:
        raise ImportError(
            f"Visualization requires {', '.join(missing)}. "
            f"Install with:  pip install metacoupling[viz]"
        )


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Adm1CouplingColors:
    """Color scheme for ADM1 metacoupling maps.

    Override any field to customise the palette.
    """

    intracoupling: str = "#D4E79E"  # light yellow-green (focal region)
    pericoupling: str = "#4CAF50"   # bright green (adjacent regions)
    telecoupling: str = "#ADD8E6"   # light blue (not adjacent, any country)
    na: str = "#D3D3D3"            # light grey (not in database)
    ocean: str = "#DCEEFB"          # light water blue (oceans + Caspian)
    lake: str = "#4A90D9"          # medium-dark blue (lakes overlay)
    border: str = "#888888"         # region border lines
    disputed: str = "#BFBFBF"      # darker grey for disputed territories
    disputed_outline: str = "#4D4D4D"  # darker outline so disputes stay visible
    disputed_hatch: str = "///"    # matplotlib hatch pattern


DEFAULT_ADM1_COLORS = Adm1CouplingColors()


# ---------------------------------------------------------------------------
# World Bank Admin 1 GeoPackage management
# ---------------------------------------------------------------------------

_WB_BOUNDARIES_RELEASE_BASE_URL = (
    "https://github.com/Hanshan-XY/wb-boundaries-data/releases/download/"
    "v2025-06-17"
)
_WB_ADM1_DOWNLOAD_ENV_VAR = "METACOUPLING_ADM1_DOWNLOAD_URL"
_WB_ADM1_URL = (
    f"{_WB_BOUNDARIES_RELEASE_BASE_URL}/"
    "World.Bank.Official.Boundaries.-.Admin.1.gpkg"
)
_WB_ADM1_FILENAMES = (
    "World Bank Official Boundaries - Admin 1.gpkg",
    "World.Bank.Official.Boundaries.-.Admin.1.gpkg",
    "WB_GAD_ADM1.gpkg",
    "wb_adm1.gpkg",
)

_WB_ADM1_DOWNLOAD_INSTRUCTIONS = (
    "The World Bank Admin 1 GeoPackage is required for ADM1 maps.\n"
    "Download it from:\n"
    "  https://github.com/Hanshan-XY/wb-boundaries-data/releases/tag/v2025-06-17\n"
    "Then pass the file path:\n"
    "  plot_focal_adm1_map('MEX001', shapefile='path/to/Admin 1.gpkg')"
)


def _get_adm1_cache_dir() -> Path:
    """Return the directory for caching the World Bank Admin 1 GeoPackage.

    Respects the ``METACOUPLING_CACHE_DIR`` environment variable.
    Falls back to ``~/.cache/metacoupling``.
    """
    env = os.environ.get("METACOUPLING_CACHE_DIR")
    if env:
        return Path(env) / "wb_adm1"
    return Path.home() / ".cache" / "metacoupling" / "wb_adm1"


def _get_adm1_geodataframe(
    shapefile: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Load an ADM1 GeoDataFrame.

    Parameters
    ----------
    shapefile:
        Path to a user-provided GeoPackage, shapefile, or GeoJSON.
        If ``None``, looks for a cached copy or raises an error with
        download instructions.

    Returns
    -------
    geopandas.GeoDataFrame
        ADM1 regions with normalised ``adm1_code`` and ``iso_code`` columns.
    """
    import geopandas as _gpd

    if shapefile is not None:
        # ---- User-provided shapefile ----
        world = _gpd.read_file(str(shapefile))
        return _normalise_adm1_columns(world)

    # ---- Look for cached copy ----
    cache_dir = _get_adm1_cache_dir()

    # Check for common file names in the cache directory
    for fname in _WB_ADM1_FILENAMES:
        cached = cache_dir / fname
        if cached.exists():
            world = _gpd.read_file(str(cached))
            return _normalise_adm1_columns(world)

    # Also check for any .gpkg file in the cache directory
    if cache_dir.exists():
        for gpkg in cache_dir.glob("*.gpkg"):
            world = _gpd.read_file(str(gpkg))
            return _normalise_adm1_columns(world)

    downloaded = _download_adm1_geopackage()
    if downloaded is not None and downloaded.exists():
        world = _gpd.read_file(str(downloaded))
        return _normalise_adm1_columns(world)

    # ---- No cached file found ----
    raise FileNotFoundError(
        f"No World Bank Admin 1 GeoPackage found in cache "
        f"({cache_dir}).\n\n{_WB_ADM1_DOWNLOAD_INSTRUCTIONS}"
    )


def _download_adm1_geopackage() -> Path | None:
    """Download and cache the default ADM1 GeoPackage."""
    cache_dir = _get_adm1_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    download_url = os.environ.get(_WB_ADM1_DOWNLOAD_ENV_VAR, _WB_ADM1_URL)
    filename = (
        Path(download_url).name
        or "World.Bank.Official.Boundaries.-.Admin.1.gpkg"
    )
    target = cache_dir / filename

    try:
        if not target.exists():
            urllib.request.urlretrieve(download_url, target)
        return target
    except Exception:
        return None


def _normalise_adm1_columns(world: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalise column names in the ADM1 GeoDataFrame.

    Ensures the result has ``adm1_code`` and ``iso_code`` columns.
    """
    # Normalise ADM1 code column
    if "adm1_code" not in world.columns:
        if "ADM1CD_c" in world.columns:
            world["adm1_code"] = world["ADM1CD_c"]
        elif "ADM1_CODE" in world.columns:
            world["adm1_code"] = world["ADM1_CODE"]
        else:
            raise ValueError(
                "ADM1 shapefile must contain an 'ADM1CD_c' or "
                "'adm1_code' column with ADM1 region codes."
            )

    # Normalise country ISO code column
    if "iso_code" not in world.columns:
        if "ISO_A3" in world.columns:
            world["iso_code"] = world["ISO_A3"]
        elif "ISO_A3_EH" in world.columns:
            world["iso_code"] = world["ISO_A3_EH"]
        else:
            raise ValueError(
                "ADM1 shapefile must contain an 'ISO_A3' or "
                "'iso_code' column with ISO alpha-3 country codes."
            )

    return world


# ---------------------------------------------------------------------------
# ADM1 classification
# ---------------------------------------------------------------------------


def _classify_adm1(
    focal_code: str,
    focal_country: str,
    neighbor_codes: set[str],
    shapefile_codes: set[str],
    shapefile_countries: dict[str, str],
    db_countries: set[str],
    mentioned_countries: set[str] | None = None,
    mentioned_adm1_codes: set[str] | None = None,
) -> dict[str, str]:
    """Classify every ADM1 region in the shapefile by coupling type.

    Parameters
    ----------
    focal_code:
        ADM1 code of the focal region.
    focal_country:
        ISO alpha-3 code of the focal region's country.
    neighbor_codes:
        ADM1 codes adjacent to the focal region (from the database).
    shapefile_codes:
        All ADM1 codes present in the shapefile.
    shapefile_countries:
        Mapping of ADM1 code → ISO alpha-3 code from the shapefile.
    db_countries:
        Set of ISO alpha-3 codes that appear in the ADM1 database.
    mentioned_countries:
        When provided, only regions whose country is in this set (or
        that are neighbors of the focal region) will be classified as
        pericoupling/telecoupling.  Regions belonging to countries
        **not** in this set are classified as ``"na"`` (grey).
        When ``None``, all DB countries are coloured (legacy behaviour).
    mentioned_adm1_codes:
        When provided, strict ADM1-level filtering is used for regions
        in the focal country: only regions in this set are coloured
        (pericoupling if a DB neighbor, telecoupling otherwise).  All
        other focal-country regions become ``"na"``.  Regions in other
        countries fall back to the country-level ``mentioned_countries``
        filter.  When ``None``, all DB neighbors are coloured
        pericoupling regardless of whether they appear in the analysis.

    Returns
    -------
    ``{adm1_code: "intracoupling"|"pericoupling"|"telecoupling"|"na"}``
    """
    classification: dict[str, str] = {}
    for code in shapefile_codes:
        if code == focal_code:
            classification[code] = "intracoupling"
            continue

        region_country = shapefile_countries.get(code)

        # STRICT MODE: when mentioned_adm1_codes is provided, apply
        # the substantive-evidence filter to ALL regions, including
        # cross-border DB neighbors. This prevents the LLM-hint-echo
        # problem where a DB neighbor (e.g., Santa Cruz, Bolivia)
        # gets coloured pericoupling even though the LLM has no
        # substantive evidence of actual interaction.
        if mentioned_adm1_codes is not None:
            if code in mentioned_adm1_codes:
                if code in neighbor_codes:
                    classification[code] = "pericoupling"
                else:
                    # Mentioned but not a DB neighbor — could be same
                    # country (non-adjacent) or a different country.
                    classification[code] = "telecoupling"
            else:
                # Not mentioned. If the region's country is a
                # mentioned country OTHER than the focal country,
                # colour it as telecoupling at the country level
                # (e.g., China). All other cases — including
                # non-mentioned DB neighbors like Santa Cruz —
                # become NA.
                if (
                    mentioned_countries is not None
                    and region_country is not None
                    and region_country != focal_country
                    and region_country in mentioned_countries
                ):
                    classification[code] = "telecoupling"
                else:
                    classification[code] = "na"
            continue

        # LEGACY MODE: mentioned_adm1_codes is None — old behaviour
        # (all DB neighbors coloured pericoupling regardless of
        # analysis content).
        if code in neighbor_codes:
            classification[code] = "pericoupling"
        elif mentioned_countries is not None:
            # Analysis-driven mode: only color countries mentioned in analysis
            if region_country and region_country in mentioned_countries:
                classification[code] = "telecoupling"
            else:
                classification[code] = "na"
        elif (
            region_country == focal_country
            or region_country in db_countries
        ):
            classification[code] = "telecoupling"
        else:
            classification[code] = "na"
    return classification


# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------


def _select_adm1_detail_countries(
    focal_country: str,
    neighbor_codes: set[str],
    shapefile_countries: dict[str, str],
) -> set[str]:
    """Return countries that should retain ADM1 detail on the map.

    ADM1 detail is reserved for the focal country and any countries that
    contain pericoupled neighbor regions. Telecoupled countries that are
    only referenced at country scale are rendered as dissolved country
    polygons instead of full subnational mosaics.
    """
    detail_countries: set[str] = {focal_country}
    for code in neighbor_codes:
        iso = shapefile_countries.get(code)
        if iso:
            detail_countries.add(iso)
    return detail_countries


def _dissolve_country_layer(
    country_layer_raw: gpd.GeoDataFrame,
    color_map: dict[str, str],
) -> gpd.GeoDataFrame | None:
    """Dissolve ADM1 regions to country polygons while preserving color.

    Countries collapsed out of ADM1 detail should still keep their coupling
    color on the map, so this helper aggregates the geometries and assigns
    each dissolved country a single coupling class.
    """
    if len(country_layer_raw) == 0:
        return None

    coupling_priority = {
        "na": 0,
        "telecoupling": 1,
        "pericoupling": 2,
        "intracoupling": 3,
    }
    country_coupling: dict[str, str] = {}
    for iso_code, group in country_layer_raw.groupby("iso_code"):
        values = [
            value for value in group["coupling"].tolist()
            if isinstance(value, str)
        ]
        if not values:
            country_coupling[iso_code] = "na"
            continue
        country_coupling[iso_code] = max(
            values,
            key=lambda value: coupling_priority.get(value, -1),
        )

    country_layer = country_layer_raw.dissolve(
        by="iso_code",
        as_index=False,
        aggfunc="first",
    )
    country_layer["coupling"] = (
        country_layer["iso_code"].map(country_coupling).fillna("na")
    )
    country_layer["_color"] = country_layer["coupling"].map(color_map)
    return country_layer


def _get_adm1_centroid(
    world: gpd.GeoDataFrame,
    adm1_code: str,
) -> tuple[float, float] | None:
    """Return the representative point for an ADM1 region."""
    row = world.loc[world["adm1_code"] == adm1_code]
    if row.empty:
        return None
    geom = row.geometry.values[0]
    point = geom.representative_point()
    return (point.x, point.y)


def _draw_mixed_flow_arrows(
    ax: object,
    world: gpd.GeoDataFrame,
    flows: list[dict[str, str]],
    *,
    focal_adm1: str | None = None,
    focal_country_iso: str | None = None,
) -> list:
    """Draw ADM1 domestic arrows and country-level international arrows.

    When ``focal_adm1`` and ``focal_country_iso`` are provided, flow
    endpoints that resolve to the focal country are substituted with
    the focal ADM1 region's centroid. This way, a flow like
    "Mato Grosso, Brazil → China" starts at Mato Grosso's centroid
    instead of Brazil's country centroid.
    """
    import re

    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    from metacouplingllm.knowledge.countries import resolve_country_code
    from metacouplingllm.visualization.worldmap import (
        _FLOW_ARROW_COLORS,
        _FLOW_ARROW_DEFAULT_COLOR,
        _get_country_centroid,
        _resolve_flow_endpoints,
    )

    drawn_categories: dict[str, str] = {}
    country_gdf = world.dissolve(by="iso_code", as_index=False)

    # Pre-compute the focal ADM1 centroid once (used for substitution).
    focal_adm1_xy: tuple[float, float] | None = None
    if focal_adm1 and focal_country_iso:
        focal_adm1_xy = _get_adm1_centroid(world, focal_adm1)

    def _resolve_with_focal_substitution(
        flow: dict[str, str],
    ) -> tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
        bool,
    ]:
        """Resolve flow endpoints, substituting focal ADM1 for focal country."""
        # First, get the country-level endpoints from the shared resolver.
        src_xy, tgt_xy, is_bidir = _resolve_flow_endpoints(flow, country_gdf)

        if focal_adm1_xy is None or focal_country_iso is None:
            return src_xy, tgt_xy, is_bidir

        # Re-parse the direction to identify which endpoint (if any)
        # matches the focal country, so we can swap in the ADM1 centroid.
        direction = flow.get("direction", "")
        if not direction:
            return src_xy, tgt_xy, is_bidir

        cleaned = re.sub(r"[Bb]idirectional\s*\(?", "", direction)
        cleaned = cleaned.rstrip(")")
        parts = re.split(r"\s*(?:→|->|=>|↔|<->|<=>)\s*", cleaned)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 2:
            return src_xy, tgt_xy, is_bidir

        src_name = parts[0]
        tgt_name = re.sub(r"\s*\(.*\)", "", parts[1]).strip()
        src_code = resolve_country_code(src_name)
        tgt_code = resolve_country_code(tgt_name)

        if src_code == focal_country_iso:
            src_xy = focal_adm1_xy
        if tgt_code == focal_country_iso:
            tgt_xy = focal_adm1_xy

        return src_xy, tgt_xy, is_bidir

    for i, flow in enumerate(flows):
        if flow.get("source_adm1") and flow.get("target_adm1"):
            src_xy = _get_adm1_centroid(world, flow["source_adm1"])
            tgt_xy = _get_adm1_centroid(world, flow["target_adm1"])
            is_bidir = bool(flow.get("is_bidirectional"))
        else:
            src_xy, tgt_xy, is_bidir = _resolve_with_focal_substitution(flow)
        if src_xy is None or tgt_xy is None:
            continue

        category = flow.get("category", "").lower()
        color = _FLOW_ARROW_COLORS.get(category, _FLOW_ARROW_DEFAULT_COLOR)
        cat_label = category.title() if category else "Flow"
        if cat_label not in drawn_categories:
            drawn_categories[cat_label] = color

        is_domestic_adm1 = bool(
            flow.get("source_adm1") and flow.get("target_adm1"),
        )
        offset = (i * 0.3) - (len(flows) * 0.15)
        base_rad = 0.28 if is_domestic_adm1 else 0.15
        arrow = FancyArrowPatch(
            posA=src_xy,
            posB=tgt_xy,
            arrowstyle="->" if not is_bidir else "<->",
            connectionstyle=f"arc3,rad={base_rad + offset * 0.05}",
            color=color,
            linewidth=2.0,
            mutation_scale=15,
            zorder=5,
            alpha=0.8,
        )
        ax.add_patch(arrow)

    handles = []
    for cat_label, color in drawn_categories.items():
        handles.append(
            mpatches.Patch(
                facecolor=color,
                edgecolor=color,
                label=f"{cat_label} Flow",
            )
        )
    return handles


def _render_adm1_map(
    world: gpd.GeoDataFrame,
    classification: dict[str, str],
    colors: Adm1CouplingColors,
    title: str,
    figsize: tuple[float, float],
    db_countries: set[str],
    adm1_detail_countries: set[str] | None = None,
    zoom_bounds: tuple[float, float, float, float] | None = None,
    flows: list[dict[str, str]] | None = None,
    mentioned_countries: set[str] | None = None,
    focal_adm1: str | None = None,
    focal_country_iso: str | None = None,
) -> matplotlib.figure.Figure:
    """Render the classified ADM1 map.

    Only countries in ``adm1_detail_countries`` are rendered at ADM1
    level. All other countries are dissolved into single country-level
    polygons (no internal ADM1 boundaries), while preserving their
    coupling colors.

    Returns a :class:`matplotlib.figure.Figure`.
    """
    import geopandas as _gpd
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as _plt

    fig, ax = _plt.subplots(1, 1, figsize=figsize)

    # Assign coupling type and colour to each geometry
    world = world.copy()
    world["coupling"] = (
        world["adm1_code"].map(classification).fillna("na")
    )

    color_map = {
        "intracoupling": colors.intracoupling,
        "pericoupling": colors.pericoupling,
        "telecoupling": colors.telecoupling,
        "na": colors.na,
    }
    world["_color"] = world["coupling"].map(color_map)

    # Split into two layers:
    #  1) Countries to show at ADM1 level (subnational boundaries)
    #  2) Countries to dissolve into country-level polygons (no internal ADM1 boundaries)
    if adm1_detail_countries is None:
        adm1_detail_countries = set()
    adm1_mask = world["iso_code"].isin(adm1_detail_countries)
    adm1_layer = world[adm1_mask]
    country_layer_raw = world[~adm1_mask]

    # Dissolve countries outside the ADM1-detail set to country polygons.
    country_layer = _dissolve_country_layer(country_layer_raw, color_map)
    if country_layer is not None and len(country_layer) > 0:
        country_layer.plot(
            ax=ax,
            color=country_layer["_color"],
            edgecolor=colors.border,
            linewidth=0.3,
        )

    # Plot ADM1-level regions on top
    if len(adm1_layer) > 0:
        adm1_layer.plot(
            ax=ax,
            color=adm1_layer["_color"],
            edgecolor=colors.border,
            linewidth=0.3,
        )

    # Overlay disputed territories with hatching using World Bank-only
    # disputed / indeterminate polygons.
    disputed_gdf = _get_disputed_overlay_geodataframe(world)
    if disputed_gdf is not None and len(disputed_gdf) > 0:
        from metacouplingllm.visualization.worldmap import _plot_disputed_overlay

        _plot_disputed_overlay(ax, disputed_gdf, colors, zorder=1.5)

    # Draw flow arrows. International flows stay country-level, while
    # same-country nearby flows can use ADM1-to-ADM1 centroids.
    # When focal_adm1 is provided, international flows touching the
    # focal country anchor on the focal ADM1 centroid instead.
    flow_handles: list = []
    if flows:
        flow_handles = _draw_mixed_flow_arrows(
            ax,
            world,
            flows,
            focal_adm1=focal_adm1,
            focal_country_iso=focal_country_iso,
        )

    # Zoom to focal country extent if requested
    if zoom_bounds is not None:
        margin = 2.0  # degrees
        ax.set_xlim(zoom_bounds[0] - margin, zoom_bounds[2] + margin)
        ax.set_ylim(zoom_bounds[1] - margin, zoom_bounds[3] + margin)

    # Legend
    legend_items = [
        mpatches.Patch(
            facecolor=colors.na,
            edgecolor="black",
            label="NA",
        ),
        mpatches.Patch(
            facecolor=colors.intracoupling,
            edgecolor="black",
            label="Intracoupling (Focal Region)",
        ),
        mpatches.Patch(
            facecolor=colors.pericoupling,
            edgecolor="black",
            label="Pericoupling (Adjacent)",
        ),
        mpatches.Patch(
            facecolor=colors.telecoupling,
            edgecolor="black",
            label="Telecoupling (Non-Adjacent)",
        ),
        mpatches.Patch(
            facecolor=colors.disputed,
            edgecolor=colors.disputed_outline,
            hatch=colors.disputed_hatch * 2,
            label="Disputed / Indeterminate",
        ),
    ]
    # Add flow arrow legend items
    legend_items.extend(flow_handles)

    ax.legend(
        handles=legend_items,
        loc="lower left",
        fontsize=9,
        title="Categories",
        title_fontsize=10,
        frameon=True,
        fancybox=True,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_axis_off()
    ax.set_facecolor(colors.ocean)
    fig.patch.set_facecolor(colors.ocean)
    fig.tight_layout()

    return fig


def _get_disputed_overlay_geodataframe(
    world: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame | None:
    """Build a disputed-territory overlay for ADM1 maps.

    The World Bank ADM1 GeoPackage does not always carry separate disputed
    polygons such as Western Sahara or Kashmir. Reuse the World Bank
    disputed-area helper so the overlay stays aligned with the World Bank
    boundary sources used elsewhere in the package.
    """
    try:
        from metacouplingllm.visualization.worldmap import (
            _get_disputed_territories_overlay,
        )

        return _get_disputed_territories_overlay(world.crs)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
) -> matplotlib.figure.Figure:
    """Generate a subnational map colored by coupling type relative to a focal ADM1 region.

    All ADM1 regions are classified into four categories:

    - **Intracoupling** (light yellow-green): the focal ADM1 region itself.
    - **Pericoupling** (bright green): regions sharing a border with the
      focal region.
    - **Telecoupling** (light blue): distant systems linked to the focal
      region. On the map, country-level telecoupled systems are usually
      dissolved to country polygons rather than showing every ADM1 boundary.
    - **NA** (grey): regions not in the database.

    Parameters
    ----------
    focal_adm1:
        ADM1 code of the focal region (e.g., ``"MEX001"``).
    shapefile:
        Path to a World Bank Admin 1 GeoPackage (``.gpkg``), shapefile,
        or GeoJSON.  If ``None``, looks for a cached copy in
        ``~/.cache/metacoupling/wb_adm1/``.  Download the GeoPackage
        from https://datacatalog.worldbank.org/search/dataset/0038272 .
    title:
        Custom map title.  Defaults to
        ``"ADM1 Metacoupling Map: {Region Name} ({Country})"``.
    colors:
        Custom :class:`Adm1CouplingColors` instance.
    figsize:
        Figure size in inches ``(width, height)``.
    zoom_to_focal:
        If ``True``, zoom the map to show only the focal country and
        its neighbors.
    mentioned_countries:
        When provided (typically from a parsed LLM analysis), only
        regions whose country is in this set are coloured as telecoupling.
        All other distant regions are grey.  The focal country is always
        included automatically.  When ``None``, the legacy behaviour is
        used (all DB countries are coloured). This controls coloring, not
        which countries keep ADM1 boundary detail.
    mentioned_adm1_codes:
        When provided, strict ADM1-level filtering is used for regions
        in the focal country: only ADM1 regions in this set are coloured
        (pericoupling if a DB neighbor, telecoupling otherwise).  All
        other focal-country regions become grey (NA).  This prevents
        non-adjacent focal-country states from being incorrectly coloured
        just because the focal country is in ``mentioned_countries``.
        Regions in other countries still use the country-level
        ``mentioned_countries`` filter.
    flows:
        Optional list of parsed flow dicts (from ``ParsedAnalysis.flows``).
        When provided, curved arrows are drawn on the map showing the
        direction of each flow between countries and, for nearby
        same-country flows, between ADM1 regions.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered map.  Call ``fig.savefig("map.png")`` to save.

    Raises
    ------
    ImportError
        If geopandas or matplotlib are not installed.
    FileNotFoundError
        If no shapefile is provided and no cached copy is available.
    ValueError
        If *focal_adm1* is not in the ADM1 database.

    Examples
    --------
    >>> fig = plot_focal_adm1_map(
    ...     "MEX001",
    ...     shapefile="World Bank Official Boundaries - Admin 1.gpkg",
    ... )
    >>> fig.savefig("mexico_adm1.png", dpi=150, bbox_inches="tight")
    """
    _check_dependencies()

    focal_adm1 = focal_adm1.strip()

    # Get info about the focal region from the database
    info = get_adm1_info(focal_adm1)
    if info is None:
        raise ValueError(
            f"ADM1 code '{focal_adm1}' is not in the pericoupling database."
        )

    focal_country = info["iso_a3"]

    if colors is None:
        colors = DEFAULT_ADM1_COLORS

    if title is None:
        title = (
            f"ADM1 Metacoupling Map: {info['name']} ({info['country_name']})"
        )

    # Get neighbors from the database
    neighbor_codes = get_adm1_neighbors(focal_adm1)

    # Load ADM1 shapefile
    world = _get_adm1_geodataframe(shapefile=shapefile)

    # Build classification
    shapefile_codes = set(world["adm1_code"].dropna().unique())
    shapefile_countries: dict[str, str] = dict(
        zip(world["adm1_code"], world["iso_code"])
    )

    # Collect all countries that appear in the ADM1 database
    from metacouplingllm.knowledge.adm1_pericoupling import _adm1_country, _ensure_loaded

    _ensure_loaded()
    assert _adm1_country is not None
    db_countries = set(_adm1_country.values())

    # If mentioned_countries is provided, always include the focal country
    if mentioned_countries is not None:
        mentioned_countries = mentioned_countries | {focal_country}

    classification = _classify_adm1(
        focal_adm1,
        focal_country,
        neighbor_codes,
        shapefile_codes,
        shapefile_countries,
        db_countries,
        mentioned_countries=mentioned_countries,
        mentioned_adm1_codes=mentioned_adm1_codes,
    )
    adm1_detail_countries = _select_adm1_detail_countries(
        focal_country,
        neighbor_codes,
        shapefile_countries,
    )

    # Compute zoom bounds if requested
    zoom_bounds = None
    if zoom_to_focal:
        focal_mask = world["iso_code"] == focal_country
        if focal_mask.any():
            bounds = world[focal_mask].total_bounds  # [minx, miny, maxx, maxy]
            zoom_bounds = (bounds[0], bounds[1], bounds[2], bounds[3])

    return _render_adm1_map(
        world, classification, colors, title, figsize, db_countries,
        adm1_detail_countries=adm1_detail_countries,
        zoom_bounds=zoom_bounds,
        flows=flows,
        mentioned_countries=mentioned_countries,
        focal_adm1=focal_adm1,
        focal_country_iso=focal_country,
    )

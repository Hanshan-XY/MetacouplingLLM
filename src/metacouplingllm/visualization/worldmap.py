"""
World map rendering for metacoupling framework analysis.

Generates choropleth maps where countries are color-coded by their coupling
type (intracoupling, pericoupling, telecoupling) relative to a focal country.

Uses **geopandas** and **matplotlib** (optional dependencies).  Install
with ``pip install metacoupling[viz]``.
"""

from __future__ import annotations

import os
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from metacouplingllm.knowledge.countries import (
    ISO_ALPHA3_NAMES,
    get_country_name,
    resolve_country_code,
)
from metacouplingllm.knowledge.pericoupling import (
    PairCouplingType,
    get_pericoupled_neighbors,
    lookup_pericoupling,
)

if TYPE_CHECKING:
    import geopandas as gpd
    import matplotlib.figure
    import matplotlib.pyplot as plt

    from metacouplingllm.llm.parser import ParsedAnalysis


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
class CouplingColors:
    """Color scheme for metacoupling world maps.

    Override any field to customise the palette.
    """

    intracoupling: str = "#D4E79E"  # light yellow-green (focal country)
    pericoupling: str = "#4CAF50"   # bright green (adjacent)
    telecoupling: str = "#ADD8E6"   # light blue (distant)
    na: str = "#D3D3D3"            # light grey (unknown / not in DB)
    ocean: str = "#DCEEFB"          # light water blue (oceans + Caspian)
    lake: str = "#4A90D9"          # medium-dark blue (lakes overlay)
    border: str = "#888888"         # country border lines
    disputed: str = "#BFBFBF"      # darker grey for disputed territories
    disputed_outline: str = "#4D4D4D"  # darker outline so disputes stay visible
    disputed_hatch: str = "///"    # matplotlib hatch pattern


DEFAULT_COLORS = CouplingColors()


# ---------------------------------------------------------------------------
# World Bank boundary data management
# ---------------------------------------------------------------------------

_WORLD_BANK_ADM0_ENV_VAR = "METACOUPLING_ADM0_SHAPEFILE"
_WORLD_BANK_ADM0_DOWNLOAD_ENV_VAR = "METACOUPLING_ADM0_DOWNLOAD_URL"
_WORLD_BANK_NDLSA_ENV_VAR = "METACOUPLING_NDLSA_SHAPEFILE"
_WORLD_BANK_NDLSA_DOWNLOAD_ENV_VAR = "METACOUPLING_NDLSA_DOWNLOAD_URL"
_WORLD_BANK_ADM0_FILENAMES = (
    "World Bank Official Boundaries - Admin 0_all_layers.gpkg",
    "World Bank Official Boundaries - Admin 0.gpkg",
    "World.Bank.Official.Boundaries.-.Admin.0_all_layers.gpkg",
    "World.Bank.Official.Boundaries.-.Admin.0.gpkg",
)
_WORLD_BANK_NDLSA_FILENAMES = (
    "World Bank Official Boundaries - NDLSA.gpkg",
    "World.Bank.Official.Boundaries.-.NDLSA.gpkg",
)
_WORLD_BANK_ADM0_SHAPEFILE_NAME = "WB_countries_Admin0_10m.shp"
_WORLD_BANK_BOUNDARIES_RELEASE_BASE_URL = (
    "https://github.com/Hanshan-XY/wb-boundaries-data/releases/download/"
    "v2025-06-17"
)
_WORLD_BANK_ADM0_RELEASE_URL = (
    f"{_WORLD_BANK_BOUNDARIES_RELEASE_BASE_URL}/"
    "World.Bank.Official.Boundaries.-.Admin.0_all_layers.gpkg"
)
_WORLD_BANK_ADM0_FALLBACK_DOWNLOAD_URL = (
    "https://datacatalogfiles.worldbank.org/ddh-published/0038272/"
    "DR0046659/wb_countries_admin0_10m.zip"
)
_WORLD_BANK_NDLSA_RELEASE_URL = (
    f"{_WORLD_BANK_BOUNDARIES_RELEASE_BASE_URL}/"
    "World.Bank.Official.Boundaries.-.NDLSA.gpkg"
)
_WORLD_BANK_ADM0_DOWNLOAD_FILENAME = (
    "World.Bank.Official.Boundaries.-.Admin.0_all_layers.gpkg"
)
_WORLD_BANK_ADM0_FALLBACK_DOWNLOAD_FILENAME = "wb_countries_admin0_10m.zip"
_WORLD_BANK_NDLSA_DOWNLOAD_FILENAME = (
    "World.Bank.Official.Boundaries.-.NDLSA.gpkg"
)
_WORLD_BANK_NDLSA_STATUS = "Non-determined legal status area"

# Mapping from shapefile ISO codes to this package's codes.
# The pericoupling DB sometimes uses older or non-standard codes.
_ISO_CODE_FIXES: dict[str, str] = {
    "ROU": "ROM",  # Romania
    "COD": "ZAR",  # Democratic Republic of the Congo
    "TLS": "TMP",  # Timor-Leste / East Timor
    "SRB": "YUG",  # Serbia (package uses YUG for former Yugoslavia/Serbia)
}


def _get_shapefile_cache_dir() -> Path:
    """Return the directory for caching map boundary files.

    Respects the ``METACOUPLING_CACHE_DIR`` environment variable.
    Falls back to ``~/.cache/metacoupling``.
    """
    env = os.environ.get("METACOUPLING_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "metacoupling"


def _get_adm0_cache_dir() -> Path:
    """Return the cache directory for World Bank ADM0 downloads."""
    return _get_shapefile_cache_dir() / "wb_adm0"


def _get_ndlsa_cache_dir() -> Path:
    """Return the cache directory for World Bank NDLSA downloads."""
    return _get_shapefile_cache_dir() / "wb_ndlsa"


def _iter_world_bank_ndlsa_candidates() -> list[Path]:
    """Return likely local paths for a World Bank NDLSA GeoPackage."""
    env_path = os.environ.get(_WORLD_BANK_NDLSA_ENV_VAR)
    cache_dir = _get_shapefile_cache_dir()
    ndlsa_cache_dir = _get_ndlsa_cache_dir()
    cwd = Path.cwd()
    search_roots = (
        cwd,
        cwd / "data",
        cwd / "data" / "shapefiles",
        Path.home() / "Downloads",
        cache_dir,
        cache_dir / "world_bank",
        ndlsa_cache_dir,
    )

    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    for root in search_roots:
        for filename in _WORLD_BANK_NDLSA_FILENAMES:
            candidates.append(root / filename)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = os.path.normcase(str(candidate))
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def _find_cached_world_bank_ndlsa() -> Path | None:
    """Return a cached World Bank NDLSA path if available."""
    cache_dir = _get_ndlsa_cache_dir()
    if not cache_dir.exists():
        return None

    for filename in _WORLD_BANK_NDLSA_FILENAMES:
        candidate = cache_dir / filename
        if candidate.exists():
            return candidate

    generic_gpkg = next(cache_dir.rglob("*.gpkg"), None)
    if generic_gpkg is not None:
        return generic_gpkg

    return None


def _iter_world_bank_adm0_candidates() -> list[Path]:
    """Return likely local paths for a World Bank ADM0 GeoPackage."""
    env_path = os.environ.get(_WORLD_BANK_ADM0_ENV_VAR)
    cache_dir = _get_shapefile_cache_dir()
    adm0_cache_dir = _get_adm0_cache_dir()
    cwd = Path.cwd()
    search_roots = (
        cwd,
        cwd / "data",
        cwd / "data" / "shapefiles",
        Path.home() / "Downloads",
        cache_dir,
        cache_dir / "world_bank",
        adm0_cache_dir,
    )

    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    for root in search_roots:
        for filename in _WORLD_BANK_ADM0_FILENAMES:
            candidates.append(root / filename)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = os.path.normcase(str(candidate))
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def _find_cached_world_bank_adm0() -> Path | None:
    """Return a cached World Bank ADM0 basemap path if available."""
    cache_dir = _get_adm0_cache_dir()
    if not cache_dir.exists():
        return None

    for filename in _WORLD_BANK_ADM0_FILENAMES:
        candidate = cache_dir / filename
        if candidate.exists():
            return candidate

    shp_candidate = next(
        cache_dir.rglob(_WORLD_BANK_ADM0_SHAPEFILE_NAME),
        None,
    )
    if shp_candidate is not None:
        return shp_candidate

    generic_shp = next(cache_dir.rglob("*.shp"), None)
    if generic_shp is not None:
        return generic_shp

    generic_gpkg = next(cache_dir.rglob("*.gpkg"), None)
    if generic_gpkg is not None:
        return generic_gpkg

    return None


def _download_world_bank_ndlsa() -> Path | None:
    """Download and cache the default World Bank NDLSA GeoPackage."""
    cached = _find_cached_world_bank_ndlsa()
    if cached is not None:
        return cached

    cache_dir = _get_ndlsa_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    download_url = os.environ.get(
        _WORLD_BANK_NDLSA_DOWNLOAD_ENV_VAR,
        _WORLD_BANK_NDLSA_RELEASE_URL,
    )
    filename = Path(download_url).name or _WORLD_BANK_NDLSA_DOWNLOAD_FILENAME
    target_path = cache_dir / filename

    try:
        if not target_path.exists():
            urllib.request.urlretrieve(download_url, target_path)
        return target_path
    except Exception:
        return None


def _download_world_bank_adm0_basemap() -> Path | None:
    """Download and cache the default World Bank ADM0 basemap.

    The current first-run download target is the GitHub-hosted
    ``Admin 0_all_layers`` GeoPackage mirror. If that fails, this falls back
    to the official World Bank ``wb_countries_admin0_10m.zip`` resource.
    If all download attempts fail, ``None`` is returned so the caller can
    fall back gracefully.
    """
    cached = _find_cached_world_bank_adm0()
    if cached is not None:
        return cached

    cache_dir = _get_adm0_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    override_url = os.environ.get(_WORLD_BANK_ADM0_DOWNLOAD_ENV_VAR)
    sources: list[tuple[str, str]] = []
    if override_url:
        filename = Path(override_url).name or _WORLD_BANK_ADM0_DOWNLOAD_FILENAME
        sources.append((override_url, filename))
    else:
        sources.extend([
            (_WORLD_BANK_ADM0_RELEASE_URL, _WORLD_BANK_ADM0_DOWNLOAD_FILENAME),
            (
                _WORLD_BANK_ADM0_FALLBACK_DOWNLOAD_URL,
                _WORLD_BANK_ADM0_FALLBACK_DOWNLOAD_FILENAME,
            ),
        ])

    for download_url, filename in sources:
        target_path = cache_dir / filename
        try:
            if not target_path.exists():
                urllib.request.urlretrieve(download_url, target_path)

            if download_url.lower().endswith(".zip"):
                with zipfile.ZipFile(target_path, "r") as zf:
                    zf.extractall(cache_dir)

            cached = _find_cached_world_bank_adm0()
            if cached is not None:
                return cached
        except Exception:
            continue

    return None


def _resolve_world_basemap_path(
    *,
    custom_shapefile: str | Path | None = None,
    adm0_shapefile: str | Path | None = None,
) -> Path | None:
    """Resolve the preferred country-level basemap path.

    Resolution order:

    1. ``custom_shapefile`` for fully custom user-provided boundaries.
    2. ``adm0_shapefile`` for an explicit World Bank ADM0 file.
    3. A locally available World Bank ADM0 GeoPackage discovered in common
       directories such as ``Downloads`` or the metacoupling cache.
    4. ``None`` when no World Bank ADM0 file is available locally.
    """
    if custom_shapefile is not None:
        return Path(custom_shapefile).expanduser()

    if adm0_shapefile is not None:
        return Path(adm0_shapefile).expanduser()

    for candidate in _iter_world_bank_adm0_candidates():
        if candidate.exists():
            return candidate

    cached = _find_cached_world_bank_adm0()
    if cached is not None:
        return cached

    return None


def _extract_world_bank_disputed_rows(
    world: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame | None:
    """Extract disputed / indeterminate rows from a World Bank layer."""
    if "WB_STATUS" not in world.columns:
        return None

    disputed = world[
        world["WB_STATUS"] == _WORLD_BANK_NDLSA_STATUS
    ].copy()
    if len(disputed) == 0:
        return None
    return disputed


def _load_world_bank_ndlsa_geodataframe() -> gpd.GeoDataFrame | None:
    """Load World Bank NDLSA polygons from a local or downloaded GeoPackage."""
    import geopandas as _gpd

    source_path = None
    for candidate in _iter_world_bank_ndlsa_candidates():
        if candidate.exists():
            source_path = candidate
            break

    if source_path is None:
        source_path = _find_cached_world_bank_ndlsa()

    if source_path is None:
        source_path = _download_world_bank_ndlsa()

    if source_path is None or not source_path.exists():
        return None

    try:
        ndlsa = _gpd.read_file(str(source_path))
    except Exception:
        return None

    if "iso_code" not in ndlsa.columns:
        if "ISO_A3" in ndlsa.columns:
            ndlsa["iso_code"] = ndlsa["ISO_A3"]
        elif "ISO_A3_EH" in ndlsa.columns:
            ndlsa["iso_code"] = ndlsa["ISO_A3_EH"]

    if "iso_code" in ndlsa.columns:
        ndlsa["iso_code"] = ndlsa["iso_code"].replace(_ISO_CODE_FIXES)

    return ndlsa


def _get_world_geodataframe(
    custom_shapefile: str | Path | None = None,
    adm0_shapefile: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Load a world-countries GeoDataFrame.

    By default, this function prefers a locally available World Bank ADM0
    file (for example ``Admin 0_all_layers.gpkg`` in ``Downloads`` or the
    metacoupling cache). If none is found, it tries to auto-download the
    hosted ``Admin 0_all_layers`` mirror, then the official World Bank
    ``wb_countries_admin0_10m.zip`` basemap into the cache. If
    *custom_shapefile* or *adm0_shapefile* is provided, that file is loaded
    instead.

    Parameters
    ----------
    custom_shapefile:
        Path to a user-provided shapefile (``.shp``), GeoJSON, or
        GeoPackage.  The file must have a column named ``ISO_A3`` or
        ``iso_code`` with ISO alpha-3 country codes.
    adm0_shapefile:
        Path to a World Bank ADM0 GeoPackage to use for country maps.
        Ignored when *custom_shapefile* is provided.

    Returns
    -------
    geopandas.GeoDataFrame
        World countries with a normalised ``iso_code`` column that matches
        this package's ISO alpha-3 conventions.
    """
    import geopandas as _gpd

    source_path = _resolve_world_basemap_path(
        custom_shapefile=custom_shapefile,
        adm0_shapefile=adm0_shapefile,
    )

    if source_path is None and custom_shapefile is None and adm0_shapefile is None:
        source_path = _download_world_bank_adm0_basemap()

    if source_path is None or not source_path.exists():
        raise FileNotFoundError(
            "Country-level maps require World Bank ADM0 boundary data. "
            "metacoupling looks for a local 'Admin 0_all_layers.gpkg' and "
            "then tries the hosted GitHub mirror and official World Bank "
            "Admin 0 10m download."
        )

    world = _gpd.read_file(str(source_path))
    if "iso_code" not in world.columns:
        if "ISO_A3" in world.columns:
            world["iso_code"] = world["ISO_A3"]
        elif "ISO_A3_EH" in world.columns:
            world["iso_code"] = world["ISO_A3_EH"]
        else:
            raise ValueError(
                "Custom shapefile must contain an 'ISO_A3' or "
                "'iso_code' column with ISO alpha-3 country codes."
            )
    world["iso_code"] = world["iso_code"].replace(_ISO_CODE_FIXES)
    return world


def _get_disputed_territories_overlay(
    target_crs: object | None = None,
    base_world: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame | None:
    """Return disputed-area polygons for map overlays.

    The overlay is sourced only from World Bank data:

    1. ``base_world`` rows marked with ``WB_STATUS = Non-determined legal
       status area`` when available.
    2. A local or downloaded World Bank ``NDLSA.gpkg`` file.
    3. A local or downloaded World Bank ``Admin 0_all_layers.gpkg`` file
       when it contains NDLSA rows.
    """
    disputed = None

    if base_world is not None:
        disputed = _extract_world_bank_disputed_rows(base_world)

    if disputed is None:
        disputed = _load_world_bank_ndlsa_geodataframe()

    if disputed is None:
        try:
            adm0_world = _get_world_geodataframe()
        except Exception:
            adm0_world = None
        if adm0_world is not None:
            disputed = _extract_world_bank_disputed_rows(adm0_world)

    if disputed is None or len(disputed) == 0:
        return None

    if target_crs is not None and disputed.crs is not None:
        try:
            if disputed.crs != target_crs:
                disputed = disputed.to_crs(target_crs)
        except Exception:
            pass

    return disputed


def _plot_disputed_overlay(
    ax,
    disputed_gdf: gpd.GeoDataFrame | None,
    colors: CouplingColors,
    *,
    zorder: float,
) -> None:
    """Draw disputed polygons with a visible hatch/outline overlay."""
    if disputed_gdf is None or len(disputed_gdf) == 0:
        return

    disputed_gdf.plot(
        ax=ax,
        color=colors.disputed,
        edgecolor="none",
        alpha=0.22,
        zorder=zorder,
    )
    disputed_gdf.plot(
        ax=ax,
        facecolor="none",
        edgecolor=colors.disputed_outline,
        hatch=colors.disputed_hatch * 2,
        linewidth=0.7,
        zorder=zorder + 0.05,
    )


# ---------------------------------------------------------------------------
# Country classification
# ---------------------------------------------------------------------------


def _classify_countries(
    focal_code: str,
    pericoupled_codes: set[str],
    shapefile_codes: set[str],
    db_codes: set[str],
    mentioned_codes: set[str] | None = None,
) -> dict[str, str]:
    """Classify every country in the shapefile by coupling type.

    Parameters
    ----------
    focal_code:
        ISO code of the focal (intracoupling) country.
    pericoupled_codes:
        Codes of countries pericoupled with the focal country.
    shapefile_codes:
        All ISO codes present in the shapefile.
    db_codes:
        All ISO codes known to this package (from ``ISO_ALPHA3_NAMES``).
    mentioned_codes:
        When provided, only countries in this set (or that are pericoupled
        with the focal country) will be classified as telecoupling.
        Countries not in this set are classified as ``"na"`` (grey).
        When ``None``, all DB countries are coloured (legacy behaviour).

    Returns
    -------
    ``{iso_code: "intracoupling"|"pericoupling"|"telecoupling"|"na"}``
    """
    classification: dict[str, str] = {}
    for code in shapefile_codes:
        if code == focal_code:
            classification[code] = "intracoupling"
        elif code in pericoupled_codes:
            classification[code] = "pericoupling"
        elif mentioned_codes is not None:
            # Analysis-driven mode: only color countries mentioned in analysis
            if code in mentioned_codes:
                classification[code] = "telecoupling"
            else:
                classification[code] = "na"
        elif code in db_codes:
            classification[code] = "telecoupling"
        else:
            classification[code] = "na"
    return classification


# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------


_FLOW_ARROW_COLORS: dict[str, str] = {
    "matter": "#E53935",       # red
    "material": "#E53935",     # alias
    "capital": "#1E88E5",      # blue
    "financial": "#1E88E5",    # alias
    "information": "#FDD835",  # yellow
    "energy": "#FB8C00",       # orange
    "people": "#8E24AA",       # purple
    "organisms": "#43A047",    # green
}

_FLOW_ARROW_DEFAULT_COLOR = "#555555"


def _get_country_centroid(
    world: gpd.GeoDataFrame, iso_code: str,
) -> tuple[float, float] | None:
    """Return the (x, y) centroid for a country by ISO code.

    Some countries (e.g., GBR) have multiple rows in the shapefile —
    one for the mainland and one for overseas territories (e.g., British
    Sovereign Base Areas on Cyprus). Naively using the first row's
    representative point would place the centroid on Cyprus instead of
    Great Britain. We avoid this by merging all matching geometries
    with ``unary_union`` and then using the representative point of the
    **largest polygon** in the result.
    """
    rows = world.loc[world["iso_code"] == iso_code]
    if rows.empty:
        return None
    # union_all() is the non-deprecated replacement for unary_union
    # (geopandas >= 1.0). Fall back to unary_union for older versions.
    if hasattr(rows.geometry, "union_all"):
        combined = rows.geometry.union_all()
    else:
        combined = rows.geometry.unary_union
    if combined.geom_type == "MultiPolygon":
        # Pick the largest polygon — this is the mainland, not overseas
        # territories or exclaves.
        largest = max(combined.geoms, key=lambda p: p.area)
        pt = largest.representative_point()
    else:
        pt = combined.representative_point()
    return (pt.x, pt.y)


def _resolve_flow_endpoints(
    flow: dict[str, str],
    world: gpd.GeoDataFrame,
) -> tuple[
    tuple[float, float] | None,
    tuple[float, float] | None,
    bool,
]:
    """Resolve source and target centroids from a flow's direction string.

    Returns ``(source_xy, target_xy, is_bidirectional)``.
    """
    direction = flow.get("direction", "")
    if not direction:
        return None, None, False

    is_bidir = "bidirectional" in direction.lower() or "↔" in direction

    # Extract country names from the direction string
    # Patterns: "Brazil → China", "Bidirectional (Brazil ↔ China)"
    # Remove "Bidirectional" wrapper
    cleaned = re.sub(r"[Bb]idirectional\s*\(?", "", direction)
    cleaned = cleaned.rstrip(")")

    # Split on arrows
    parts = re.split(r"\s*(?:→|->|=>|↔|<->|<=>)\s*", cleaned)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) < 2:
        return None, None, is_bidir

    src_name = parts[0]
    tgt_name = parts[1]

    # Handle multi-target: "Brazil → China (and other importers)"
    tgt_name = re.sub(r"\s*\(.*\)", "", tgt_name).strip()

    src_code = resolve_country_code(src_name)
    tgt_code = resolve_country_code(tgt_name)

    if not src_code or not tgt_code:
        return None, None, is_bidir

    src_xy = _get_country_centroid(world, src_code)
    tgt_xy = _get_country_centroid(world, tgt_code)

    return src_xy, tgt_xy, is_bidir


def _draw_flow_arrows(
    ax: object,
    world: gpd.GeoDataFrame,
    flows: list[dict[str, str]],
) -> list:
    """Draw curved flow arrows on the map axes.

    Returns legend handles for flow categories.
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    drawn_categories: dict[str, str] = {}  # category → color

    # Group flows by (source, target) pair so overlapping arrows between
    # the same country pair are spread apart with increasing curvature.
    pair_counts: dict[tuple, int] = {}

    for flow in flows:
        src_xy, tgt_xy, is_bidir = _resolve_flow_endpoints(flow, world)
        if src_xy is None or tgt_xy is None:
            continue

        category = flow.get("category", "").lower()
        color = _FLOW_ARROW_COLORS.get(category, _FLOW_ARROW_DEFAULT_COLOR)

        # Track for legend
        cat_label = category.title() if category else "Flow"
        if cat_label not in drawn_categories:
            drawn_categories[cat_label] = color

        # Compute offset for arrows between the same country pair.
        # Use a canonical key (sorted endpoints) so A→B and B→A share
        # the same offset sequence but curve on opposite sides.
        pair_key = (
            min(src_xy, tgt_xy),
            max(src_xy, tgt_xy),
        )
        slot = pair_counts.get(pair_key, 0)
        pair_counts[pair_key] = slot + 1

        # Alternate positive/negative curvature so forward and reverse
        # arrows separate visually.  Base radius 0.15, each additional
        # arrow in the same pair adds 0.10 curvature.
        base_rad = 0.15 + slot * 0.10
        # Reverse arrows (tgt_xy < src_xy canonically) get negative rad
        # to curve on the opposite side.
        if src_xy > tgt_xy:
            base_rad = -base_rad

        # Draw curved arrow
        arrow = FancyArrowPatch(
            posA=src_xy,
            posB=tgt_xy,
            arrowstyle="->" if not is_bidir else "<->",
            connectionstyle=f"arc3,rad={base_rad}",
            color=color,
            linewidth=2.0,
            mutation_scale=15,
            zorder=5,
            alpha=0.8,
        )
        ax.add_patch(arrow)

    # Build flow legend handles
    handles = []
    for cat_label, color in drawn_categories.items():
        handles.append(
            mpatches.FancyArrowPatch(
                (0, 0), (1, 0),
                arrowstyle="->",
                color=color,
                linewidth=2,
            )
            if False  # placeholder — use Line2D instead for legend
            else mpatches.Patch(facecolor=color, edgecolor=color, label=f"{cat_label} Flow")
        )
    return handles


def _render_map(
    world: gpd.GeoDataFrame,
    classification: dict[str, str],
    colors: CouplingColors,
    title: str,
    figsize: tuple[float, float],
    flows: list[dict[str, str]] | None = None,
) -> matplotlib.figure.Figure:
    """Render the classified world map.

    Parameters
    ----------
    world:
        World GeoDataFrame with ``iso_code`` column.
    classification:
        Mapping of ISO codes to coupling type.
    colors:
        Color scheme.
    title:
        Map title.
    figsize:
        Figure size.
    flows:
        Optional list of parsed flow dicts.  When provided, curved arrows
        are drawn on the map showing the direction of each flow.

    Returns a :class:`matplotlib.figure.Figure`.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as _plt

    fig, ax = _plt.subplots(1, 1, figsize=figsize)

    # Assign coupling type and colour to each geometry
    world = world.copy()
    world["coupling"] = (
        world["iso_code"].map(classification).fillna("na")
    )

    color_map = {
        "intracoupling": colors.intracoupling,
        "pericoupling": colors.pericoupling,
        "telecoupling": colors.telecoupling,
        "na": colors.na,
    }
    world["_color"] = world["coupling"].map(color_map)

    # Plot all countries
    world.plot(
        ax=ax,
        color=world["_color"],
        edgecolor=colors.border,
        linewidth=0.5,
    )

    # Overlay disputed territories with hatching using the raw disputed
    # polygons so they remain visible even when the main world layer has
    # been dissolved or merged.
    disputed_gdf = _get_disputed_territories_overlay(world.crs, world)
    _plot_disputed_overlay(ax, disputed_gdf, colors, zorder=3)

    # Draw flow arrows if flows are provided
    flow_handles: list = []
    if flows:
        flow_handles = _draw_flow_arrows(ax, world, flows)

    # Legend — coupling categories
    legend_items = [
        mpatches.Patch(
            facecolor=colors.na,
            edgecolor="black",
            label="NA",
        ),
        mpatches.Patch(
            facecolor=colors.intracoupling,
            edgecolor="black",
            label="Intracoupling Countries",
        ),
        mpatches.Patch(
            facecolor=colors.pericoupling,
            edgecolor="black",
            label="Pericoupling Countries",
        ),
        mpatches.Patch(
            facecolor=colors.telecoupling,
            edgecolor="black",
            label="Telecoupling Countries",
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
        fontsize=10,
        title="Categories",
        title_fontsize=11,
        frameon=True,
        fancybox=True,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_axis_off()
    ax.set_facecolor(colors.ocean)
    fig.patch.set_facecolor(colors.ocean)
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Public API — Function 1: Database-only map
# ---------------------------------------------------------------------------


def plot_focal_country_map(
    focal_country: str,
    *,
    title: str | None = None,
    colors: CouplingColors | None = None,
    figsize: tuple[float, float] = (15, 8),
    custom_shapefile: str | Path | None = None,
    adm0_shapefile: str | Path | None = None,
    mentioned_countries: set[str] | None = None,
) -> matplotlib.figure.Figure:
    """Generate a world map colored by coupling type relative to a focal country.

    All countries in the pericoupling database are classified:

    - **Intracoupling** (light yellow-green): the focal country itself.
    - **Pericoupling** (bright green): countries geographically adjacent to
      the focal country according to the database.
    - **Telecoupling** (light blue): countries that are in the database but
      are geographically distant from the focal country.
    - **NA** (grey): countries not in the database.

    Parameters
    ----------
    focal_country:
        Country name or ISO alpha-3 code (e.g., ``"Mexico"`` or ``"MEX"``).
    title:
        Custom map title.  Defaults to ``"Metacoupling Map: {Country Name}"``.
    colors:
        Custom :class:`CouplingColors` instance.  Defaults to the standard
        metacoupling palette.
    figsize:
        Figure size in inches ``(width, height)``.
    custom_shapefile:
        Path to a user-provided shapefile (``.shp``), GeoJSON, or
        GeoPackage to use instead of the default World Bank ADM0 data
        source. The file must contain an ``ISO_A3`` or ``iso_code``
        column. When provided, this takes precedence over
        *adm0_shapefile*.
    adm0_shapefile:
        Path to a World Bank ADM0 GeoPackage for country maps. If omitted,
        metacoupling tries to discover a local ``Admin 0_all_layers.gpkg``
        automatically, then auto-downloads the hosted
        ``Admin 0_all_layers`` mirror and official World Bank Admin 0 10m
        basemap if needed.
    mentioned_countries:
        When provided (typically from a parsed LLM analysis), only
        countries in this set are coloured as telecoupling.
        All other distant countries are grey.  The focal country is always
        included automatically.  When ``None``, the legacy behaviour is
        used (all DB countries are coloured).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered map.  Call ``fig.savefig("map.png")`` to save.

    Raises
    ------
    ImportError
        If geopandas or matplotlib are not installed.
    ValueError
        If *focal_country* cannot be resolved to an ISO code.

    Examples
    --------
    >>> fig = plot_focal_country_map("Mexico")
    >>> fig.savefig("mexico_metacoupling.png", dpi=150, bbox_inches="tight")

    >>> # Use a custom base map
    >>> fig = plot_focal_country_map("China", custom_shapefile="my_countries.shp")
    """
    _check_dependencies()

    # Resolve focal country
    focal_code = resolve_country_code(focal_country)
    if focal_code is None:
        raise ValueError(
            f"Could not resolve '{focal_country}' to an ISO country code."
        )

    if colors is None:
        colors = DEFAULT_COLORS

    if title is None:
        title = f"Metacoupling Map: {get_country_name(focal_code)}"

    # Get pericoupled neighbors from the database
    pericoupled = get_pericoupled_neighbors(focal_code)

    # Load world shapefile
    world = _get_world_geodataframe(
        custom_shapefile=custom_shapefile,
        adm0_shapefile=adm0_shapefile,
    )

    # Classify every country
    shapefile_codes = set(world["iso_code"].dropna().unique())
    db_codes = set(ISO_ALPHA3_NAMES.keys())
    
    # If mentioned_countries is provided, always include the focal country
    if mentioned_countries is not None:
        mentioned_countries = mentioned_countries | {focal_code}
    
    classification = _classify_countries(
        focal_code, pericoupled, shapefile_codes, db_codes,
        mentioned_codes=mentioned_countries
    )

    return _render_map(world, classification, colors, title, figsize, flows=None)


# ---------------------------------------------------------------------------
# Public API — Function 2: Analysis-based map
# ---------------------------------------------------------------------------


def _extract_countries_from_analysis(
    parsed: ParsedAnalysis,
) -> dict[str, str | None]:
    """Extract country names from a parsed analysis and resolve their codes.

    Returns ``{role: iso_code_or_None}`` for focal, adjacent, sending,
    receiving, and spillover. Only returns one code per role (the first
    resolved).
    """
    result: dict[str, str | None] = {}
    role_texts: dict[str, list[str]] = {
        "focal": [],
        "adjacent": [],
        "sending": [],
        "receiving": [],
        "spillover": [],
    }
    for _, section in parsed.iter_coupling_sections():
        for entry in section.systems:
            role = entry.get("role", "").strip().lower()
            scope = entry.get("system_scope", "").strip().lower()
            map_role = "adjacent" if scope == "adjacent" else role
            if map_role not in role_texts:
                continue
            for key in ("name", "geographic_scope"):
                value = entry.get(key, "")
                if value:
                    role_texts[map_role].append(value)

    for role in ("focal", "adjacent", "sending", "receiving", "spillover"):
        # Try each text for a country match
        code: str | None = None
        for text in role_texts[role]:
            code = resolve_country_code(text)
            if code:
                break
            # Scan sub-chunks
            for chunk in re.split(r"[,;/()]+", text):
                chunk = chunk.strip()
                if chunk:
                    code = resolve_country_code(chunk)
                    if code:
                        break
            if code:
                break
        result[role] = code

    return result


def _extract_all_analysis_countries(
    parsed: ParsedAnalysis,
) -> dict[str, set[str]]:
    """Extract **all** country codes mentioned in the analysis per role.

    Unlike :func:`_extract_countries_from_analysis` (one code per role),
    this function returns *all* codes found in each role's text, as well
    as countries mentioned in the coupling classification and flows.

    Returns
    -------
    ``{role: {iso_codes}}`` where role is ``"sending"``, ``"receiving"``,
    ``"spillover"``, ``"focal"``, ``"adjacent"``, or ``"other"`` (for
    countries found outside role-specific system entries).
    """
    role_codes: dict[str, set[str]] = {
        "focal": set(),
        "adjacent": set(),
        "sending": set(),
        "receiving": set(),
        "spillover": set(),
        "other": set(),
    }

    # --- Systems: scan all text fields for country names ---
    for _, section in parsed.iter_coupling_sections():
        for entry in section.systems:
            role = entry.get("role", "").strip().lower()
            scope = entry.get("system_scope", "").strip().lower()
            map_role = "adjacent" if scope == "adjacent" else role
            if map_role not in role_codes:
                map_role = "other"
            for key, value in entry.items():
                if key in {"role", "system_scope"}:
                    continue
                if isinstance(value, str) and value:
                    _scan_text_for_countries(value, role_codes[map_role])

    # --- Flows: direction strings often contain country names ---
    for flow in parsed.iter_flow_entries():
        direction = flow.get("direction", "")
        if direction:
            _scan_text_for_countries(direction, role_codes["other"])
        desc = flow.get("description", "")
        if desc:
            _scan_text_for_countries(desc, role_codes["other"])

    # --- Coupling classification ---
    if parsed.coupling_classification:
        _scan_text_for_countries(
            parsed.coupling_classification, role_codes["other"],
        )

    # --- Cross-coupling interactions / research gaps / grouped content ---
    for item in parsed.cross_coupling_interactions:
        if item:
            _scan_text_for_countries(item, role_codes["other"])
    for gap in parsed.research_gaps:
        if gap:
            _scan_text_for_countries(gap, role_codes["other"])
    for kind in ("causes", "effects"):
        for _section_name, _category, item in parsed.iter_category_items(kind):
            if item:
                _scan_text_for_countries(item, role_codes["other"])

    return role_codes


def _collect_analysis_map_country_codes(
    all_role_codes: dict[str, set[str]],
    focal_code: str,
) -> set[str]:
    """Collect all country codes that should be colored on an analysis map.

    Countries mentioned in sending, receiving, spillover, or other structured
    analysis sections are all retained so the figure matches the narrative more
    closely. The focal country is always included.
    """
    mentioned_codes: set[str] = {focal_code}
    for role in ("focal", "adjacent", "sending", "receiving", "spillover", "other"):
        mentioned_codes.update(all_role_codes.get(role, set()))
    return mentioned_codes


def _scan_text_for_countries(text: str, codes: set[str]) -> None:
    """Scan a text string for country names and add resolved codes to *codes*."""
    # Try the full text first
    code = resolve_country_code(text)
    if code:
        codes.add(code)

    # Split on common delimiters including arrows
    for chunk in re.split(r"[,;/()\[\]]+|→|->|=>|↔|<->|<=>", text):
        chunk = chunk.strip()
        if not chunk or len(chunk) < 3:
            continue
        code = resolve_country_code(chunk)
        if code:
            codes.add(code)

    # Also try individual multi-word segments that might be country names
    # e.g., "Argentina, USA, and other major..." → "Argentina", "USA"
    for chunk in re.split(r"\s+and\s+|\s+or\s+", text):
        chunk = chunk.strip().rstrip(".,;:")
        if chunk:
            code = resolve_country_code(chunk)
            if code:
                codes.add(code)


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
) -> matplotlib.figure.Figure:
    """Generate a world map from a parsed LLM analysis result.

    Extracts countries from the systems identified by the LLM, then colors
    the map based on their relationship to the focal country (determined
    by *focal_role*).

    Parameters
    ----------
    parsed_analysis:
        A :class:`~metacouplingllm.llm.parser.ParsedAnalysis` object, typically
        obtained from ``AnalysisResult.parsed``.
    focal_role:
        Which system role to treat as the focal country for coloring.
        Defaults to ``"sending"``.
    title:
        Custom map title.  Defaults to ``"Metacoupling Analysis Map"``.
    colors:
        Custom :class:`CouplingColors` instance.
    figsize:
        Figure size in inches.
    custom_shapefile:
        Path to a user-provided shapefile (``.shp``), GeoJSON, or
        GeoPackage to use instead of the default World Bank ADM0 data
        source. The file must contain an ``ISO_A3`` or ``iso_code``
        column. When provided, this takes precedence over
        *adm0_shapefile*.
    adm0_shapefile:
        Path to a World Bank ADM0 GeoPackage for country maps. If omitted,
        metacoupling tries to discover a local ``Admin 0_all_layers.gpkg``
        automatically, then auto-downloads the hosted
        ``Admin 0_all_layers`` mirror and official World Bank Admin 0 10m
        basemap if needed.
    extra_mentioned_countries:
        Optional validated country codes to color in addition to those
        extracted from the parsed analysis.
    flows:
        Optional pre-resolved flow list for arrow rendering. When omitted,
        raw ``parsed_analysis.flows`` are used.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ImportError
        If geopandas or matplotlib are not installed.
    ValueError
        If no countries can be extracted from the analysis.

    Examples
    --------
    >>> result = advisor.analyze("Avocado trade between Mexico, US, Canada")
    >>> fig = plot_analysis_map(result.parsed)
    >>> fig.savefig("avocado_map.png", dpi=150, bbox_inches="tight")
    """
    _check_dependencies()

    if colors is None:
        colors = DEFAULT_COLORS

    # ------------------------------------------------------------------
    # When structured overrides are provided (from second LLM call),
    # skip the regex-based country extraction entirely.
    # ------------------------------------------------------------------
    if focal_code_override and mentioned_countries_override is not None:
        focal_code = focal_code_override
        mentioned_codes = set(mentioned_countries_override)
        if title is None:
            title = (
                f"Metacoupling Analysis Map: "
                f"{get_country_name(focal_code) or focal_code}"
            )
    else:
        # Legacy path: regex extraction from analysis text
        all_role_codes = _extract_all_analysis_countries(parsed_analysis)

        focal_codes = all_role_codes.get(focal_role, set())
        if not focal_codes and focal_role == "sending":
            focal_codes = all_role_codes.get("focal", set())
        focal_code: str | None = None

        if focal_codes:
            focal_code = sorted(focal_codes)[0]
        else:
            single = _extract_countries_from_analysis(parsed_analysis)
            focal_code = single.get(focal_role)
            if focal_code is None and focal_role == "sending":
                focal_code = single.get("focal")

        if focal_code is None:
            for flow in parsed_analysis.iter_flow_entries():
                direction = flow.get("direction", "")
                if not direction:
                    continue
                parts = re.split(r"→|->|=>|↔|<->|<=>", direction)
                if parts:
                    src = re.sub(r"\([^)]*\)", "", parts[0]).strip()
                    code = resolve_country_code(src)
                    if code:
                        focal_code = code
                        break

        if focal_code is None:
            other_codes = all_role_codes.get("other", set())
            if other_codes:
                focal_code = sorted(other_codes)[0]

        if focal_code is None:
            raise ValueError(
                f"Could not resolve a country from the '{focal_role}' "
                f"system. Try a different focal_role or use "
                f"plot_focal_country_map() with an explicit country name."
            )

        if title is None:
            title = (
                f"Metacoupling Analysis Map: "
                f"{get_country_name(focal_code)}"
            )

        mentioned_codes = _collect_analysis_map_country_codes(
            all_role_codes,
            focal_code,
        )
        if extra_mentioned_countries:
            mentioned_codes.update(extra_mentioned_countries)

    # Load world shapefile
    world = _get_world_geodataframe(
        custom_shapefile=custom_shapefile,
        adm0_shapefile=adm0_shapefile,
    )
    shapefile_codes = set(world["iso_code"].dropna().unique())

    # Classify ONLY mentioned countries; everything else is "na"
    classification: dict[str, str] = {}
    for code in shapefile_codes:
        if code == focal_code:
            classification[code] = "intracoupling"
        elif code in mentioned_codes:
            # Determine relationship with focal country from DB
            result = lookup_pericoupling(focal_code, code)
            if result.pair_type == PairCouplingType.PERICOUPLED:
                classification[code] = "pericoupling"
            else:
                classification[code] = "telecoupling"
        else:
            classification[code] = "na"

    # Extract flows from the parsed analysis for arrow rendering
    if flows is None:
        parsed_flows = list(parsed_analysis.iter_flow_entries())
        flows = parsed_flows if parsed_flows else None

    return _render_map(world, classification, colors, title, figsize, flows=flows)

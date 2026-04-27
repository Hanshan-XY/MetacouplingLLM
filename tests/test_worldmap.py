"""Tests for visualization/worldmap.py — metacoupling world map generation."""

from __future__ import annotations

import pytest

# Check if geopandas + matplotlib are available
_viz_available = True
try:
    import geopandas  # noqa: F401
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — avoids tkinter errors
    import matplotlib.pyplot as plt
except ImportError:
    _viz_available = False

skip_no_viz = pytest.mark.skipif(
    not _viz_available,
    reason="geopandas/matplotlib not installed",
)


# ---------------------------------------------------------------------------
# Tests that do NOT require geopandas/matplotlib
# ---------------------------------------------------------------------------


class TestGetPericoupledNeighbors:
    """Test the new get_pericoupled_neighbors() in pericoupling.py."""

    def test_mexico_neighbors_include_usa_and_gtm(self):
        from metacouplingllm.knowledge.pericoupling import get_pericoupled_neighbors

        neighbors = get_pericoupled_neighbors("MEX")
        assert "USA" in neighbors
        assert "GTM" in neighbors
        assert "BLZ" in neighbors  # Belize borders Mexico

    def test_china_has_many_neighbors(self):
        from metacouplingllm.knowledge.pericoupling import get_pericoupled_neighbors

        neighbors = get_pericoupled_neighbors("CHN")
        assert "RUS" in neighbors
        assert "IND" in neighbors
        assert len(neighbors) >= 10  # China borders many countries

    def test_by_name(self):
        from metacouplingllm.knowledge.pericoupling import get_pericoupled_neighbors

        neighbors = get_pericoupled_neighbors("Mexico")
        assert "USA" in neighbors

    def test_unknown_returns_empty(self):
        from metacouplingllm.knowledge.pericoupling import get_pericoupled_neighbors

        assert get_pericoupled_neighbors("Atlantis") == set()

    def test_returns_set_of_strings(self):
        from metacouplingllm.knowledge.pericoupling import get_pericoupled_neighbors

        neighbors = get_pericoupled_neighbors("USA")
        assert isinstance(neighbors, set)
        for code in neighbors:
            assert isinstance(code, str)
            assert len(code) == 3


class TestClassifyCountries:
    """Test the _classify_countries logic (no geopandas needed)."""

    def test_focal_is_intracoupling(self):
        from metacouplingllm.visualization.worldmap import _classify_countries

        result = _classify_countries(
            "MEX",
            {"USA", "GTM"},
            {"MEX", "USA", "GTM", "BRA", "XYZ"},
            {"MEX", "USA", "GTM", "BRA"},
        )
        assert result["MEX"] == "intracoupling"

    def test_pericoupled_is_pericoupling(self):
        from metacouplingllm.visualization.worldmap import _classify_countries

        result = _classify_countries(
            "MEX",
            {"USA", "GTM"},
            {"MEX", "USA", "GTM", "BRA"},
            {"MEX", "USA", "GTM", "BRA"},
        )
        assert result["USA"] == "pericoupling"
        assert result["GTM"] == "pericoupling"

    def test_distant_is_telecoupling(self):
        from metacouplingllm.visualization.worldmap import _classify_countries

        result = _classify_countries(
            "MEX",
            {"USA", "GTM"},
            {"MEX", "USA", "GTM", "BRA"},
            {"MEX", "USA", "GTM", "BRA"},
        )
        assert result["BRA"] == "telecoupling"

    def test_unknown_is_na(self):
        from metacouplingllm.visualization.worldmap import _classify_countries

        result = _classify_countries(
            "MEX",
            {"USA"},
            {"MEX", "USA", "XYZ"},
            {"MEX", "USA"},
        )
        assert result["XYZ"] == "na"


class TestCouplingColors:
    """Test the CouplingColors dataclass."""

    def test_default_colors(self):
        from metacouplingllm.visualization.worldmap import DEFAULT_COLORS

        assert DEFAULT_COLORS.intracoupling.startswith("#")
        assert DEFAULT_COLORS.pericoupling.startswith("#")
        assert DEFAULT_COLORS.telecoupling.startswith("#")
        assert DEFAULT_COLORS.na.startswith("#")
        assert DEFAULT_COLORS.ocean == "#DCEEFB"
        assert DEFAULT_COLORS.lake == "#4A90D9"
        assert DEFAULT_COLORS.disputed == "#BFBFBF"
        assert DEFAULT_COLORS.disputed_outline == "#4D4D4D"
        assert DEFAULT_COLORS.disputed_hatch == "///"

    def test_custom_colors(self):
        from metacouplingllm.visualization.worldmap import CouplingColors

        custom = CouplingColors(intracoupling="#FF0000", pericoupling="#00FF00")
        assert custom.intracoupling == "#FF0000"
        assert custom.pericoupling == "#00FF00"
        # Others should still have defaults
        assert custom.telecoupling == "#ADD8E6"
        assert custom.lake == "#4A90D9"
        assert custom.disputed == "#BFBFBF"
        assert custom.disputed_outline == "#4D4D4D"


class TestWorldBasemapResolution:
    """Tests for country basemap source selection."""

    def test_custom_shapefile_takes_priority(self):
        from pathlib import Path

        from metacouplingllm.visualization.worldmap import _resolve_world_basemap_path

        resolved = _resolve_world_basemap_path(
            custom_shapefile=Path("custom.gpkg"),
            adm0_shapefile=Path("adm0.gpkg"),
        )

        assert resolved == Path("custom.gpkg")

    def test_explicit_adm0_shapefile_is_used(self):
        from pathlib import Path

        from metacouplingllm.visualization.worldmap import _resolve_world_basemap_path

        resolved = _resolve_world_basemap_path(
            adm0_shapefile=Path("adm0.gpkg"),
        )

        assert resolved == Path("adm0.gpkg")

    def test_auto_discovers_world_bank_all_layers_in_downloads(
        self, monkeypatch, tmp_path
    ):
        from pathlib import Path

        from metacouplingllm.visualization.worldmap import _resolve_world_basemap_path

        fake_home = tmp_path / "home"
        downloads = fake_home / "Downloads"
        downloads.mkdir(parents=True)
        expected = downloads / "World Bank Official Boundaries - Admin 0_all_layers.gpkg"
        expected.write_text("placeholder", encoding="utf-8")

        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap.Path.home",
            lambda: fake_home,
        )
        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap.Path.cwd",
            lambda: Path(tmp_path),
        )
        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap._get_shapefile_cache_dir",
            lambda: tmp_path / "cache",
        )

        resolved = _resolve_world_basemap_path()

        assert resolved == expected

    def test_uses_cached_downloaded_shapefile(self, monkeypatch, tmp_path):
        from pathlib import Path

        from metacouplingllm.visualization.worldmap import _resolve_world_basemap_path

        cache_dir = tmp_path / "cache" / "wb_adm0" / "WB_countries_Admin0_10m"
        cache_dir.mkdir(parents=True)
        expected = cache_dir / "WB_countries_Admin0_10m.shp"
        expected.write_text("placeholder", encoding="utf-8")

        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap.Path.home",
            lambda: tmp_path / "home",
        )
        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap.Path.cwd",
            lambda: Path(tmp_path),
        )
        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap._get_shapefile_cache_dir",
            lambda: tmp_path / "cache",
        )

        resolved = _resolve_world_basemap_path()

        assert resolved == expected

    def test_download_world_bank_adm0_basemap_extracts_zip(
        self, monkeypatch, tmp_path
    ):
        import zipfile
        from pathlib import Path

        from metacouplingllm.visualization.worldmap import (
            _download_world_bank_adm0_basemap,
        )

        source_zip = tmp_path / "source.zip"
        with zipfile.ZipFile(source_zip, "w") as zf:
            zf.writestr(
                "WB_countries_Admin0_10m/WB_countries_Admin0_10m.shp",
                "placeholder",
            )

        def fake_urlretrieve(url, dest):
            Path(dest).write_bytes(source_zip.read_bytes())
            return str(dest), None

        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap._get_shapefile_cache_dir",
            lambda: tmp_path / "cache",
        )
        monkeypatch.setenv(
            "METACOUPLING_ADM0_DOWNLOAD_URL",
            "https://example.com/wb_adm0.zip",
        )
        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap.urllib.request.urlretrieve",
            fake_urlretrieve,
        )

        resolved = _download_world_bank_adm0_basemap()

        assert resolved is not None
        assert resolved.name == "WB_countries_Admin0_10m.shp"
        assert resolved.exists()

    def test_download_world_bank_adm0_basemap_saves_direct_gpkg(
        self, monkeypatch, tmp_path
    ):
        from pathlib import Path

        from metacouplingllm.visualization.worldmap import (
            _download_world_bank_adm0_basemap,
        )

        def fake_urlretrieve(url, dest):
            Path(dest).write_text("placeholder", encoding="utf-8")
            return str(dest), None

        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap._get_shapefile_cache_dir",
            lambda: tmp_path / "cache",
        )
        monkeypatch.setenv(
            "METACOUPLING_ADM0_DOWNLOAD_URL",
            "https://example.com/World.Bank.Official.Boundaries.-.Admin.0_all_layers.gpkg",
        )
        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap.urllib.request.urlretrieve",
            fake_urlretrieve,
        )

        resolved = _download_world_bank_adm0_basemap()

        assert resolved is not None
        assert (
            resolved.name
            == "World.Bank.Official.Boundaries.-.Admin.0_all_layers.gpkg"
        )
        assert resolved.exists()


@skip_no_viz
class TestDisputedOverlay:
    """Test disputed overlay extraction from World Bank boundary layers."""

    def test_world_bank_all_layers_overlay_is_preferred(self):
        import geopandas as gpd
        from shapely.geometry import Polygon

        from metacouplingllm.visualization.worldmap import (
            _get_disputed_territories_overlay,
        )

        base_world = gpd.GeoDataFrame(
            {
                "NAM_0": ["United States", "Western Sahara"],
                "WB_STATUS": [
                    "Member country",
                    "Non-determined legal status area",
                ],
                "iso_code": ["USA", "ESH"],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        overlay = _get_disputed_territories_overlay(
            "EPSG:4326",
            base_world=base_world,
        )

        assert overlay is not None
        assert list(overlay["NAM_0"]) == ["Western Sahara"]

    def test_ndlsa_layer_is_used_when_base_world_lacks_status(self, monkeypatch):
        import geopandas as gpd
        from shapely.geometry import Polygon

        from metacouplingllm.visualization.worldmap import (
            _get_disputed_territories_overlay,
        )

        raw = gpd.GeoDataFrame(
            {
                "NAM_0": ["Western Sahara", "Jammu and Kashmir"],
                "WB_STATUS": [
                    "Non-determined legal status area",
                    "Non-determined legal status area",
                ],
                "ISO_A3": ["ESH", None],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap._load_world_bank_ndlsa_geodataframe",
            lambda: raw,
        )
        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap._get_world_geodataframe",
            lambda: None,
        )

        overlay = _get_disputed_territories_overlay("EPSG:4326")

        assert overlay is not None
        assert set(overlay["NAM_0"]) == {
            "Western Sahara",
            "Jammu and Kashmir",
        }

    def test_adm0_all_layers_fallback_is_used_when_ndlsa_missing(
        self, monkeypatch
    ):
        import geopandas as gpd
        from shapely.geometry import Polygon

        from metacouplingllm.visualization.worldmap import (
            _get_disputed_territories_overlay,
        )

        raw = gpd.GeoDataFrame(
            {
                "NAM_0": ["United States", "Western Sahara"],
                "WB_STATUS": [
                    "Member country",
                    "Non-determined legal status area",
                ],
                "iso_code": ["USA", "ESH"],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap._load_world_bank_ndlsa_geodataframe",
            lambda: None,
        )
        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap._get_world_geodataframe",
            lambda: raw,
        )

        overlay = _get_disputed_territories_overlay("EPSG:4326")

        assert overlay is not None
        assert list(overlay["NAM_0"]) == ["Western Sahara"]


class TestCheckDependencies:
    """Test dependency checking."""

    @skip_no_viz
    def test_check_succeeds_when_installed(self):
        from metacouplingllm.visualization.worldmap import _check_dependencies

        _check_dependencies()  # should not raise


@skip_no_viz
class TestWorldBankAutoDownload:
    """Tests for first-run World Bank ADM0 auto-download flow."""

    def test_get_world_geodataframe_uses_downloaded_world_bank_data(
        self, monkeypatch
    ):
        import geopandas as gpd
        from shapely.geometry import Point

        from metacouplingllm.visualization.worldmap import _get_world_geodataframe

        calls: dict[str, int] = {"download": 0, "read": 0}

        def fake_download():
            from pathlib import Path

            calls["download"] += 1
            path = Path("downloaded.shp")
            path.write_text("placeholder", encoding="utf-8")
            return path

        def fake_read_file(path):
            calls["read"] += 1
            return gpd.GeoDataFrame(
                {
                    "ISO_A3": ["USA"],
                    "geometry": [Point(0, 0)],
                },
                geometry="geometry",
                crs="EPSG:4326",
            )

        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap._resolve_world_basemap_path",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap._download_world_bank_adm0_basemap",
            fake_download,
        )
        monkeypatch.setattr(
            "geopandas.read_file",
            fake_read_file,
        )

        world = _get_world_geodataframe()

        assert calls["download"] == 1
        assert calls["read"] == 1
        assert list(world["iso_code"]) == ["USA"]

    def test_get_world_geodataframe_raises_when_world_bank_data_unavailable(
        self, monkeypatch
    ):
        from metacouplingllm.visualization.worldmap import _get_world_geodataframe

        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap._resolve_world_basemap_path",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap._download_world_bank_adm0_basemap",
            lambda: None,
        )

        with pytest.raises(FileNotFoundError):
            _get_world_geodataframe()


class TestExtractCountriesFromAnalysis:
    """Test country extraction from ParsedAnalysis."""

    def test_extracts_from_nested_systems(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from metacouplingllm.visualization.worldmap import (
            _extract_countries_from_analysis,
        )

        analysis = ParsedAnalysis(
            systems={
                "sending": {"name": "Mexico", "geographic_scope": "Michoacan"},
                "receiving": {"name": "United States", "geographic_scope": "California"},
                "spillover": {"name": "Canada"},
            },
        )
        result = _extract_countries_from_analysis(analysis)
        assert result["sending"] == "MEX"
        assert result["receiving"] == "USA"
        assert result["spillover"] == "CAN"

    def test_extracts_from_flat_systems(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from metacouplingllm.visualization.worldmap import (
            _extract_countries_from_analysis,
        )

        analysis = ParsedAnalysis(
            systems={
                "sending": "Brazil soybean regions",
                "receiving": "China consumer markets",
            },
        )
        result = _extract_countries_from_analysis(analysis)
        assert result["sending"] == "BRA"
        assert result["receiving"] == "CHN"

    def test_returns_none_for_unresolvable(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from metacouplingllm.visualization.worldmap import (
            _extract_countries_from_analysis,
        )

        analysis = ParsedAnalysis(
            systems={
                "sending": "Tropical forests",
                "receiving": "Global markets",
            },
        )
        result = _extract_countries_from_analysis(analysis)
        assert result["sending"] is None
        assert result["receiving"] is None


class TestExtractAllAnalysisCountries:
    """Test the multi-country extraction for analysis-based maps."""

    def test_extracts_multiple_countries_from_geographic_scope(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from metacouplingllm.visualization.worldmap import (
            _extract_all_analysis_countries,
        )

        analysis = ParsedAnalysis(
            systems={
                "sending": {
                    "name": "Brazil Soybean Industry",
                    "geographic_scope": "Amazon and Cerrado regions in Brazil",
                },
                "receiving": {
                    "name": "China",
                    "geographic_scope": "Soybean-importing regions within China",
                },
                "spillover": {
                    "name": "Other soy-producing nations",
                    "geographic_scope": "Regions in Argentina, USA",
                },
            },
        )
        result = _extract_all_analysis_countries(analysis)
        assert "BRA" in result["sending"]
        assert "CHN" in result["receiving"]
        assert "ARG" in result["spillover"]
        assert "USA" in result["spillover"]

    def test_extracts_countries_from_flows(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from metacouplingllm.visualization.worldmap import (
            _extract_all_analysis_countries,
        )

        analysis = ParsedAnalysis(
            systems={
                "sending": {"name": "Brazil"},
                "receiving": {"name": "China"},
            },
            flows=[
                {"direction": "Brazil → China", "category": "matter"},
                {"direction": "Japan → Brazil", "category": "capital"},
            ],
        )
        result = _extract_all_analysis_countries(analysis)
        assert "BRA" in result["sending"]
        assert "CHN" in result["receiving"]
        # Japan mentioned in flows should appear in "other"
        assert "JPN" in result["other"]

    def test_empty_analysis_returns_empty_sets(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from metacouplingllm.visualization.worldmap import (
            _extract_all_analysis_countries,
        )

        analysis = ParsedAnalysis(systems={})
        result = _extract_all_analysis_countries(analysis)
        assert result["sending"] == set()
        assert result["receiving"] == set()
        assert result["spillover"] == set()

    def test_extracts_countries_from_effects_and_suggestions(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from metacouplingllm.visualization.worldmap import (
            _extract_all_analysis_countries,
        )

        analysis = ParsedAnalysis(
            systems={
                "sending": {"name": "Brazil"},
                "receiving": {"name": "China"},
                "spillover": {"name": "Argentina"},
            },
            effects={
                "socioeconomic": [
                    "Competitive pressure on soybean producers in the United States.",
                ],
            },
            suggestions=[
                "Consider competitor producers such as the United States and Argentina.",
            ],
        )

        result = _extract_all_analysis_countries(analysis)
        assert "USA" in result["other"]
        assert "ARG" in result["other"]


class TestCollectAnalysisMapCountryCodes:
    """Test which extracted countries are kept on the analysis map."""

    def test_includes_spillover_countries(self):
        from metacouplingllm.visualization.worldmap import (
            _collect_analysis_map_country_codes,
        )

        all_role_codes = {
            "sending": {"BRA"},
            "receiving": {"CHN"},
            "spillover": {"ARG"},
            "other": {"USA"},
        }

        result = _collect_analysis_map_country_codes(all_role_codes, "BRA")

        assert result == {"BRA", "CHN", "ARG", "USA"}


# ---------------------------------------------------------------------------
# Integration tests that require geopandas + matplotlib
# ---------------------------------------------------------------------------


@skip_no_viz
class TestPlotFocalCountryMap:
    """Integration tests for plot_focal_country_map."""

    def test_returns_figure(self):
        import matplotlib.figure

        from metacouplingllm.visualization import plot_focal_country_map

        fig = plot_focal_country_map("Mexico")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_by_iso_code(self):
        import matplotlib.figure

        from metacouplingllm.visualization import plot_focal_country_map

        fig = plot_focal_country_map("CHN")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_custom_title(self):
        from metacouplingllm.visualization import plot_focal_country_map

        fig = plot_focal_country_map("Brazil", title="Custom Title")
        ax = fig.axes[0]
        assert ax.get_title() == "Custom Title"
        plt.close(fig)

    def test_invalid_country_raises_valueerror(self):
        from metacouplingllm.visualization import plot_focal_country_map

        with pytest.raises(ValueError, match="Could not resolve"):
            plot_focal_country_map("Atlantis")

    def test_custom_figsize(self):
        from metacouplingllm.visualization import plot_focal_country_map

        fig = plot_focal_country_map("USA", figsize=(10, 5))
        w, h = fig.get_size_inches()
        assert abs(w - 10) < 0.5
        assert abs(h - 5) < 0.5
        plt.close(fig)


@skip_no_viz
class TestPlotAnalysisMap:
    """Integration tests for plot_analysis_map."""

    def test_returns_figure(self):
        import matplotlib.figure

        from metacouplingllm.llm.parser import ParsedAnalysis
        from metacouplingllm.visualization import plot_analysis_map

        analysis = ParsedAnalysis(
            systems={
                "sending": {"name": "Mexico", "geographic_scope": "Michoacan"},
                "receiving": {"name": "United States"},
                "spillover": {"name": "Canada"},
            },
        )
        fig = plot_analysis_map(analysis)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_raises_if_no_country_found(self):
        from metacouplingllm.llm.parser import ParsedAnalysis
        from metacouplingllm.visualization import plot_analysis_map

        analysis = ParsedAnalysis(
            systems={
                "sending": "Tropical forests",
                "receiving": "Global markets",
            },
        )
        with pytest.raises(ValueError, match="Could not resolve"):
            plot_analysis_map(analysis)

    def test_flat_systems(self):
        import matplotlib.figure

        from metacouplingllm.llm.parser import ParsedAnalysis
        from metacouplingllm.visualization import plot_analysis_map

        analysis = ParsedAnalysis(
            systems={
                "sending": "Ethiopia coffee regions",
                "receiving": "European Markets",
            },
        )
        fig = plot_analysis_map(analysis)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_with_flow_arrows(self):
        """Test that flow arrows are drawn on the map."""
        import matplotlib.figure

        from metacouplingllm.llm.parser import ParsedAnalysis
        from metacouplingllm.visualization import plot_analysis_map

        analysis = ParsedAnalysis(
            systems={
                "sending": {"name": "Brazil"},
                "receiving": {"name": "China"},
                "spillover": {"name": "Argentina"},
            },
            flows=[
                {
                    "category": "matter",
                    "direction": "Brazil → China",
                    "description": "Soybeans exported",
                },
                {
                    "category": "capital",
                    "direction": "China → Brazil",
                    "description": "Payment for soybeans",
                },
                {
                    "category": "information",
                    "direction": "Bidirectional (Brazil ↔ China)",
                    "description": "Market signals",
                },
            ],
        )
        fig = plot_analysis_map(analysis)
        assert isinstance(fig, matplotlib.figure.Figure)
        # Check that arrows were added as patches
        ax = fig.axes[0]
        from matplotlib.patches import FancyArrowPatch
        arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
        assert len(arrows) == 3
        plt.close(fig)

    def test_no_flows_no_arrows(self):
        """Without flows, no arrows should be drawn."""
        import matplotlib.figure

        from metacouplingllm.llm.parser import ParsedAnalysis
        from metacouplingllm.visualization import plot_analysis_map

        analysis = ParsedAnalysis(
            systems={
                "sending": {"name": "Brazil"},
                "receiving": {"name": "China"},
            },
        )
        fig = plot_analysis_map(analysis)
        assert isinstance(fig, matplotlib.figure.Figure)
        ax = fig.axes[0]
        from matplotlib.patches import FancyArrowPatch
        arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
        assert len(arrows) == 0
        plt.close(fig)

    def test_unresolvable_flow_direction_skipped(self):
        """Flows with unresolvable countries should be silently skipped."""
        import matplotlib.figure

        from metacouplingllm.llm.parser import ParsedAnalysis
        from metacouplingllm.visualization import plot_analysis_map

        analysis = ParsedAnalysis(
            systems={
                "sending": {"name": "Brazil"},
                "receiving": {"name": "China"},
            },
            flows=[
                {
                    "category": "matter",
                    "direction": "Atlantis → Narnia",
                    "description": "Fictional trade",
                },
            ],
        )
        fig = plot_analysis_map(analysis)
        assert isinstance(fig, matplotlib.figure.Figure)
        ax = fig.axes[0]
        from matplotlib.patches import FancyArrowPatch
        arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
        assert len(arrows) == 0
        plt.close(fig)

"""Tests for visualization/adm1_map.py — ADM1 subnational map rendering."""

import os
import pytest

from metacouplingllm.knowledge.adm1_pericoupling import (
    get_adm1_info,
    get_adm1_neighbors,
    get_adm1_codes_for_country,
    _ensure_loaded,
    _adm1_country,
)

# Check if geopandas is available
try:
    import geopandas
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for testing
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

skip_no_geopandas = pytest.mark.skipif(
    not HAS_GEOPANDAS, reason="geopandas/matplotlib not installed"
)

# Check if the World Bank Admin 1 GeoPackage is available
_WB_ADM1_GPKG = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "World Bank Official Boundaries - Admin 1.gpkg",
)
HAS_GPKG = os.path.isfile(_WB_ADM1_GPKG)

skip_no_gpkg = pytest.mark.skipif(
    not HAS_GPKG,
    reason=f"World Bank Admin 1 GPKG not found at {_WB_ADM1_GPKG}",
)


class TestAdm1Classification:
    """Test the _classify_adm1 helper (no geopandas needed)."""

    def test_classification_logic(self):
        from metacouplingllm.visualization.adm1_map import _classify_adm1

        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_country as ac
        assert ac is not None
        db_countries = set(ac.values())

        # Create a mock scenario
        focal_code = "AFG001"
        focal_country = "AFG"
        neighbor_codes = {"AFG024", "PAK005", "TJK001"}
        shapefile_codes = {
            "AFG001",  # focal
            "AFG024",  # neighbor (same country)
            "AFG010",  # same country, not neighbor
            "PAK005",  # neighbor (cross-border)
            "BRA001",  # telecoupling (different country)
            "XYZ999",  # unknown
        }
        shapefile_countries = {
            "AFG001": "AFG",
            "AFG024": "AFG",
            "AFG010": "AFG",
            "PAK005": "PAK",
            "BRA001": "BRA",
            "XYZ999": "XYZ",
        }

        classification = _classify_adm1(
            focal_code,
            focal_country,
            neighbor_codes,
            shapefile_codes,
            shapefile_countries,
            db_countries,
        )

        assert classification["AFG001"] == "intracoupling"
        assert classification["AFG024"] == "pericoupling"
        assert classification["AFG010"] == "telecoupling"
        assert classification["PAK005"] == "pericoupling"
        assert classification["BRA001"] == "telecoupling"
        assert classification["XYZ999"] == "na"

    def test_mentioned_countries_filters_telecoupling(self):
        """When mentioned_countries is provided, only those countries get telecoupling."""
        from metacouplingllm.visualization.adm1_map import _classify_adm1

        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_country as ac
        assert ac is not None
        db_countries = set(ac.values())

        focal_code = "USA023"  # Michigan
        focal_country = "USA"
        neighbor_codes = {"USA014", "CAN008"}  # Indiana, Ontario
        shapefile_codes = {
            "USA023",  # focal
            "USA014",  # neighbor
            "CAN008",  # neighbor (cross-border)
            "USA005",  # same country, distant
            "CHN001",  # China region (mentioned)
            "BRA001",  # Brazil region (NOT mentioned)
            "JPN001",  # Japan region (NOT mentioned)
        }
        shapefile_countries = {
            "USA023": "USA",
            "USA014": "USA",
            "CAN008": "CAN",
            "USA005": "USA",
            "CHN001": "CHN",
            "BRA001": "BRA",
            "JPN001": "JPN",
        }

        # Only USA and CHN are mentioned in the analysis
        mentioned = {"USA", "CHN"}

        classification = _classify_adm1(
            focal_code,
            focal_country,
            neighbor_codes,
            shapefile_codes,
            shapefile_countries,
            db_countries,
            mentioned_countries=mentioned,
        )

        assert classification["USA023"] == "intracoupling"
        assert classification["USA014"] == "pericoupling"
        assert classification["CAN008"] == "pericoupling"  # neighbor stays pericoupled
        assert classification["USA005"] == "telecoupling"   # same country, mentioned
        assert classification["CHN001"] == "telecoupling"   # mentioned country
        assert classification["BRA001"] == "na"             # NOT mentioned → grey
        assert classification["JPN001"] == "na"             # NOT mentioned → grey

    def test_mentioned_countries_none_uses_legacy(self):
        """When mentioned_countries is None, all DB countries are telecoupling."""
        from metacouplingllm.visualization.adm1_map import _classify_adm1

        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_country as ac
        assert ac is not None
        db_countries = set(ac.values())

        classification = _classify_adm1(
            "AFG001", "AFG", {"AFG024"},
            {"AFG001", "AFG024", "BRA001"},
            {"AFG001": "AFG", "AFG024": "AFG", "BRA001": "BRA"},
            db_countries,
            mentioned_countries=None,
        )
        # Legacy: BRA is in db_countries → telecoupling
        assert classification["BRA001"] == "telecoupling"

    def test_mentioned_adm1_codes_strict_filter_focal_country(self):
        """With mentioned_adm1_codes, non-mentioned focal-country states are NA.

        This is the Mato Grosso → China soybean scenario: only
        substantively mentioned Brazilian states should be coloured.
        """
        from metacouplingllm.visualization.adm1_map import _classify_adm1

        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_country as ac
        assert ac is not None
        db_countries = set(ac.values())

        focal_code = "BRA011"  # Mato Grosso
        focal_country = "BRA"
        neighbor_codes = {
            "BRA004", "BRA009", "BRA012", "BRA018",
            "BRA026", "BRA031", "BOL008",
        }
        shapefile_codes = {
            "BRA011",  # focal
            "BRA004", "BRA009", "BRA012",  # mentioned neighbors
            "BRA018", "BRA026", "BRA031",  # mentioned neighbors
            "BOL008",                        # mentioned cross-border neighbor
            "BRA013", "BRA014", "BRA020",   # Brazilian states NOT mentioned
            "CHN001",                        # China region (mentioned country)
        }
        shapefile_countries = {
            "BRA011": "BRA", "BRA004": "BRA", "BRA009": "BRA",
            "BRA012": "BRA", "BRA018": "BRA", "BRA026": "BRA",
            "BRA031": "BRA", "BRA013": "BRA", "BRA014": "BRA",
            "BRA020": "BRA", "BOL008": "BOL", "CHN001": "CHN",
        }

        # All neighbors are mentioned in the analysis
        mentioned_adm1 = {
            "BRA004", "BRA009", "BRA012", "BRA018",
            "BRA026", "BRA031", "BOL008",
        }
        mentioned = {"BRA", "CHN"}  # focal + receiving

        classification = _classify_adm1(
            focal_code,
            focal_country,
            neighbor_codes,
            shapefile_codes,
            shapefile_countries,
            db_countries,
            mentioned_countries=mentioned,
            mentioned_adm1_codes=mentioned_adm1,
        )

        # Focal
        assert classification["BRA011"] == "intracoupling"
        # Mentioned DB neighbors in focal country
        assert classification["BRA004"] == "pericoupling"
        assert classification["BRA009"] == "pericoupling"
        assert classification["BRA012"] == "pericoupling"
        assert classification["BRA018"] == "pericoupling"
        assert classification["BRA026"] == "pericoupling"
        assert classification["BRA031"] == "pericoupling"
        # Cross-border mentioned neighbor
        assert classification["BOL008"] == "pericoupling"
        # Non-mentioned Brazilian states — THE BUG FIX
        assert classification["BRA013"] == "na", (
            "Non-mentioned focal-country state should be grey"
        )
        assert classification["BRA014"] == "na"
        assert classification["BRA020"] == "na"
        # Other mentioned country (not focal) — country-level
        assert classification["CHN001"] == "telecoupling"

    def test_mentioned_adm1_codes_db_neighbor_not_mentioned(self):
        """DB neighbors that aren't mentioned in the analysis become NA too."""
        from metacouplingllm.visualization.adm1_map import _classify_adm1

        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_country as ac
        assert ac is not None
        db_countries = set(ac.values())

        focal_code = "BRA011"
        focal_country = "BRA"
        neighbor_codes = {"BRA004", "BRA009", "BRA012", "BRA018"}
        shapefile_codes = {
            "BRA011",
            "BRA004", "BRA009",  # mentioned neighbors
            "BRA012", "BRA018",  # DB neighbors NOT mentioned
        }
        shapefile_countries = {
            "BRA011": "BRA", "BRA004": "BRA", "BRA009": "BRA",
            "BRA012": "BRA", "BRA018": "BRA",
        }
        mentioned_adm1 = {"BRA004", "BRA009"}  # only 2 of 4 neighbors
        mentioned = {"BRA"}

        classification = _classify_adm1(
            focal_code,
            focal_country,
            neighbor_codes,
            shapefile_codes,
            shapefile_countries,
            db_countries,
            mentioned_countries=mentioned,
            mentioned_adm1_codes=mentioned_adm1,
        )

        assert classification["BRA011"] == "intracoupling"
        # Mentioned neighbors — pericoupling
        assert classification["BRA004"] == "pericoupling"
        assert classification["BRA009"] == "pericoupling"
        # DB neighbors NOT mentioned — NA (strict mode)
        assert classification["BRA012"] == "na"
        assert classification["BRA018"] == "na"

    def test_mentioned_adm1_codes_non_neighbor_mentioned(self):
        """Non-neighbor mentioned region in focal country = telecoupling (non-adjacent)."""
        from metacouplingllm.visualization.adm1_map import _classify_adm1

        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_country as ac
        assert ac is not None
        db_countries = set(ac.values())

        focal_code = "BRA011"
        focal_country = "BRA"
        neighbor_codes = {"BRA004"}
        shapefile_codes = {"BRA011", "BRA004", "BRA013"}
        shapefile_countries = {
            "BRA011": "BRA", "BRA004": "BRA", "BRA013": "BRA",
        }
        # BRA013 is NOT a DB neighbor but IS mentioned in the analysis
        mentioned_adm1 = {"BRA004", "BRA013"}
        mentioned = {"BRA"}

        classification = _classify_adm1(
            focal_code,
            focal_country,
            neighbor_codes,
            shapefile_codes,
            shapefile_countries,
            db_countries,
            mentioned_countries=mentioned,
            mentioned_adm1_codes=mentioned_adm1,
        )

        assert classification["BRA004"] == "pericoupling"  # DB neighbor + mentioned
        assert classification["BRA013"] == "telecoupling"  # non-neighbor + mentioned

    def test_mentioned_adm1_codes_cross_border_neighbor_not_mentioned(self):
        """Cross-border DB neighbors (e.g. Santa Cruz, Bolivia) should
        become NA when not in mentioned_adm1_codes, just like
        focal-country DB neighbors.

        This is the Mato Grosso → China scenario: the LLM has no
        substantive evidence of interaction with Santa Cruz, so it
        must NOT be coloured pericoupling on the map.
        """
        from metacouplingllm.visualization.adm1_map import _classify_adm1

        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_country as ac
        assert ac is not None
        db_countries = set(ac.values())

        focal_code = "BRA011"  # Mato Grosso
        focal_country = "BRA"
        neighbor_codes = {
            "BRA004", "BRA009", "BRA012",
            "BOL008",  # Santa Cruz, Bolivia — cross-border neighbor
        }
        shapefile_codes = {
            "BRA011",        # focal
            "BRA004", "BRA009", "BRA012",  # BR DB neighbors
            "BOL008",        # Santa Cruz (cross-border DB neighbor)
            "CHN001",        # China
        }
        shapefile_countries = {
            "BRA011": "BRA", "BRA004": "BRA", "BRA009": "BRA",
            "BRA012": "BRA", "BOL008": "BOL", "CHN001": "CHN",
        }
        # Empty mentioned_adm1 — LLM has no substantive evidence
        # for any region other than the focal.
        mentioned_adm1: set[str] = set()
        mentioned = {"BRA", "CHN"}  # focal + receiving

        classification = _classify_adm1(
            focal_code,
            focal_country,
            neighbor_codes,
            shapefile_codes,
            shapefile_countries,
            db_countries,
            mentioned_countries=mentioned,
            mentioned_adm1_codes=mentioned_adm1,
        )

        # Focal still coloured
        assert classification["BRA011"] == "intracoupling"
        # Brazilian DB neighbors NOT mentioned → NA (previous fix)
        assert classification["BRA004"] == "na"
        assert classification["BRA009"] == "na"
        assert classification["BRA012"] == "na"
        # Cross-border DB neighbor NOT mentioned → NA (THE NEW FIX)
        assert classification["BOL008"] == "na", (
            "Cross-border DB neighbor should be NA when not "
            "substantively mentioned in the analysis"
        )
        # Non-focal mentioned country still colored telecoupling
        assert classification["CHN001"] == "telecoupling"

    def test_mentioned_adm1_codes_cross_border_neighbor_mentioned(self):
        """When a cross-border DB neighbor IS explicitly mentioned,
        it should be coloured pericoupling (because it's both a DB
        neighbor AND substantively discussed).
        """
        from metacouplingllm.visualization.adm1_map import _classify_adm1

        _ensure_loaded()
        from metacouplingllm.knowledge.adm1_pericoupling import _adm1_country as ac
        assert ac is not None
        db_countries = set(ac.values())

        focal_code = "BRA011"
        focal_country = "BRA"
        neighbor_codes = {"BRA004", "BOL008"}
        shapefile_codes = {"BRA011", "BRA004", "BOL008", "BOL001"}
        shapefile_countries = {
            "BRA011": "BRA", "BRA004": "BRA",
            "BOL008": "BOL", "BOL001": "BOL",
        }
        # BOL008 is mentioned, BOL001 is not
        mentioned_adm1 = {"BOL008"}
        mentioned = {"BRA", "BOL"}

        classification = _classify_adm1(
            focal_code,
            focal_country,
            neighbor_codes,
            shapefile_codes,
            shapefile_countries,
            db_countries,
            mentioned_countries=mentioned,
            mentioned_adm1_codes=mentioned_adm1,
        )

        assert classification["BRA011"] == "intracoupling"
        # BOL008 mentioned + DB neighbor → pericoupling
        assert classification["BOL008"] == "pericoupling"
        # BOL001 is in a mentioned country (BOL) but is not a DB
        # neighbor and not in mentioned_adm1 → telecoupling at
        # country level
        assert classification["BOL001"] == "telecoupling"
        # BRA004 is a DB neighbor but not mentioned → NA
        assert classification["BRA004"] == "na"


class TestAdm1RenderHelpers:
    """Tests for ADM1/country layer selection in ADM1 maps."""

    def test_select_adm1_detail_countries_keeps_focal_and_neighbors(self):
        from metacouplingllm.visualization.adm1_map import (
            _select_adm1_detail_countries,
        )

        detail = _select_adm1_detail_countries(
            focal_country="USA",
            neighbor_codes={"USA015", "CAN008", "ZZZ999"},
            shapefile_countries={
                "USA015": "USA",
                "CAN008": "CAN",
            },
        )

        assert detail == {"USA", "CAN"}

    @skip_no_geopandas
    def test_dissolve_country_layer_preserves_country_color(self):
        import geopandas as gpd
        from shapely.geometry import Polygon

        from metacouplingllm.visualization.adm1_map import _dissolve_country_layer

        color_map = {
            "intracoupling": "#D4E79E",
            "pericoupling": "#4CAF50",
            "telecoupling": "#ADD8E6",
            "na": "#D3D3D3",
        }
        country_layer_raw = gpd.GeoDataFrame(
            {
                "adm1_code": ["CHN001", "CHN002", "MEX001"],
                "iso_code": ["CHN", "CHN", "MEX"],
                "coupling": ["telecoupling", "telecoupling", "na"],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                    Polygon([(3, 0), (4, 0), (4, 1), (3, 1)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        dissolved = _dissolve_country_layer(country_layer_raw, color_map)

        assert dissolved is not None
        couplings = dict(zip(dissolved["iso_code"], dissolved["coupling"]))
        colors = dict(zip(dissolved["iso_code"], dissolved["_color"]))
        assert couplings["CHN"] == "telecoupling"
        assert colors["CHN"] == "#ADD8E6"
        assert couplings["MEX"] == "na"

    @skip_no_geopandas
    def test_draw_mixed_flow_arrows_draws_domestic_adm1_arrow(self):
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from shapely.geometry import Polygon

        from metacouplingllm.visualization.adm1_map import _draw_mixed_flow_arrows

        world = gpd.GeoDataFrame(
            {
                "adm1_code": ["USA023", "USA015", "CHN001"],
                "iso_code": ["USA", "USA", "CHN"],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1.2, 0), (2.2, 0), (2.2, 1), (1.2, 1)]),
                    Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        flows = [
            {
                "category": "matter",
                "direction": "Michigan → Indiana",
                "source_adm1": "USA023",
                "target_adm1": "USA015",
                "is_bidirectional": False,
            },
        ]

        fig, ax = plt.subplots()
        handles = _draw_mixed_flow_arrows(ax, world, flows)

        assert len(handles) == 1
        assert handles[0].get_label() == "Matter Flow"
        assert len(ax.patches) >= 1
        plt.close(fig)

    @skip_no_geopandas
    def test_focal_adm1_substitution_in_international_flow(self):
        """International flows should anchor on the focal ADM1 centroid,
        not on the focal country's centroid.
        """
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
        from shapely.geometry import Polygon

        from metacouplingllm.visualization.adm1_map import _draw_mixed_flow_arrows

        # Build a synthetic world where:
        # - BRA011 (Mato Grosso) is a small polygon at (0.5, 0.5)
        # - BRA099 is a large polygon centered around (10, 0)
        # - Dissolved Brazil's representative_point lands inside BRA099
        #   (at approximately (10, 0)) — clearly distinct from BRA011.
        # - CHN001 is far east at (30, 0.5).
        world = gpd.GeoDataFrame(
            {
                "adm1_code": ["BRA011", "BRA099", "CHN001"],
                "iso_code": ["BRA", "BRA", "CHN"],
                "adm1_name": ["Mato Grosso", "Other BR", "China"],
                "country_name": ["Brazil", "Brazil", "China"],
            },
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(5, -5), (15, -5), (15, 5), (5, 5)]),
                Polygon([(30, 0), (31, 0), (31, 1), (30, 1)]),
            ],
            crs="EPSG:4326",
        )
        flows = [
            {
                "category": "matter",
                "direction": "Brazil → China",
                "description": "Soybean exports",
            },
        ]

        # Without focal_adm1: arrow starts near Brazil's dissolved centroid
        fig1, ax1 = plt.subplots()
        _draw_mixed_flow_arrows(ax1, world, flows)
        arrows1 = [p for p in ax1.patches if isinstance(p, FancyArrowPatch)]
        assert len(arrows1) == 1
        src_no_focal = arrows1[0]._posA_posB[0]
        plt.close(fig1)

        # With focal_adm1=BRA011, arrow should start at Mato Grosso's
        # centroid (~0.5, 0.5) instead of Brazil's dissolved rep_point
        # (~10, 0).
        fig2, ax2 = plt.subplots()
        _draw_mixed_flow_arrows(
            ax2,
            world,
            flows,
            focal_adm1="BRA011",
            focal_country_iso="BRA",
        )
        arrows2 = [p for p in ax2.patches if isinstance(p, FancyArrowPatch)]
        assert len(arrows2) == 1
        src_with_focal = arrows2[0]._posA_posB[0]
        plt.close(fig2)

        # The focal-substituted source should be clearly west of the
        # non-focal source.
        assert src_with_focal[0] < src_no_focal[0] - 5, (
            f"Expected focal substitution to move source west. "
            f"no_focal={src_no_focal}, with_focal={src_with_focal}"
        )
        # Focal substitution should be close to Mato Grosso centroid (~0.5)
        assert abs(src_with_focal[0] - 0.5) < 0.6
        assert abs(src_with_focal[1] - 0.5) < 0.6

    @skip_no_geopandas
    def test_focal_adm1_substitution_on_target(self):
        """Reverse flows (e.g. China → Brazil) should substitute the
        target centroid with the focal ADM1 centroid.
        """
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
        from shapely.geometry import Polygon

        from metacouplingllm.visualization.adm1_map import _draw_mixed_flow_arrows

        world = gpd.GeoDataFrame(
            {
                "adm1_code": ["BRA011", "BRA099", "CHN001"],
                "iso_code": ["BRA", "BRA", "CHN"],
                "adm1_name": ["Mato Grosso", "Other BR", "China"],
                "country_name": ["Brazil", "Brazil", "China"],
            },
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(5, -5), (15, -5), (15, 5), (5, 5)]),
                Polygon([(30, 0), (31, 0), (31, 1), (30, 1)]),
            ],
            crs="EPSG:4326",
        )
        flows = [
            {
                "category": "capital",
                "direction": "China → Brazil",
                "description": "Payments",
            },
        ]

        fig, ax = plt.subplots()
        _draw_mixed_flow_arrows(
            ax,
            world,
            flows,
            focal_adm1="BRA011",
            focal_country_iso="BRA",
        )
        arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
        assert len(arrows) == 1
        tgt = arrows[0]._posA_posB[1]
        plt.close(fig)

        # Target should be at Mato Grosso centroid (~0.5, 0.5)
        assert abs(tgt[0] - 0.5) < 0.6
        assert abs(tgt[1] - 0.5) < 0.6


class TestAdm1DownloadHelpers:
    """Tests for ADM1 auto-download helper logic."""

    def test_download_adm1_geopackage_writes_file(self, monkeypatch, tmp_path):
        from pathlib import Path

        from metacouplingllm.visualization.adm1_map import _download_adm1_geopackage

        def fake_urlretrieve(url, dest):
            Path(dest).write_text("placeholder", encoding="utf-8")
            return str(dest), None

        monkeypatch.setattr(
            "metacouplingllm.visualization.adm1_map._get_adm1_cache_dir",
            lambda: tmp_path / "cache",
        )
        monkeypatch.setenv(
            "METACOUPLING_ADM1_DOWNLOAD_URL",
            "https://example.com/World.Bank.Official.Boundaries.-.Admin.1.gpkg",
        )
        monkeypatch.setattr(
            "metacouplingllm.visualization.adm1_map.urllib.request.urlretrieve",
            fake_urlretrieve,
        )

        downloaded = _download_adm1_geopackage()

        assert downloaded is not None
        assert downloaded.name == "World.Bank.Official.Boundaries.-.Admin.1.gpkg"
        assert downloaded.exists()


@skip_no_geopandas
class TestAdm1MapDependencies:
    """Test dependency checking for ADM1 maps."""

    def test_check_dependencies_passes(self):
        from metacouplingllm.visualization.adm1_map import _check_dependencies
        # Should not raise
        _check_dependencies()

    def test_get_adm1_geodataframe_uses_downloaded_file(
        self, monkeypatch, tmp_path
    ):
        from pathlib import Path

        import geopandas as gpd
        from shapely.geometry import Point

        from metacouplingllm.visualization.adm1_map import _get_adm1_geodataframe

        def fake_download():
            path = tmp_path / "downloaded.gpkg"
            path.write_text("placeholder", encoding="utf-8")
            return path

        def fake_read_file(path):
            return gpd.GeoDataFrame(
                {
                    "ADM1CD_c": ["USA023"],
                    "ISO_A3": ["USA"],
                    "geometry": [Point(0, 0)],
                },
                geometry="geometry",
                crs="EPSG:4326",
            )

        monkeypatch.setattr(
            "metacouplingllm.visualization.adm1_map._get_adm1_cache_dir",
            lambda: Path("missing-cache"),
        )
        monkeypatch.setattr(
            "metacouplingllm.visualization.adm1_map._download_adm1_geopackage",
            fake_download,
        )
        monkeypatch.setattr(
            "geopandas.read_file",
            fake_read_file,
        )

        gdf = _get_adm1_geodataframe()

        assert "adm1_code" in gdf.columns
        assert "iso_code" in gdf.columns
        assert list(gdf["adm1_code"]) == ["USA023"]


@skip_no_geopandas
class TestDisputedOverlay:
    """Test disputed-territory overlay fallback logic."""

    def test_disputed_overlay_uses_worldmap_helper(self, monkeypatch):
        import geopandas as gpd
        from shapely.geometry import Polygon

        from metacouplingllm.visualization.adm1_map import (
            _get_disputed_overlay_geodataframe,
        )

        source = gpd.GeoDataFrame(
            {
                "adm1_code": ["MAR001"],
                "iso_code": ["MAR"],
                "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        fallback = gpd.GeoDataFrame(
            {
                "NAME": ["Siachen Glacier"],
                "iso_code": ["-99"],
                "geometry": [Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        monkeypatch.setattr(
            "metacouplingllm.visualization.worldmap._get_disputed_territories_overlay",
            lambda *args, **kwargs: fallback,
        )

        overlay = _get_disputed_overlay_geodataframe(source)

        assert overlay is not None
        assert "Siachen Glacier" in set(overlay["NAME"])


@skip_no_geopandas
class TestAdm1Colors:
    """Test the ADM1 color scheme."""

    def test_default_colors(self):
        from metacouplingllm.visualization.adm1_map import (
            Adm1CouplingColors,
            DEFAULT_ADM1_COLORS,
        )
        assert DEFAULT_ADM1_COLORS.intracoupling == "#D4E79E"
        assert DEFAULT_ADM1_COLORS.pericoupling == "#4CAF50"
        assert DEFAULT_ADM1_COLORS.telecoupling == "#ADD8E6"
        assert DEFAULT_ADM1_COLORS.na == "#D3D3D3"
        assert DEFAULT_ADM1_COLORS.ocean == "#DCEEFB"
        assert DEFAULT_ADM1_COLORS.lake == "#4A90D9"
        assert DEFAULT_ADM1_COLORS.disputed == "#BFBFBF"
        assert DEFAULT_ADM1_COLORS.disputed_outline == "#4D4D4D"
        assert DEFAULT_ADM1_COLORS.disputed_hatch == "///"

    def test_custom_colors(self):
        from metacouplingllm.visualization.adm1_map import Adm1CouplingColors
        custom = Adm1CouplingColors(intracoupling="#FF0000")
        assert custom.intracoupling == "#FF0000"
        assert custom.pericoupling == "#4CAF50"  # others keep defaults
        assert custom.lake == "#4A90D9"
        assert custom.disputed == "#BFBFBF"
        assert custom.disputed_outline == "#4D4D4D"


@skip_no_geopandas
@skip_no_gpkg
class TestPlotFocalAdm1Map:
    """Integration tests for plot_focal_adm1_map (require GPKG file)."""

    def test_basic_plot(self):
        from metacouplingllm.visualization.adm1_map import plot_focal_adm1_map

        fig = plot_focal_adm1_map("AFG001", shapefile=_WB_ADM1_GPKG)
        assert fig is not None
        # Check that axes exist
        axes = fig.get_axes()
        assert len(axes) > 0

    def test_custom_title(self):
        from metacouplingllm.visualization.adm1_map import plot_focal_adm1_map

        fig = plot_focal_adm1_map(
            "AFG001",
            shapefile=_WB_ADM1_GPKG,
            title="Test Title",
        )
        ax = fig.get_axes()[0]
        assert ax.get_title() == "Test Title"

    def test_zoom_to_focal(self):
        from metacouplingllm.visualization.adm1_map import plot_focal_adm1_map

        fig = plot_focal_adm1_map(
            "AFG001",
            shapefile=_WB_ADM1_GPKG,
            zoom_to_focal=True,
        )
        ax = fig.get_axes()[0]
        # Zoom should restrict the view — xlim and ylim should not be
        # the full world extent
        xlim = ax.get_xlim()
        assert xlim[1] - xlim[0] < 350  # Not full world width

    def test_invalid_code_raises(self):
        from metacouplingllm.visualization.adm1_map import plot_focal_adm1_map

        with pytest.raises(ValueError, match="not in the pericoupling database"):
            plot_focal_adm1_map("ZZZZZ", shapefile=_WB_ADM1_GPKG)

    def test_missing_shapefile_raises(self):
        from metacouplingllm.visualization.adm1_map import plot_focal_adm1_map

        with pytest.raises((FileNotFoundError, Exception)):
            plot_focal_adm1_map("AFG001", shapefile="/nonexistent/file.gpkg")


@skip_no_geopandas
@skip_no_gpkg
class TestColumnNormalisation:
    """Test that the GeoDataFrame column normalisation works."""

    def test_normalise_wb_columns(self):
        from metacouplingllm.visualization.adm1_map import _get_adm1_geodataframe

        gdf = _get_adm1_geodataframe(shapefile=_WB_ADM1_GPKG)
        assert "adm1_code" in gdf.columns
        assert "iso_code" in gdf.columns
        # Check that ADM1 codes look correct (most are AAA000 but some
        # have 2-digit suffixes like NZL02 or special codes like MOZXXX)
        assert gdf["adm1_code"].str.match(r"^[A-Z]{3}[A-Z0-9]{2,3}$").all()

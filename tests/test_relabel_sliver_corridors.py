"""Tests for the source-relabel stage (scripts/relabel_sliver_corridors.py).

The manifest-consistency checks run in CI (committed CSV). The geometry
verification needs the pinned WB Admin-1 GeoPackage, which is not committed,
so it skips automatically when the file is absent.
"""
import importlib.util
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RELABEL_CSV = REPO / "src" / "metacouplingllm" / "data" / "sliver_corridor_relabel.csv"


def _load():
    spec = importlib.util.spec_from_file_location(
        "relabel_sliver_corridors", REPO / "scripts" / "relabel_sliver_corridors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gpkg():
    """Locate the WB Admin-1 GeoPackage if present (env override or Downloads)."""
    env = os.environ.get("WB_ADM1_GPKG")
    if env and Path(env).exists():
        return env
    for c in [Path.home() / "Downloads" / "World Bank Official Boundaries - Admin 1 (1).gpkg",
              Path.home() / "Downloads" / "World Bank Official Boundaries - Admin 1.gpkg"]:
        if c.exists():
            return str(c)
    return None


class TestManifest:
    def test_manifest_loads_and_has_ten_hosts(self):
        mod = _load()
        m = mod.load_relabel_manifest()
        assert len(m) == 10, f"expected 10 reviewed host polygons, got {len(m)}"
        # Arusha is the only multi-owner host (NW->Mara, E->Kilimanjaro)
        assert set(m["TZA001"]["owners"]) == {"TZA016", "TZA011"}
        # ARG017 carries the per-host opening override
        assert m["ARG017"]["d_deg"] == pytest.approx(3e-3)
        assert m["SRB002"]["d_deg"] == pytest.approx(2e-3)


class TestGeometry:
    @pytest.fixture(scope="class")
    def relabeled(self):
        gpkg = _gpkg()
        if gpkg is None:
            pytest.skip("WB Admin-1 GeoPackage not available")
        import geopandas as gpd
        mod = _load()
        g = gpd.read_file(gpkg, layer="WB_GAD_ADM1").to_crs(4326)
        bad = ~g.geometry.is_valid
        if bad.any():
            g.loc[bad, g.geometry.name] = g.geometry[bad].buffer(0)
        g = g.reset_index(drop=True)
        g2, log = mod.relabel(g)
        return mod, g, g2, log

    def test_area_conserved(self, relabeled):
        mod, g, g2, _ = relabeled
        drift = (sum(mod._area_km2(x) for x in g2.geometry)
                 - sum(mod._area_km2(x) for x in g.geometry))
        assert abs(drift) < 0.01, f"area drift {drift:.4f} km2 (must be ~0)"

    def test_twelve_corridors_moved(self, relabeled):
        _, _, _, log = relabeled
        assert len(log) == 12

    def test_bogus_edges_removed(self, relabeled):
        mod, g, g2, _ = relabeled
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        idx = {c: i for i, c in enumerate(g2["ADM1CD_c"])}

        def contact(gdf, ca, cb):
            a, b = gdf.geometry.iloc[idx[ca]], gdf.geometry.iloc[idx[cb]]
            if a.distance(b) > 0:
                return 0.0
            inter = a.boundary.intersection(b.boundary)
            if inter.is_empty or inter.geom_type in ("Point", "MultiPoint"):
                return 0.0
            return geod.geometry_length(inter) / 1000.0

        for ca, cb in [("KEN027", "TZA001"), ("KEN039", "TZA001"),
                       ("ARG017", "BOL007"), ("SRB002", "ROU028")]:
            assert contact(g, ca, cb) > 5, f"{ca}-{cb} should be a bogus edge before"
            assert contact(g2, ca, cb) == 0.0, f"{ca}-{cb} should be gone after relabel"

    def test_starved_edge_recovered(self, relabeled):
        mod, g, g2, _ = relabeled
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        idx = {c: i for i, c in enumerate(g2["ADM1CD_c"])}

        def contact(gdf, ca, cb):
            a, b = gdf.geometry.iloc[idx[ca]], gdf.geometry.iloc[idx[cb]]
            inter = a.boundary.intersection(b.boundary)
            if inter.is_empty or inter.geom_type in ("Point", "MultiPoint"):
                return 0.0
            return geod.geometry_length(inter) / 1000.0

        # Migori<->Mara: starved to ~20 km (18 in-lake), recovered to ~103 km raw
        assert contact(g, "KEN027", "TZA016") < 25
        assert contact(g2, "KEN027", "TZA016") > 90

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
        # every host uses the default opening radius
        assert all(s["d_deg"] == pytest.approx(mod.OPENING_D_DEG) for s in m.values())


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

    def test_no_zero_width_spikes(self, relabeled):
        # the cut points are shared vertices, so no host or owner ring runs out
        # along a line and back (the spike the overlay left before 2026-09-22)
        import math
        mod, _, g2, _ = relabeled
        idx = {c: i for i, c in enumerate(g2["ADM1CD_c"])}
        units = {u for h, s in mod.load_relabel_manifest().items() for u in (h, *s["owners"])}
        for u in sorted(units):
            geom = g2.geometry.iloc[idx[u]]
            for ring in mod._rings(geom):
                c = list(ring.coords)[:-1]
                for i in range(len(c)):
                    a, t, b = c[i - 1], c[i], c[(i + 1) % len(c)]
                    vx, vy, wx, wy = a[0] - t[0], a[1] - t[1], b[0] - t[0], b[1] - t[1]
                    la, lb = math.hypot(vx, vy), math.hypot(wx, wy)
                    spike = (la and lb and (vx * wx + vy * wy) / (la * lb) > 0.999999
                             and abs(vx * wy - vy * wx) / max(la, lb) < 1e-6)
                    assert not spike, f"{u}: zero-width spike at {t}"

    def test_owner_takes_the_hosts_frontage_exactly(self, relabeled):
        # Lamwo's corridor carries ~20 km of the South Sudan border to Kitgum:
        # the two units' frontage with Eastern Equatoria is conserved and
        # Kitgum's share is measured in full (12.92 km while it was not noded)
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        _, g, g2, _ = relabeled
        idx = {c: i for i, c in enumerate(g2["ADM1CD_c"])}

        def km(gdf, ca, cb):
            s = gdf.geometry.iloc[idx[ca]].boundary.intersection(gdf.geometry.iloc[idx[cb]].boundary)
            return 0.0 if s.is_empty else geod.geometry_length(s) / 1000.0

        before = km(g, "UGA065", "SSD002") + km(g, "UGA056", "SSD002")
        after = km(g2, "UGA065", "SSD002") + km(g2, "UGA056", "SSD002")
        assert after == pytest.approx(before, abs=1e-6)
        assert km(g2, "UGA056", "SSD002") == pytest.approx(21.279, abs=0.01)

    def test_corridors_close_at_junctions(self, relabeled):
        # Salta's corridor would stop 304.2 m short of the Potosi-Tarija point and
        # Branicevo's run 13.8 m past the Caras-Severin-Mehedinti point; closed at
        # those points, neither Salta<->Potosi nor Bor<->Caras-Severin has a line
        _, _, g2, log = relabeled
        closed = {e["host"]: e["closed_at_junction_m"] for e in log if e["closed_at_junction_m"]}
        assert set(closed) == {"ARG017", "SRB002"}
        assert closed["ARG017"] == [pytest.approx(304.2, abs=0.1)]
        assert closed["SRB002"] == [pytest.approx(13.8, abs=0.1)]
        idx = {c: i for i, c in enumerate(g2["ADM1CD_c"])}
        for ca, cb in [("ARG017", "BOL007"), ("SRB001", "ROU013")]:
            s = g2.geometry.iloc[idx[ca]].boundary.intersection(g2.geometry.iloc[idx[cb]].boundary)
            assert s.is_empty or s.geom_type in ("Point", "MultiPoint"), f"{ca}-{cb} line contact"

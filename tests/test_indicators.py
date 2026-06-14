"""Tests for the metacouplingllm.indicators submodule (PR #35).

Covers spec Section 20 worked test cases + the Brazil soybean
end-to-end example from spec Section 11 + pandas-optional gating.

Pandas is an optional dependency; this whole file skips when
pandas isn't installed (mirrors the python-docx pattern in
``tests/test_output.py``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

# Skip the entire module when pandas isn't installed -- mirrors the
# python-docx skip in tests/test_output.py.
pd = pytest.importorskip("pandas")

from metacouplingllm.indicators import (  # noqa: E402
    build_adjacency,
    classify_coupling,
    compute_flow_shares,
    compute_mfci,
    compute_mfe,
    summarize_metacoupling,
)
from metacouplingllm.indicators._math import (  # noqa: E402
    normalised_hhi,
    raw_hhi,
    shannon_entropy_normalised,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _brazil_soybean_edges() -> "pd.DataFrame":
    """Spec §11 worked example: Brazil exports 100 tons of soybean."""
    return pd.DataFrame([
        {"focal_system_id": "Brazil", "destination_id": "Brazil",
         "coupling_type": "I", "flow_value": 10},
        {"focal_system_id": "Brazil", "destination_id": "Argentina",
         "coupling_type": "P", "flow_value": 20},
        {"focal_system_id": "Brazil", "destination_id": "China",
         "coupling_type": "T", "flow_value": 60},
        {"focal_system_id": "Brazil", "destination_id": "EU",
         "coupling_type": "T", "flow_value": 5},
        {"focal_system_id": "Brazil", "destination_id": "Japan",
         "coupling_type": "T", "flow_value": 5},
    ])


def _suppress_user_warnings():
    """Context-manager-friendly: most tests don't care about the
    expected UserWarnings (zero-flow, single-partner, etc.).
    Specific tests that DO want to assert a warning use
    ``pytest.warns(UserWarning)`` explicitly."""
    return warnings.catch_warnings()


# ---------------------------------------------------------------------------
# TestFlowShares (4 cases)
# ---------------------------------------------------------------------------


class TestFlowShares:
    def test_brazil_soybean_shares_match_spec(self):
        """Spec §20 test 1: F_I=10, F_P=20, F_T=70 -> 0.10 / 0.20 / 0.70."""
        edges = _brazil_soybean_edges()
        with _suppress_user_warnings():
            warnings.simplefilter("ignore")
            out = compute_flow_shares(edges)
        row = out.iloc[0]
        assert row["F_I"] == 10
        assert row["F_P"] == 20
        assert row["F_T"] == 70
        assert row["F_total"] == 100
        assert row["IFS"] == pytest.approx(0.10)
        assert row["PFS"] == pytest.approx(0.20)
        assert row["TFS"] == pytest.approx(0.70)

    def test_shares_sum_to_one(self):
        edges = _brazil_soybean_edges()
        with _suppress_user_warnings():
            warnings.simplefilter("ignore")
            out = compute_flow_shares(edges)
        s = out.iloc[0]["IFS"] + out.iloc[0]["PFS"] + out.iloc[0]["TFS"]
        assert s == pytest.approx(1.0)

    def test_zero_total_flow_returns_nan_shares_and_warns(self):
        """Spec §13 edge case: empty edges for a system -> NaN shares
        plus a UserWarning."""
        edges = pd.DataFrame({
            "focal_system_id": ["A"],
            "destination_id": ["B"],
            "coupling_type": ["T"],
            "flow_value": [0.0],
        })
        with pytest.warns(UserWarning, match="zero total flow"):
            out = compute_flow_shares(edges)
        assert np.isnan(out.iloc[0]["IFS"])
        assert np.isnan(out.iloc[0]["PFS"])
        assert np.isnan(out.iloc[0]["TFS"])

    def test_multi_system_batch(self):
        edges = pd.DataFrame([
            {"focal_system_id": "A", "destination_id": "A",
             "coupling_type": "I", "flow_value": 50},
            {"focal_system_id": "A", "destination_id": "X",
             "coupling_type": "T", "flow_value": 50},
            {"focal_system_id": "B", "destination_id": "B",
             "coupling_type": "I", "flow_value": 25},
            {"focal_system_id": "B", "destination_id": "Y",
             "coupling_type": "P", "flow_value": 75},
        ])
        with _suppress_user_warnings():
            warnings.simplefilter("ignore")
            out = compute_flow_shares(edges)
        out = out.set_index("focal_system_id")
        assert out.loc["A", "IFS"] == pytest.approx(0.5)
        assert out.loc["A", "TFS"] == pytest.approx(0.5)
        assert out.loc["B", "IFS"] == pytest.approx(0.25)
        assert out.loc["B", "PFS"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# TestMFE (4 cases)
# ---------------------------------------------------------------------------


class TestMFE:
    def test_brazil_soybean_mfe_approx_0_73(self):
        """Spec §20 test 2: shares (0.10, 0.20, 0.70) -> MFE ≈ 0.73."""
        shares = pd.DataFrame([
            {"IFS": 0.10, "PFS": 0.20, "TFS": 0.70}
        ])
        out = compute_mfe(shares)
        assert out.iloc[0]["MFE"] == pytest.approx(0.7298, abs=0.001)

    def test_perfect_evenness_returns_one(self):
        """Spec §20 test 3: (1/3, 1/3, 1/3) -> MFE = 1."""
        shares = pd.DataFrame([
            {"IFS": 1 / 3, "PFS": 1 / 3, "TFS": 1 / 3}
        ])
        out = compute_mfe(shares)
        assert out.iloc[0]["MFE"] == pytest.approx(1.0)

    def test_complete_dominance_returns_zero(self):
        """Spec §20 test 4: (0, 0, 1) -> MFE = 0."""
        shares = pd.DataFrame([
            {"IFS": 0.0, "PFS": 0.0, "TFS": 1.0}
        ])
        out = compute_mfe(shares)
        assert out.iloc[0]["MFE"] == pytest.approx(0.0)

    def test_0_ln_0_convention(self):
        """0·ln(0) should be 0, not NaN.  Test entropy primitive
        directly with a share array containing zeros."""
        # (0, 0.5, 0.5) should give entropy = ln(2)/ln(3) ≈ 0.631
        # because the 0-share contributes nothing.
        h = shannon_entropy_normalised(np.array([0.0, 0.5, 0.5]))
        assert h == pytest.approx(np.log(2) / np.log(3))


# ---------------------------------------------------------------------------
# TestMFCI (4 cases)
# ---------------------------------------------------------------------------


class TestMFCI:
    def test_brazil_telecoupled_mfci_approx_0_617(self):
        """Spec §20 test 5: telecoupled shares 0.857/0.071/0.071
        -> HHI≈0.745, MFCI≈0.617 with n=3."""
        edges = _brazil_soybean_edges()
        with _suppress_user_warnings():
            warnings.simplefilter("ignore")
            out = compute_mfci(edges)
        # T row for Brazil
        tele = out.loc[
            (out["focal_system_id"] == "Brazil")
            & (out["coupling_type"] == "T")
        ].iloc[0]
        assert tele["n_partners"] == 3
        assert tele["HHI"] == pytest.approx(0.745, abs=0.005)
        assert tele["MFCI"] == pytest.approx(0.617, abs=0.005)

    def test_single_partner_mfci_is_one_with_warning(self):
        """Spec §20 test 6: only one partner -> MFCI=1 by convention
        plus a UserWarning (mentioned in §8 + §14.6)."""
        edges = pd.DataFrame([
            {"focal_system_id": "A", "destination_id": "X",
             "coupling_type": "T", "flow_value": 100},
        ])
        with pytest.warns(UserWarning):
            out = compute_mfci(edges)
        t_row = out.loc[out["coupling_type"] == "T"].iloc[0]
        assert t_row["n_partners"] == 1
        assert t_row["MFCI"] == pytest.approx(1.0)

    def test_zero_flow_coupling_type_returns_nan(self):
        """Spec §20 test 7: F_ic=0 -> MFCI = NaN."""
        edges = pd.DataFrame([
            {"focal_system_id": "A", "destination_id": "X",
             "coupling_type": "P", "flow_value": 50},
        ])
        with _suppress_user_warnings():
            warnings.simplefilter("ignore")
            out = compute_mfci(edges)
        # No T-coupling edges -> T row is NaN.
        t_row = out.loc[out["coupling_type"] == "T"].iloc[0]
        assert np.isnan(t_row["MFCI"])
        # I row also NaN since no I edges either.
        i_row = out.loc[out["coupling_type"] == "I"].iloc[0]
        assert np.isnan(i_row["MFCI"])

    def test_effective_n_partners_matches_reciprocal_hhi(self):
        """Spec §9.2: ENP = 1 / HHI_raw."""
        edges = _brazil_soybean_edges()
        with _suppress_user_warnings():
            warnings.simplefilter("ignore")
            out = compute_mfci(edges)
        tele = out.loc[
            (out["focal_system_id"] == "Brazil")
            & (out["coupling_type"] == "T")
        ].iloc[0]
        assert tele["effective_n_partners"] == pytest.approx(
            1.0 / tele["HHI"]
        )


# ---------------------------------------------------------------------------
# TestClassifyCoupling (4 cases)
# ---------------------------------------------------------------------------


class TestClassifyCoupling:
    def test_self_loop_classifies_as_intra(self):
        edges = pd.DataFrame([
            {"origin_id": "Brazil", "destination_id": "Brazil"},
        ])
        # No adjacency needed for self-loops.
        out = classify_coupling(edges, focal_id="Brazil")
        assert out.iloc[0]["coupling_type"] == "I"

    def test_adjacent_classifies_as_peri(self):
        edges = pd.DataFrame([
            {"origin_id": "Brazil", "destination_id": "Argentina"},
            {"origin_id": "Brazil", "destination_id": "Bolivia"},
        ])
        adjacency = pd.DataFrame([
            {"origin_id": "Brazil", "destination_id": "Argentina",
             "adjacent": 1},
            {"origin_id": "Brazil", "destination_id": "Bolivia",
             "adjacent": 1},
            {"origin_id": "Brazil", "destination_id": "China",
             "adjacent": 0},
        ])
        out = classify_coupling(edges, focal_id="Brazil", adjacency=adjacency)
        assert (out["coupling_type"] == "P").all()

    def test_non_adjacent_classifies_as_tele(self):
        edges = pd.DataFrame([
            {"origin_id": "Brazil", "destination_id": "China"},
            {"origin_id": "Brazil", "destination_id": "Japan"},
        ])
        adjacency = pd.DataFrame([
            {"origin_id": "Brazil", "destination_id": "Argentina",
             "adjacent": 1},
            # China + Japan not listed -> not adjacent.
        ])
        out = classify_coupling(edges, focal_id="Brazil", adjacency=adjacency)
        assert (out["coupling_type"] == "T").all()

    def test_missing_adjacency_raises_for_cross_system_edges(self):
        """When edges include cross-system flows but no adjacency
        table is provided, raise ValueError per spec §13."""
        edges = pd.DataFrame([
            {"origin_id": "Brazil", "destination_id": "China"},
        ])
        with pytest.raises(ValueError, match="adjacency"):
            classify_coupling(edges, focal_id="Brazil", adjacency=None)

    def test_na_adjacency_flag_is_not_treated_as_adjacent(self):
        """A <NA> flag (e.g. from build_adjacency) means 'unknown' and
        must NOT be read as adjacent.  bool(NaN) is True in Python, so
        classify_coupling guards it explicitly -> the edge is T, not P."""
        edges = pd.DataFrame([
            {"origin_id": "Brazil", "destination_id": "EU"},
        ])
        adjacency = pd.DataFrame({
            "origin_id": ["Brazil"],
            "destination_id": ["EU"],
            "adjacent": pd.array([pd.NA], dtype="Int64"),
        })
        out = classify_coupling(edges, focal_id="Brazil", adjacency=adjacency)
        assert out.iloc[0]["coupling_type"] == "T"


# ---------------------------------------------------------------------------
# TestBuildAdjacency (14 cases)
# ---------------------------------------------------------------------------


def _flag(adj: "pd.DataFrame", dest: str):
    """Pull the single `adjacent` value for a destination row."""
    return adj.loc[adj["destination_id"] == dest, "adjacent"].iloc[0]


class TestBuildAdjacency:
    def test_adm0_country_names(self):
        pairs = pd.DataFrame({
            "origin_id": ["Brazil", "Brazil"],
            "destination_id": ["Argentina", "China"],
        })
        adj = build_adjacency(pairs, level="adm0")
        assert _flag(adj, "Argentina") == 1   # shared border -> pericoupled
        assert _flag(adj, "China") == 0        # distant -> telecoupled

    def test_adm0_iso_codes(self):
        pairs = pd.DataFrame({
            "origin_id": ["MEX"],
            "destination_id": ["USA"],
        })
        adj = build_adjacency(pairs, level="adm0")
        assert _flag(adj, "USA") == 1

    def test_adm0_unknown_id_is_na_and_warns(self):
        pairs = pd.DataFrame({
            "origin_id": ["Brazil"],
            "destination_id": ["EU"],   # a bloc, not a country in the DB
        })
        with pytest.warns(UserWarning, match="could not be resolved"):
            adj = build_adjacency(pairs, level="adm0")
        assert pd.isna(_flag(adj, "EU"))

    def test_self_pair_dropped(self):
        pairs = pd.DataFrame({
            "origin_id": ["Brazil", "Brazil"],
            "destination_id": ["Brazil", "Argentina"],
        })
        adj = build_adjacency(pairs, level="adm0")
        assert list(adj["destination_id"]) == ["Argentina"]

    def test_duplicate_pairs_collapsed(self):
        pairs = pd.DataFrame({
            "origin_id": ["Brazil", "Brazil", "Brazil"],
            "destination_id": ["Argentina", "Argentina", "China"],
        })
        adj = build_adjacency(pairs, level="adm0")
        assert len(adj) == 2

    def test_adm1_codes(self):
        pairs = pd.DataFrame({
            "origin_id": ["MEX008"],            # Chihuahua
            "destination_id": ["USA044"],       # Texas
        })
        adj = build_adjacency(pairs, level="adm1")
        assert _flag(adj, "USA044") == 1

    def test_adm1_names_and_region_country(self):
        pairs = pd.DataFrame({
            "origin_id": ["Chihuahua", "Chihuahua, Mexico"],
            "destination_id": ["Texas, United States", "Florida, United States"],
        })
        adj = build_adjacency(pairs, level="adm1")
        assert _flag(adj, "Texas, United States") == 1     # adjacent
        assert _flag(adj, "Florida, United States") == 0   # not adjacent

    def test_adm1_unresolved_name_is_na_and_warns(self):
        pairs = pd.DataFrame({
            "origin_id": ["Chihuahua, Mexico"],
            "destination_id": ["Atlantis"],
        })
        with pytest.warns(UserWarning, match="could not be resolved"):
            adj = build_adjacency(pairs, level="adm1")
        assert pd.isna(_flag(adj, "Atlantis"))

    def test_custom_column_names(self):
        pairs = pd.DataFrame({
            "from": ["Brazil"],
            "to": ["Argentina"],
        })
        adj = build_adjacency(
            pairs, level="adm0", origin_col="from", destination_col="to",
            flag_col="is_adj",
        )
        assert list(adj.columns) == ["from", "to", "is_adj"]
        assert adj["is_adj"].iloc[0] == 1

    def test_de_facto_borders_passthrough(self):
        pairs = pd.DataFrame({
            "origin_id": ["China"],
            "destination_id": ["Pakistan"],
        })
        on = build_adjacency(pairs, level="adm0", de_facto_borders=True)
        off = build_adjacency(pairs, level="adm0", de_facto_borders=False)
        assert _flag(on, "Pakistan") == 1    # disputed land folded in
        assert _flag(off, "Pakistan") == 0   # strict standard layer

    def test_coupling_standard_passthrough(self):
        # BRA004 <-> PER016: a no-bridge river pair -> telecoupled under
        # moderate, pericoupled under lenient.
        pairs = pd.DataFrame({
            "origin_id": ["BRA004"],
            "destination_id": ["PER016"],
        })
        moderate = build_adjacency(pairs, level="adm1", coupling_standard="moderate")
        lenient = build_adjacency(pairs, level="adm1", coupling_standard="lenient")
        assert _flag(moderate, "PER016") == 0
        assert _flag(lenient, "PER016") == 1

    def test_invalid_level_raises(self):
        pairs = pd.DataFrame({"origin_id": ["A"], "destination_id": ["B"]})
        with pytest.raises(ValueError, match="adm0.*adm1"):
            build_adjacency(pairs, level="county")

    def test_missing_columns_raises(self):
        pairs = pd.DataFrame({"origin_id": ["A"], "dest": ["B"]})
        with pytest.raises(KeyError, match="destination_id"):
            build_adjacency(pairs, level="adm0")

    def test_feeds_classify_coupling_end_to_end(self):
        """build_adjacency output plugs straight into classify_coupling and
        reproduces the same I/P/T the hand-authored adjacency would."""
        edges = pd.DataFrame([
            {"focal_system_id": "Brazil", "origin_id": "Brazil",
             "destination_id": "Brazil", "flow_value": 10},
            {"focal_system_id": "Brazil", "origin_id": "Brazil",
             "destination_id": "Argentina", "flow_value": 20},
            {"focal_system_id": "Brazil", "origin_id": "Brazil",
             "destination_id": "China", "flow_value": 70},
        ])
        adj = build_adjacency(edges, level="adm0")
        out = classify_coupling(edges, focal_id="Brazil", adjacency=adj)
        by_dest = out.set_index("destination_id")["coupling_type"]
        assert by_dest["Brazil"] == "I"
        assert by_dest["Argentina"] == "P"
        assert by_dest["China"] == "T"

    def test_adm1_same_region_two_forms_dropped(self):
        """Regression: the same ADM1 region written two ways (code vs
        'Region, Country') is a self-pair -> dropped, not flagged 0.
        Both resolve to MEX008; is_adm1_pericoupled maps same-region to
        False, which would otherwise be emitted as adjacent=0."""
        pairs = pd.DataFrame({
            "origin_id": ["MEX008"],
            "destination_id": ["Chihuahua, Mexico"],
        })
        adj = build_adjacency(pairs, level="adm1")
        assert len(adj) == 0

    def test_adm0_same_country_two_forms_dropped(self):
        """Regression: the same country written two ways (name vs ISO) is a
        self-pair -> dropped silently, not a spurious <NA> + warning."""
        pairs = pd.DataFrame({
            "origin_id": ["Mexico"],
            "destination_id": ["MEX"],
        })
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            adj = build_adjacency(pairs, level="adm0")
        assert len(adj) == 0
        assert len(rec) == 0     # not reported as "unresolved"

    def test_flag_column_is_nullable_int64(self):
        """The flag dtype is load-bearing (lets 1/0/<NA> coexist and feeds
        classify_coupling's pd.isna guard) -- pin it explicitly so a float /
        object regression can't pass via numeric coercion."""
        pairs = pd.DataFrame({
            "origin_id": ["Brazil", "Brazil", "Brazil"],
            "destination_id": ["Argentina", "China", "EU"],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adj = build_adjacency(pairs, level="adm0")
        assert str(adj["adjacent"].dtype) == "Int64"

    def test_single_aggregated_warning_lists_all_unresolved(self):
        """All unresolved pairs are reported in exactly ONE UserWarning."""
        pairs = pd.DataFrame({
            "origin_id": ["Brazil", "Brazil"],
            "destination_id": ["EU", "Atlantis"],
        })
        with pytest.warns(UserWarning) as rec:
            adj = build_adjacency(pairs, level="adm0")
        assert len(rec) == 1
        msg = str(rec[0].message)
        assert "EU" in msg and "Atlantis" in msg
        assert pd.isna(_flag(adj, "EU")) and pd.isna(_flag(adj, "Atlantis"))

    def test_warning_truncates_after_ten_unresolved(self):
        """>10 unresolved pairs -> the single message truncates with
        '(and N more)' rather than listing all of them."""
        pairs = pd.DataFrame({
            "origin_id": ["Brazil"] * 12,
            "destination_id": [f"Atlantis{i}" for i in range(12)],
        })
        with pytest.warns(UserWarning) as rec:
            build_adjacency(pairs, level="adm0")
        assert len(rec) == 1
        assert "(and 2 more)" in str(rec[0].message)

    def test_null_cells_dropped(self):
        """Rows with a null origin or destination are dropped (not coerced
        to a 'nan' string or kept as <NA>)."""
        pairs = pd.DataFrame({
            "origin_id": ["Brazil", None, "Brazil"],
            "destination_id": ["Argentina", "China", np.nan],
        })
        adj = build_adjacency(pairs, level="adm0")
        assert list(adj["destination_id"]) == ["Argentina"]
        assert "nan" not in [str(x) for x in adj["origin_id"]]


# ---------------------------------------------------------------------------
# TestSummarizeMetacoupling (2 cases)
# ---------------------------------------------------------------------------


class TestSummarizeMetacoupling:
    def test_brazil_soybean_end_to_end(self):
        """End-to-end: spec §11 worked example reproduces all key
        values in one summary table."""
        edges = _brazil_soybean_edges()
        with _suppress_user_warnings():
            warnings.simplefilter("ignore")
            out = summarize_metacoupling(edges)
        row = out.iloc[0]
        # Flow shares
        assert row["IFS"] == pytest.approx(0.10)
        assert row["PFS"] == pytest.approx(0.20)
        assert row["TFS"] == pytest.approx(0.70)
        # MFE
        assert row["MFE"] == pytest.approx(0.7298, abs=0.001)
        # TFCI
        assert row["TFCI"] == pytest.approx(0.617, abs=0.005)
        # n_T = 3, IFCI = 1 (single self-loop partner)
        assert row["n_T"] == 3
        assert row["IFCI"] == pytest.approx(1.0)

    def test_grouped_by_year(self):
        """Time-series support via group_cols=[\"year\"]."""
        edges = pd.DataFrame([
            {"focal_system_id": "A", "destination_id": "A",
             "coupling_type": "I", "flow_value": 10, "year": 2020},
            {"focal_system_id": "A", "destination_id": "X",
             "coupling_type": "T", "flow_value": 90, "year": 2020},
            {"focal_system_id": "A", "destination_id": "A",
             "coupling_type": "I", "flow_value": 50, "year": 2021},
            {"focal_system_id": "A", "destination_id": "X",
             "coupling_type": "T", "flow_value": 50, "year": 2021},
        ])
        with _suppress_user_warnings():
            warnings.simplefilter("ignore")
            out = summarize_metacoupling(edges, group_cols=["year"])
        # One row per (system, year).
        assert len(out) == 2
        by_year = out.set_index("year")
        assert by_year.loc[2020, "IFS"] == pytest.approx(0.10)
        assert by_year.loc[2021, "IFS"] == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# TestPandasOptional (1 case)
# ---------------------------------------------------------------------------


class TestPandasOptional:
    def test_importerror_message_contains_install_hint(self, monkeypatch):
        """When pandas import fails, _require_pandas should raise an
        ImportError whose message points users to the install command."""
        import builtins

        from metacouplingllm.indicators import core

        original_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("simulated missing pandas")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        with pytest.raises(ImportError, match=r"pip install metacouplingllm\[indicators\]"):
            core._require_pandas()

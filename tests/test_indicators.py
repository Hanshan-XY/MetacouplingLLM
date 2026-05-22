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
        -> HHI≈0.745, MFCI≈0.617 with m=3."""
        edges = _brazil_soybean_edges()
        with _suppress_user_warnings():
            warnings.simplefilter("ignore")
            out = compute_mfci(edges)
        # T row for Brazil
        tele = out.loc[
            (out["focal_system_id"] == "Brazil")
            & (out["coupling_type"] == "T")
        ].iloc[0]
        assert tele["m_partners"] == 3
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
        assert t_row["m_partners"] == 1
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
        # m_T = 3, IFCI = 1 (single self-loop partner)
        assert row["m_T"] == 3
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

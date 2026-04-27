"""Tests for knowledge/pericoupling.py — pericoupling database lookup."""

from metacouplingllm.knowledge.pericoupling import (
    PairCouplingType,
    PericouplingResult,
    is_pericoupled,
    lookup_pericoupling,
    _get_pairs,
)


class TestPericoupledPairsLoading:
    """Verify that the CSV database is loaded correctly."""

    def test_pairs_loaded(self):
        pairs = _get_pairs()
        assert len(pairs) > 0, "No pericoupled pairs loaded from CSV"

    def test_pairs_are_frozensets(self):
        pairs = _get_pairs()
        for pair in pairs:
            assert isinstance(pair, frozenset)
            assert len(pair) == 2

    def test_known_pericoupled_pair_present(self):
        """MEX-USA should be in the dataset."""
        pairs = _get_pairs()
        assert frozenset({"MEX", "USA"}) in pairs

    def test_known_telecoupled_pair_absent(self):
        """MEX-CAN should NOT be in the pericoupled set."""
        pairs = _get_pairs()
        assert frozenset({"MEX", "CAN"}) not in pairs


class TestLookupPericoupling:
    """Test the lookup_pericoupling function."""

    def test_pericoupled_by_code(self):
        result = lookup_pericoupling("MEX", "USA")
        assert result.pair_type == PairCouplingType.PERICOUPLED
        assert result.sending_code == "MEX"
        assert result.receiving_code == "USA"
        assert result.confidence == "database"

    def test_pericoupled_by_name(self):
        result = lookup_pericoupling("Mexico", "United States")
        assert result.pair_type == PairCouplingType.PERICOUPLED
        assert result.sending_code == "MEX"
        assert result.receiving_code == "USA"

    def test_telecoupled_by_code(self):
        result = lookup_pericoupling("MEX", "CAN")
        assert result.pair_type == PairCouplingType.TELECOUPLED
        assert result.sending_code == "MEX"
        assert result.receiving_code == "CAN"
        assert result.confidence == "database"

    def test_telecoupled_by_name(self):
        result = lookup_pericoupling("Brazil", "China")
        assert result.pair_type == PairCouplingType.TELECOUPLED
        assert result.sending_code == "BRA"
        assert result.receiving_code == "CHN"

    def test_symmetric_lookup(self):
        """Order should not matter — MEX-USA == USA-MEX."""
        r1 = lookup_pericoupling("MEX", "USA")
        r2 = lookup_pericoupling("USA", "MEX")
        assert r1.pair_type == r2.pair_type == PairCouplingType.PERICOUPLED

    def test_can_usa_pericoupled(self):
        result = lookup_pericoupling("Canada", "United States")
        assert result.pair_type == PairCouplingType.PERICOUPLED

    def test_mex_gtm_pericoupled(self):
        result = lookup_pericoupling("Mexico", "Guatemala")
        assert result.pair_type == PairCouplingType.PERICOUPLED

    def test_unresolved_country(self):
        result = lookup_pericoupling("Atlantis", "USA")
        assert result.pair_type == PairCouplingType.UNKNOWN
        assert result.confidence == "unresolved"
        assert result.sending_code is None
        assert result.receiving_code == "USA"

    def test_same_country(self):
        result = lookup_pericoupling("USA", "United States")
        assert result.pair_type == PairCouplingType.UNKNOWN
        assert result.confidence == "same_country"

    def test_result_is_frozen_dataclass(self):
        result = lookup_pericoupling("MEX", "USA")
        assert isinstance(result, PericouplingResult)


class TestIsPericoupled:
    """Test the is_pericoupled convenience function."""

    def test_true_for_pericoupled(self):
        assert is_pericoupled("Mexico", "United States") is True

    def test_false_for_telecoupled(self):
        assert is_pericoupled("Brazil", "China") is False

    def test_none_for_unknown(self):
        assert is_pericoupled("Atlantis", "USA") is None

    def test_mex_can_telecoupled(self):
        """Mexico and Canada are NOT adjacent — telecoupled."""
        assert is_pericoupled("Mexico", "Canada") is False

    def test_can_usa_pericoupled(self):
        assert is_pericoupled("Canada", "USA") is True


class TestAvocadoTradeScenario:
    """The user's primary test case: avocado trade MEX-USA-CAN."""

    def test_mexico_us_pericoupled(self):
        assert is_pericoupled("Mexico", "United States") is True

    def test_mexico_canada_telecoupled(self):
        assert is_pericoupled("Mexico", "Canada") is False

    def test_canada_us_pericoupled(self):
        assert is_pericoupled("Canada", "United States") is True

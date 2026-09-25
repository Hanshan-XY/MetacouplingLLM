"""Guard tests for the geometry build script (scripts/build_pericoupling_db.py).

That module is executed manually (via ``scripts/build_all.py``) and is never
imported by the package or the rest of the suite, so a syntax or import error
in it reaches ``main()`` completely undetected.  That is exactly what happened
when PR #81's squash merged a broken ``_ADM1_FALSE_POSITIVE_DENYLIST`` block (a
stray ``})`` plus reverted N=4 entries -- a genuine ``SyntaxError``) into the
script: it hit main because no test parsed the file.

These are the minimal, fast guards that would have caught it, and no more:

  * ``test_module_parses``   -- the file must parse.  Pure ``ast.parse`` with no
    third-party import, so it fires even in a bare environment.
  * ``test_denylist_is_exactly_the_reviewed_pairs`` -- the reviewed denylist
    must stay exactly the maintainer-decided pairs.  A regression, or a
    duplicated/broken block, fails here.  It imports the module, so it also
    catches import-time errors that ``ast.parse`` cannot see.

No gpkg build -- importing runs only the module-level definitions (``main()``
is behind ``if __name__ == "__main__"``), so both tests are cheap.
"""
import ast
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
BUILD_SCRIPT = SCRIPTS / "build_pericoupling_db.py"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBuildPericouplingDb:
    def test_module_parses(self):
        """A SyntaxError in the build script (cf. PR #81) must fail here.

        Dependency-free: ``ast.parse`` needs no geo stack, so this is the true
        minimal guard -- it would have failed the moment PR #81's broken
        denylist block landed, in any environment.
        """
        source = BUILD_SCRIPT.read_text(encoding="utf-8")
        ast.parse(source, filename=str(BUILD_SCRIPT))

    def test_denylist_is_exactly_the_reviewed_pairs(self):
        """The reviewed ADM1 false-positive denylist is exactly the five
        maintainer-decided pairs (docs/FUTURE_EDGE_AUDITS.md; decisions
        2026-07-18 for the first two, 2026-07-25 wu1 map rulings for the three
        mid-lake contacts), and the reviewed unit-merge map is exactly
        RUS050->RUS024.

        Importing the module also surfaces any top-level import error (which
        ``ast.parse`` cannot), and pins the values so a future regression or
        a duplicated/broken block -- the PR #81 failure class -- is caught.
        """
        mod = _load("build_pericoupling_db")
        assert mod._ADM1_FALSE_POSITIVE_DENYLIST == {
            frozenset({"LBR006", "LBR014"}),
            frozenset({"VEN001", "VEN003"}),
            frozenset({"CAN003", "CAN006"}),
            frozenset({"COD009", "UGA102"}),
            frozenset({"TZA016", "UGA040"}),
        }
        assert mod._ADM1_UNIT_MERGES == {"RUS050": "RUS024"}

    def test_tract_administrators_follow_natural_earth(self):
        """Each disputed-area tract is assigned whole to the administrator
        Natural Earth v5.1.2 records for most of it (campaign da1, maintainer
        rulings 2026-09-25; build_data/ndlsa_ne/SPEC_da1_defacto_administrators.md):
        only the UN Buffer Zone stays unassigned, and every assigned tract has a
        province list (empty = country level only)."""
        mod = _load("build_pericoupling_db")
        admin = mod._NDLSA_TRACT_ADMIN
        assert len(admin) == 24
        assert [t for t, a in admin.items() if a is None] == ["UN Buffer Zone"]
        assert admin["Karakoram Range"] == "CHN"          # the Shaksgam Valley
        assert admin["Kauirik"] == "CHN"                  # majority 57%
        assert admin["Lapthal"] == "IND" and admin["Shipki Pass"] == "IND"
        assert admin["No Man's Land"] == "ISR" and admin["Abyei"] == "SDN"
        assert admin["Western Sahara"] == "MAR"           # whole tract, no exception
        for island in ("British Indian Ocean Territory",
                       "South Georgia and South Sandwich Islands", "Falkland Islands"):
            assert admin[island] == "GBR"
            assert mod._NDLSA_TRACT_ADM1[island] == []
        assert {t for t, a in admin.items() if a is not None} == set(mod._NDLSA_TRACT_ADM1)

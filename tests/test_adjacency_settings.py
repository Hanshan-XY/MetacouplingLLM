"""The assistant's adjacency settings reach the prompt hints and the maps.

``MetacouplingAssistant(de_facto_borders=..., coupling_standard=...)``
passes both settings to every adjacency lookup it makes: the
pericoupling hint in the system prompt, the post-LLM validation and the
automatic map.  The defaults are the lookups' own, so with the default
settings the prompt and the maps show the lookups' default adjacency.
"""

from __future__ import annotations

import re

import pytest

from metacouplingllm.core import MetacouplingAssistant
from metacouplingllm.knowledge.adm1_pericoupling import is_adm1_pericoupled
from metacouplingllm.llm.client import LLMResponse
from metacouplingllm.prompts.builder import PromptBuilder

from ._helpers import make_parsed_analysis

try:
    import geopandas  # noqa: F401
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    _viz_available = True
except ImportError:
    _viz_available = False

skip_no_viz = pytest.mark.skipif(
    not _viz_available,
    reason="geopandas/matplotlib not installed",
)

STANDARDS = ("stringent", "moderate", "lenient")

# Ohio and Ontario meet only across Lake Erie, with no fixed crossing, so
# they are adjacent under "lenient" only.
OHIO_ACID_RAIN = (
    "Acid rain from coal-fired power plants in Ohio, United States, "
    "falling on lakes in Ontario, Canada."
)
ONTARIO = "Ontario (CAN009, Canada)"

# Romania and Moldova share only rivers, with bridges, so they are
# adjacent under "moderate" and "lenient" but not under "stringent".
ROMANIA_MOLDOVA = "Timber exports from Romania to Moldova"

# China and Pakistan meet only across disputed land administered by
# Pakistan, so they are adjacent only in the de-facto view.
CHINA_PAKISTAN = "Trade from China to Pakistan"

# An analysis of the Ohio description, in the coupling-first layout.
OHIO_ANALYSIS = """\
### 1. Coupling Classification
Pericoupling: sulfur emissions from Ohio fall as acid rain on Ontario.

### 3. Pericoupling Analysis
#### 3.1 Systems Identification
**Sending System (adjacent)**: Ohio Coal Power System
- **Geographic scope**: Ohio, United States
**Receiving System (adjacent)**: Ontario Lake Ecosystems
- **Geographic scope**: Ontario, Canada
"""

_ADM1_CODE_RE = re.compile(r"\(([A-Z]{3}\d{3})[,)]")


class _AnalysisClient:
    """Answers every call with the Ohio analysis."""

    def chat(self, messages, temperature=0.7, max_tokens=None):
        return LLMResponse(content=OHIO_ANALYSIS)


def _system_prompt(client, description: str, **settings) -> str:
    """Return the system prompt an assistant sends for *description*."""
    advisor = MetacouplingAssistant(
        client, generate_abstract=False, **settings,
    )
    advisor.analyze(description)
    return advisor.history[0].content


def _hint_adm1_codes(system_prompt: str) -> set[str]:
    """Return the ADM1 codes the system prompt's ADM1 hint lists."""
    _, found, rest = system_prompt.partition("## ADM1 ADJACENCY DATABASE")
    assert found, "no ADM1 hint in the system prompt"
    hint = rest.split("\n## ", 1)[0]
    codes: set[str] = set()
    for line in hint.splitlines():
        match = _ADM1_CODE_RE.search(line)
        if line.startswith("- ") and match:
            codes.add(match.group(1))
    return codes


# ---------------------------------------------------------------------------
# ADM1 hint
# ---------------------------------------------------------------------------


class TestAdm1HintFollowsSettings:
    def test_ohio_and_ontario_are_adjacent_only_under_lenient(self):
        """The premise of the Ohio tests, read from the database."""
        assert {
            s: is_adm1_pericoupled("USA036", "CAN009", coupling_standard=s)
            for s in STANDARDS
        } == {"stringent": False, "moderate": False, "lenient": True}

    @pytest.mark.parametrize("standard", STANDARDS)
    def test_ohio_hint_lists_ontario_only_under_lenient(self, standard):
        hint = PromptBuilder._build_adm1_pericoupling_hint(
            OHIO_ACID_RAIN, coupling_standard=standard,
        )
        assert hint is not None
        assert "**Ohio** (USA036, United States of America)" in hint
        lenient = standard == "lenient"
        assert (ONTARIO in hint) is lenient
        assert ("**Cross-border (pericoupled) neighbors:**" in hint) is lenient

    def test_stringent_drops_ohio_river_neighbors(self):
        """Kentucky and West Virginia lie across the Ohio River, which is
        bridged: listed under "moderate", not under "stringent"."""
        moderate = PromptBuilder._build_adm1_pericoupling_hint(
            OHIO_ACID_RAIN, coupling_standard="moderate",
        )
        stringent = PromptBuilder._build_adm1_pericoupling_hint(
            OHIO_ACID_RAIN, coupling_standard="stringent",
        )
        for neighbor in ("Kentucky (USA018", "West Virginia (USA049"):
            assert neighbor in moderate
            assert neighbor not in stringent
        assert "Indiana (USA015" in stringent

    @pytest.mark.parametrize("de_facto", [True, False])
    def test_de_facto_borders_reaches_adm1_hint(self, de_facto):
        """Arunachal Pradesh meets Xizang only in the de-facto view."""
        hint = PromptBuilder._build_adm1_pericoupling_hint(
            "Hydropower dams in Arunachal Pradesh, India",
            de_facto_borders=de_facto,
        )
        assert hint is not None
        assert "**Arunachal Pradesh** (IND003, India)" in hint
        assert ("Xizang Zizhiqu (CHN029, China)" in hint) is de_facto

    def test_system_prompt_uses_the_builder_settings(self):
        lenient = PromptBuilder(max_examples=0, coupling_standard="lenient")
        moderate = PromptBuilder(max_examples=0, coupling_standard="moderate")
        assert ONTARIO in lenient.build_system_prompt(
            research_context=OHIO_ACID_RAIN,
        )
        assert ONTARIO not in moderate.build_system_prompt(
            research_context=OHIO_ACID_RAIN,
        )

    def test_builder_rejects_unknown_standard(self):
        with pytest.raises(ValueError, match="coupling_standard"):
            PromptBuilder(coupling_standard="bogus")


# ---------------------------------------------------------------------------
# Country (ADM0) hint
# ---------------------------------------------------------------------------


class TestCountryHintFollowsSettings:
    @pytest.mark.parametrize(
        "standard, label",
        [
            ("stringent", "telecoupled"),
            ("moderate", "pericoupled"),
            ("lenient", "pericoupled"),
        ],
    )
    def test_pair_label_follows_standard(self, standard, label):
        hint = PromptBuilder._build_coupling_hint(
            ROMANIA_MOLDOVA, coupling_standard=standard,
        )
        assert hint is not None
        assert f"Romania (ROU) and Moldova (MDA) are **{label}**" in hint

    @pytest.mark.parametrize(
        "de_facto, label", [(True, "pericoupled"), (False, "telecoupled")],
    )
    def test_pair_label_follows_de_facto_borders(self, de_facto, label):
        hint = PromptBuilder._build_coupling_hint(
            CHINA_PAKISTAN, de_facto_borders=de_facto,
        )
        assert hint is not None
        assert f"China (CHN) and Pakistan (PAK) are **{label}**" in hint

    def test_neighbor_list_follows_standard(self):
        """A one-country description lists that country's neighbors."""
        moderate = PromptBuilder._build_coupling_hint(
            "Mining in Moldova", coupling_standard="moderate",
        )
        stringent = PromptBuilder._build_coupling_hint(
            "Mining in Moldova", coupling_standard="stringent",
        )
        assert "- Romania (ROU)" in moderate
        assert "- Romania (ROU)" not in stringent
        assert "- Ukraine (UKR)" in stringent

    def test_system_prompt_uses_the_builder_settings(self):
        builder = PromptBuilder(max_examples=0, coupling_standard="stringent")
        prompt = builder.build_system_prompt(research_context=ROMANIA_MOLDOVA)
        assert "ADJACENCY DATABASE" not in prompt  # no ADM1 hint
        assert "Romania (ROU) and Moldova (MDA) are **telecoupled**" in prompt


# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------


class TestDefaultSettingsLeavePromptUnchanged:
    def test_default_ohio_hint_lists_ontario(self):
        hint = PromptBuilder._build_adm1_pericoupling_hint(OHIO_ACID_RAIN)
        assert hint is not None
        assert hint.endswith(
            "**Same-country (pericoupled) neighbors:**\n"
            "- Indiana (USA015, United States of America)\n"
            "- Kentucky (USA018, United States of America)\n"
            "- Michigan (USA023, United States of America)\n"
            "- Pennsylvania (USA039, United States of America)\n"
            "- West Virginia (USA049, United States of America)\n"
            "\n"
            "**Cross-border (pericoupled) neighbors:**\n"
            "- Ontario (CAN009, Canada)"
        )

    @pytest.mark.parametrize(
        "description",
        [OHIO_ACID_RAIN, ROMANIA_MOLDOVA, CHINA_PAKISTAN, "Mining in Moldova"],
    )
    def test_builder_defaults_are_the_lookup_defaults(self, description):
        default = PromptBuilder(max_examples=2)
        explicit = PromptBuilder(
            max_examples=2, de_facto_borders=True, coupling_standard="lenient",
        )
        assert default.build_system_prompt(
            research_context=description,
        ) == explicit.build_system_prompt(research_context=description)

    @pytest.mark.parametrize(
        "description", [OHIO_ACID_RAIN, ROMANIA_MOLDOVA, CHINA_PAKISTAN],
    )
    def test_default_assistant_sends_the_default_builder_prompt(
        self, mock_llm_client, description,
    ):
        assert _system_prompt(
            mock_llm_client, description,
        ) == PromptBuilder(max_examples=2).build_system_prompt(
            research_context=description,
        )


# ---------------------------------------------------------------------------
# The assistant's prompt
# ---------------------------------------------------------------------------


class TestAssistantPromptFollowsSettings:
    def test_lenient_assistant_hint_lists_ontario(self, mock_llm_client):
        assert ONTARIO in _system_prompt(
            mock_llm_client, OHIO_ACID_RAIN, coupling_standard="lenient",
        )
        assert ONTARIO not in _system_prompt(
            mock_llm_client, OHIO_ACID_RAIN, coupling_standard="moderate",
        )

    def test_stringent_assistant_country_hint(self, mock_llm_client):
        prompt = _system_prompt(
            mock_llm_client, ROMANIA_MOLDOVA, coupling_standard="stringent",
        )
        assert "Romania (ROU) and Moldova (MDA) are **telecoupled**" in prompt

    def test_strict_borders_assistant_country_hint(self, mock_llm_client):
        prompt = _system_prompt(
            mock_llm_client, CHINA_PAKISTAN, de_facto_borders=False,
        )
        assert "China (CHN) and Pakistan (PAK) are **telecoupled**" in prompt

    @pytest.mark.parametrize("standard", STANDARDS)
    def test_hint_and_validation_list_the_same_neighbors(self, standard):
        """Under every standard the system prompt's hint and the
        validation block list the same neighbors of Ohio; before, a
        lenient assistant was told Ontario was no neighbor and then
        validated as if it were."""
        advisor = MetacouplingAssistant(
            _AnalysisClient(),
            generate_abstract=False,
            coupling_standard=standard,
        )
        result = advisor.analyze(OHIO_ACID_RAIN)
        info = result.parsed.pericoupling_info
        assert info is not None
        assert info["focal_region"] == "Ohio (USA036)"
        validated = set(
            _ADM1_CODE_RE.findall(
                info.get("domestic_neighbors", "")
                + info.get("cross_border_neighbors", "")
            )
        )
        hinted = _hint_adm1_codes(advisor.history[0].content)
        assert hinted == validated
        assert ("CAN009" in hinted) is (standard == "lenient")


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

SETTINGS = {"de_facto_borders": False, "coupling_standard": "moderate"}


@pytest.fixture
def map_calls(monkeypatch):
    """Replace both map renderers; record what the assistant passes them."""
    calls: list[tuple[str, dict]] = []

    def fake_country_map(parsed, **kwargs):
        calls.append(("country", kwargs))
        return "country-figure"

    def fake_adm1_map(focal_adm1, **kwargs):
        calls.append(("adm1", kwargs))
        return "adm1-figure"

    monkeypatch.setattr(
        "metacouplingllm.visualization.worldmap.plot_analysis_map",
        fake_country_map,
    )
    monkeypatch.setattr(
        "metacouplingllm.visualization.adm1_map.plot_focal_adm1_map",
        fake_adm1_map,
    )
    return calls


def _ohio_parsed(flows=None):
    return make_parsed_analysis(
        systems={
            "sending": {"name": "Ohio", "geographic_scope": "Ohio, United States"},
            "receiving": {
                "name": "Kentucky",
                "geographic_scope": "Kentucky, United States",
            },
        },
        flows=flows,
    )


def _romania_parsed():
    return make_parsed_analysis(
        systems={
            "sending": {"name": "Romania", "geographic_scope": "Romania"},
            "receiving": {"name": "Moldova", "geographic_scope": "Moldova"},
        },
    )


class TestAssistantMapsFollowSettings:
    def test_structured_adm1_map(self, map_calls):
        advisor = MetacouplingAssistant(
            _AnalysisClient(), auto_map=True, **SETTINGS,
        )
        parsed = _ohio_parsed()
        parsed.map_data = {
            "focal_country": "USA",
            "adm1_region": "USA036",
            "receiving_countries": ["CAN"],
            "spillover_countries": [],
            "flows": [],
        }
        assert advisor._generate_map(parsed) == "adm1-figure"
        (kind, kwargs), = map_calls
        assert kind == "adm1"
        assert kwargs["de_facto_borders"] is False
        assert kwargs["coupling_standard"] == "moderate"

    def test_structured_country_map(self, map_calls):
        advisor = MetacouplingAssistant(
            _AnalysisClient(), auto_map=True, **SETTINGS,
        )
        parsed = _romania_parsed()
        parsed.map_data = {
            "focal_country": "ROU",
            "adm1_region": None,
            "receiving_countries": ["MDA"],
            "spillover_countries": [],
            "flows": [],
        }
        assert advisor._generate_map(parsed) == "country-figure"
        (kind, kwargs), = map_calls
        assert kind == "country"
        assert kwargs["de_facto_borders"] is False
        assert kwargs["coupling_standard"] == "moderate"

    def test_text_scan_adm1_map(self, map_calls):
        """Without map data the map falls back to a text scan."""
        advisor = MetacouplingAssistant(
            _AnalysisClient(), auto_map=True, **SETTINGS,
        )
        assert advisor._generate_map(_ohio_parsed()) == "adm1-figure"
        (kind, kwargs), = map_calls
        assert kind == "adm1"
        assert kwargs["de_facto_borders"] is False
        assert kwargs["coupling_standard"] == "moderate"

    def test_text_scan_country_map(self, map_calls):
        advisor = MetacouplingAssistant(
            _AnalysisClient(), auto_map=True, **SETTINGS,
        )
        assert advisor._generate_map(_romania_parsed()) == "country-figure"
        (kind, kwargs), = map_calls
        assert kind == "country"
        assert kwargs["de_facto_borders"] is False
        assert kwargs["coupling_standard"] == "moderate"

    @pytest.mark.parametrize(
        "standard, arrow", [("moderate", True), ("stringent", False)],
    )
    def test_adm1_flow_arrows_follow_standard(self, map_calls, standard, arrow):
        """An Ohio -> Kentucky flow is drawn between the two states only
        while they count as neighbors (the Ohio River is bridged)."""
        advisor = MetacouplingAssistant(
            _AnalysisClient(), auto_map=True, coupling_standard=standard,
        )
        parsed = _ohio_parsed(
            flows=[
                {
                    "category": "Matter",
                    "direction": "Ohio → Kentucky",
                    "description": "Coal shipped across the Ohio River",
                },
            ],
        )
        advisor._generate_map(parsed)
        (kind, kwargs), = map_calls
        assert kind == "adm1"
        targets = {
            flow.get("target_adm1") for flow in (kwargs["flows"] or [])
        }
        assert ("USA018" in targets) is arrow


@skip_no_viz
class TestMapFunctionsFollowSettings:
    """The map functions color neighbors under the settings they get.

    The basemap loaders are replaced by small tables and the renderers by
    functions that return the classification instead of drawing it.
    """

    @pytest.fixture
    def country_world(self, monkeypatch):
        import pandas as pd

        from metacouplingllm.visualization import worldmap

        world = pd.DataFrame({"iso_code": ["MDA", "ROU", "UKR", "FRA"]})
        monkeypatch.setattr(
            worldmap, "_get_world_geodataframe", lambda **kwargs: world,
        )
        monkeypatch.setattr(
            worldmap,
            "_render_map",
            lambda world, classification, *args, **kwargs: classification,
        )

    @pytest.fixture
    def adm1_world(self, monkeypatch):
        import pandas as pd

        from metacouplingllm.visualization import adm1_map

        world = pd.DataFrame(
            {
                "adm1_code": ["USA036", "USA015", "USA018", "CAN009"],
                "iso_code": ["USA", "USA", "USA", "CAN"],
            }
        )
        monkeypatch.setattr(
            adm1_map, "_get_adm1_geodataframe", lambda **kwargs: world,
        )
        monkeypatch.setattr(
            adm1_map,
            "_render_adm1_map",
            lambda world, classification, *args, **kwargs: (
                classification, kwargs["adm1_detail_countries"],
            ),
        )

    @pytest.mark.parametrize(
        "standard, romania",
        [("moderate", "pericoupling"), ("stringent", "telecoupling")],
    )
    def test_focal_country_map(self, country_world, standard, romania):
        from metacouplingllm.visualization.worldmap import (
            plot_focal_country_map,
        )

        classification = plot_focal_country_map(
            "Moldova", coupling_standard=standard,
        )
        assert classification["ROU"] == romania
        assert classification["UKR"] == "pericoupling"

    @pytest.mark.parametrize(
        "standard, romania",
        [("moderate", "pericoupling"), ("stringent", "telecoupling")],
    )
    def test_analysis_map(self, country_world, standard, romania):
        from metacouplingllm.visualization.worldmap import plot_analysis_map

        classification = plot_analysis_map(
            _romania_parsed(),
            focal_code_override="MDA",
            mentioned_countries_override={"MDA", "ROU"},
            flows=[],
            coupling_standard=standard,
        )
        assert classification["ROU"] == romania
        assert classification["FRA"] == "na"

    @pytest.mark.parametrize(
        "standard, ontario",
        [("lenient", "pericoupling"), ("moderate", "telecoupling")],
    )
    def test_focal_adm1_map(self, adm1_world, standard, ontario):
        from metacouplingllm.visualization.adm1_map import plot_focal_adm1_map

        classification, detail_countries = plot_focal_adm1_map(
            "USA036", coupling_standard=standard,
        )
        assert classification["CAN009"] == ontario
        # Canada keeps ADM1 detail only when it holds a neighbor.
        assert ("CAN" in detail_countries) is (standard == "lenient")

    def test_visualization_wrappers_forward_settings(self, monkeypatch):
        from metacouplingllm import visualization
        from metacouplingllm.visualization import adm1_map, worldmap

        captured: dict[str, dict] = {}

        def _recorder(name):
            def record(*args, **kwargs):
                captured[name] = kwargs

            return record

        monkeypatch.setattr(
            worldmap, "plot_focal_country_map", _recorder("country"),
        )
        monkeypatch.setattr(worldmap, "plot_analysis_map", _recorder("analysis"))
        monkeypatch.setattr(adm1_map, "plot_focal_adm1_map", _recorder("adm1"))

        visualization.plot_focal_country_map("Moldova", **SETTINGS)
        visualization.plot_analysis_map(_romania_parsed(), **SETTINGS)
        visualization.plot_focal_adm1_map("USA036", **SETTINGS)

        assert set(captured) == {"country", "analysis", "adm1"}
        for kwargs in captured.values():
            assert kwargs["de_facto_borders"] is False
            assert kwargs["coupling_standard"] == "moderate"

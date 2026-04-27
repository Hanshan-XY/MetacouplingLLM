"""Integration tests — full pipeline from advisor to formatted output."""

from metacouplingllm import (
    AgentLevel,
    AnalysisResult,
    AnthropicAdapter,
    CauseCategory,
    CouplingType,
    FlowCategory,
    FrameworkComponent,
    LLMClient,
    LLMResponse,
    Message,
    MetacouplingAssistant,
    OpenAIAdapter,
    SystemRole,
)


# A comprehensive mock response simulating a real LLM analysis.
FULL_MOCK_RESPONSE = """\
### 1. Coupling Classification

This research involves **telecoupling** between distant lithium-producing \
and electric vehicle (EV) manufacturing systems. The extraction of lithium \
in South America and its use in EV batteries in Europe creates \
socioeconomic-environmental interactions across vast distances.

### 2. Systems Identification

- **Sending**: Lithium mining regions in Chile and Argentina (Lithium \
Triangle). Human: mining companies, local communities, government agencies. \
Natural: salt flats (salares), freshwater resources, Andean ecosystems.

- **Receiving**: European EV manufacturing hubs (Germany, France). Human: \
auto manufacturers, battery producers, consumers. Natural: urban-industrial \
landscapes, waste management systems.

- **Spillover**: Australia (competing lithium producer affected by market \
shifts), Bolivia (adjacent Lithium Triangle country), downstream communities \
along water systems.

### 3. Flows Analysis

- [Matter] Chile/Argentina → Europe: Lithium carbonate and lithium \
hydroxide shipped for battery manufacturing
- [Capital] Europe → Chile/Argentina: Payment for lithium; foreign \
direct investment in mining operations
- [Information] Bidirectional: Market data, environmental regulations, \
mining technology innovations
- [Energy] Chile/Argentina → Europe: Embodied energy in extracted and \
processed lithium products

### 4. Agents

- Lithium mining corporations (SQM, Albemarle) in sending systems
- European auto manufacturers (VW, BMW, Stellantis) in receiving systems
- Chilean and Argentine government mining agencies (sending)
- European Union trade and environmental regulators (receiving)
- Indigenous communities near salt flats (sending/spillover)
- International commodity traders (intermediary)

### 5. Causes

**Proximate causes**
- Rapid growth of EV market in Europe driven by climate policy
- EU mandates for EV adoption and battery production

**Underlying causes**
- Global climate change mitigation efforts and Paris Agreement commitments
- Technological advances in lithium-ion battery chemistry
- European industrial policy to secure critical mineral supply chains

### 6. Effects

**Sending system**
- [Socioeconomic] Mining revenue and employment; but community displacement \
and water rights conflicts
- [Environmental] Water depletion in arid salar regions; ecosystem \
disruption; dust and contamination

**Receiving system**
- [Socioeconomic] Green industry jobs; reduced oil dependency; consumer \
access to EVs
- [Environmental] Reduced transport emissions; but battery waste management \
challenges

**Spillover**
- [Socioeconomic] Market competition affecting Australian lithium miners; \
geopolitical tensions over critical minerals
- [Environmental] Potential mining expansion into Bolivia's Salar de Uyuni

### 7. Research Gaps and Suggestions

- Quantify water footprint of lithium extraction relative to local supplies
- Assess battery recycling telecouplings (end-of-life flows back to \
processing centers)
- Map full supply chain including intermediate processing in China
- Investigate Indigenous community agency in telecoupling governance
- Compare with alternative battery chemistries (sodium-ion) and their \
telecoupling implications
"""


class MockFullClient:
    """Mock client returning a comprehensive response."""

    def chat(self, messages, temperature=0.7, max_tokens=None):
        return LLMResponse(
            content=FULL_MOCK_RESPONSE,
            usage={"prompt_tokens": 3000, "completion_tokens": 800},
        )


class TestFullPipeline:
    def test_end_to_end_analysis(self):
        """Test complete pipeline: advisor → LLM → parser → formatter."""
        advisor = MetacouplingAssistant(MockFullClient())
        result = advisor.analyze(
            "My research examines lithium mining in South America and its "
            "connection to electric vehicle production in Europe."
        )

        # Result type
        assert isinstance(result, AnalysisResult)
        assert result.turn_number == 1

        # Parsed data present
        parsed = result.parsed
        assert parsed.is_parsed
        assert "telecoupling" in parsed.coupling_classification.lower()
        assert len(parsed.systems) > 0
        assert len(parsed.flows) >= 3
        assert len(parsed.agents) >= 4
        assert len(parsed.causes) > 0
        assert len(parsed.effects) > 0
        assert len(parsed.suggestions) >= 3

        # Formatted output readable
        formatted = result.formatted
        assert "METACOUPLING FRAMEWORK ANALYSIS" in formatted
        assert "COUPLING CLASSIFICATION" in formatted
        assert "SYSTEMS IDENTIFICATION" in formatted
        assert "FLOWS ANALYSIS" in formatted

        # Raw text preserved
        assert result.raw == FULL_MOCK_RESPONSE

    def test_multi_turn_conversation(self):
        """Test multi-turn refinement maintains conversation context."""
        client = MockFullClient()
        advisor = MetacouplingAssistant(client)

        result1 = advisor.analyze("Lithium mining telecoupling study")
        assert result1.turn_number == 1
        assert len(advisor.history) == 3  # system + user + assistant

        result2 = advisor.refine(
            "Focus more on the water depletion effects in sending systems",
            focus_component="effects",
        )
        assert result2.turn_number == 2
        assert len(advisor.history) == 5  # +user + assistant


class TestPublicAPIImports:
    """Verify all public API exports are importable and functional."""

    def test_advisor_importable(self):
        assert MetacouplingAssistant is not None

    def test_result_importable(self):
        assert AnalysisResult is not None

    def test_protocol_importable(self):
        assert LLMClient is not None
        assert Message is not None
        assert LLMResponse is not None

    def test_adapters_importable(self):
        assert OpenAIAdapter is not None
        assert AnthropicAdapter is not None

    def test_enums_importable(self):
        assert CouplingType.TELECOUPLING.value == "telecoupling"
        assert SystemRole.SENDING.value == "sending"
        assert FlowCategory.MATTER.value == "matter"
        assert AgentLevel.INDIVIDUAL.value == "individuals / households"
        assert CauseCategory.SOCIOECONOMIC.value == "economic"
        assert FrameworkComponent.SYSTEMS.value == "systems"

    def test_version_accessible(self):
        import metacouplingllm
        assert hasattr(metacoupling, "__version__")
        assert metacouplingllm.__version__ == "0.1.0"

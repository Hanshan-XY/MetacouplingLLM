# Metacoupling

`metacoupling` helps researchers apply the telecoupling and metacoupling
frameworks to new research topics with LLMs. It can structure analyses from
plain-language study descriptions, recommend relevant literature, validate
pericoupling relationships, ground outputs with web results, and generate
country- or ADM1-level maps.

## Installation

Choose the smallest install that matches your workflow:

```bash
pip install metacoupling
pip install "metacoupling[openai]"
pip install "metacoupling[anthropic]"
pip install "metacoupling[search]"
pip install "metacoupling[viz]"
pip install "metacoupling[all]"
```

## Quick Start

If you want to use the built-in OpenAI adapter, install the matching extra:

```bash
pip install "metacoupling[openai]"
```

```python
from openai import OpenAI
from metacouplingllm import (
    JOURNAL_ARTICLES_2025,
    MetacouplingAssistant,
    OpenAIAdapter,
)

client = OpenAI(api_key="your-key")
advisor = MetacouplingAssistant(
    OpenAIAdapter(client, model="gpt-5.2"),
    web_search=True,
    web_search_max_results=5,
    web_structured_extraction=True,  # Recommended with web_search + auto_map
    auto_map=True,
    rag_corpus=JOURNAL_ARTICLES_2025,
    rag_top_k=10,
    rag_min_score=0.15,
)

result = advisor.analyze("""
    My research examines what the impact of Brazil's soybeans exports is.
""")
print(result.formatted)

if result.map:
    result.map.savefig("map.png", dpi=150, bbox_inches="tight")
```

Recommended default: use `web_structured_extraction=True` whenever you enable
both `web_search=True` and `auto_map=True`. It runs a second, conservative LLM
pass over web snippets to extract validated receiving countries, spillover
countries, and map-ready flows. The validated output is also available on
`result.web_map_signals`.

For country-level maps, metacoupling now:

- uses a local World Bank `Admin 0_all_layers.gpkg` when available
- otherwise auto-downloads the hosted World Bank `Admin 0_all_layers` mirror
- and falls back to the official World Bank `wb_countries_admin0_10m.zip`
- and uses World Bank-only disputed / indeterminate overlays

ADM1 maps can also auto-download the hosted World Bank `Admin 1.gpkg`
mirror when no local `adm1_shapefile` is provided, plus the hosted World
Bank `NDLSA.gpkg` overlay for disputed areas.

You can still force a specific country basemap with:

```python
advisor = MetacouplingAssistant(
    OpenAIAdapter(client, model="gpt-4o"),
    auto_map=True,
    adm0_shapefile=r"C:\path\to\World Bank Official Boundaries - Admin 0_all_layers.gpkg",
)
```

## Core Capabilities

- Structured metacoupling analyses from free-text research descriptions
- Multi-turn refinement across systems, flows, agents, causes, and effects
- Literature recommendations from a curated telecoupling/metacoupling database
- Optional web-search grounding with inline web citations
- Country-level and ADM1 pericoupling validation
- Optional map generation for country and subnational analyses

## Documentation

- [Introduction](INTRODUCTION.md): package overview, architecture, and examples
- [Manual](MANUAL.md): detailed usage guidance

## License

MIT. See [LICENSE](LICENSE).

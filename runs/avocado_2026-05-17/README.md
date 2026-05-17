# Pipeline trace — Mexican avocado sustainability

**Query:** Impact of avocado production and trade in Mexico on sustainability

**Model:** `gpt-5.5`

**Wall-clock:** 323.0s | **LLM calls:** 3 | **Total tokens:** 27377

## Artifacts (in pipeline order)

| # | File | Description |
|---|---|---|
| 00 | [`00_run_config.md`](./00_run_config.md) | Config, env, git SHA, all assistant params |
| 01 | [`01_web_results_raw.md`](./01_web_results_raw.md) | Raw web search results (pre-extraction) |
| 02 | [`02_llm_call_1_web_extraction.md`](./02_llm_call_1_web_extraction.md) | LLM call #1: web structured extraction (request + response) |
| 03 | [`03_web_structured_signals.md`](./03_web_structured_signals.md) | Parsed output of call #1 (structured map signals) |
| 04 | [`04_rag_chunks.md`](./04_rag_chunks.md) | 8 retrieved RAG chunks with full text + scores |
| 05 | [`05_llm_call_2_main_analysis.md`](./05_llm_call_2_main_analysis.md) | LLM call #2: main framework analysis (system + user + response) |
| 06 | [`06_parsed_analysis.md`](./06_parsed_analysis.md) | Parsed `ParsedAnalysis` fields from call #2's response |
| 07 | [`07_llm_call_3_map_extraction.md`](./07_llm_call_3_map_extraction.md) | LLM call #3: structured map extraction (request + response) |
| 08 | [`08_map_data.md`](./08_map_data.md) | Parsed `map_data` dict (structured input to renderer) |
| 09 | [`09_formatted_output.md`](./09_formatted_output.md) | Final human-readable formatted text |
| 10 | [`10_pipeline_metadata.md`](./10_pipeline_metadata.md) | Map type, notices, warnings, token usage table, wall-clock breakdown |
| — | [`map.png`](./map.png) | Rendered matplotlib figure (if `auto_map` produced one) |

# Pipeline trace

**Query:** Impact of avocado production and trade in Mexico on sustainability

**Model:** `gpt-5.5`

**Wall-clock:** 5263.9s | **LLM calls:** 3 | **Total tokens:** 74537

## Reading this trace

- Files are numbered by **pipeline stage** (the order steps run), not by model-call order. For example, the structured web-extraction call appears early, at `02`, because it runs during the web-search stage, before the main analysis. The one exception is the abstract call (`11`): it runs right after the formatted output (`09`) but is numbered after the metadata (`10`).
- The **LLM calls** count above, and the token table in `10_pipeline_metadata.md`, include only calls captured through the assistant's `chat()` proxy. Two web-stage model calls bypass that proxy: the provider's native web search (when used) and the structured web extraction (`02`), which is sent to the unwrapped adapter so that it can use the provider's schema-constrained output mode. Only their results are kept: the web results in `01` and the structured signals in `03`.
- Not every file appears in every run: the web, RAG, map, and supplementary stages are written only when the corresponding feature is enabled.

## Artifacts

| File | Description |
|---|---|
| [`00_run_config.md`](./00_run_config.md) | Query, model, parameters, git SHA, and environment. |
| [`01_web_results_raw.md`](./01_web_results_raw.md) | Raw results returned by the web search. |
| [`02_llm_call_web_extraction.md`](./02_llm_call_web_extraction.md) | Structured web-extraction model call (not captured; see above). |
| [`03_web_structured_signals.md`](./03_web_structured_signals.md) | Structured signals and evidence cards from the web results. |
| [`04_rag_chunks.md`](./04_rag_chunks.md) | Literature passages retrieved from the RAG corpus. |
| [`05_llm_call_main_analysis.md`](./05_llm_call_main_analysis.md) | Main framework-analysis model call, with the full six-layer system prompt. |
| [`06_parsed_analysis.md`](./06_parsed_analysis.md) | Parsed ParsedAnalysis structure (parser output of the main response). |
| [`07_llm_call_map_extraction.md`](./07_llm_call_map_extraction.md) | Map-signal extraction model call. |
| [`08_map_data.md`](./08_map_data.md) | Structured map data used to render the figure. |
| [`09_formatted_output.md`](./09_formatted_output.md) | Final formatted report. |
| [`10_pipeline_metadata.md`](./10_pipeline_metadata.md) | Per-call token usage, wall-clock breakdown, and map metadata. |
| [`11_llm_call_abstract.md`](./11_llm_call_abstract.md) | Abstract-generation model call (prompt rebuilt and response regenerated on 2026-10-07; see the file). |
| [`map.png`](./map.png) | Rendered metacoupling map. |

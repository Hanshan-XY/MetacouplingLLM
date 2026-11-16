# 02 — LLM call: structured web extraction

_Not captured by the in-process trace: the web-extraction call is sent to the unwrapped adapter, not through the assistant's `chat()` proxy, so that it can use the provider's schema-constrained output mode. Its input, the web results, is in file 01 and its output, the structured signals, in file 03._

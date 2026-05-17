# 03 — Web structured signals

Parsed output of LLM call #1 (file 02). This is what gets appended to the main analysis prompt as additional structured context AND feeds the map renderer.

```json
{
  "focal_country": "MEX",
  "receiving_systems": [
    {
      "country": "USA",
      "kind": "direct",
      "confidence": 0.95,
      "evidence": [
        "W2",
        "W4"
      ],
      "reason": "Snippets explicitly describe the U.S.-Mexico avocado trade and avocado supply chain."
    }
  ],
  "spillover_systems": [],
  "flows": [
    {
      "category": "matter",
      "source_country": "MEX",
      "target_country": "USA",
      "direction": "Mexico → United States",
      "description": "Mexican avocados move through the U.S.-Mexico avocado trade/supply chain to the United States.",
      "kind": "direct",
      "confidence": 0.95,
      "evidence": [
        "W2",
        "W4"
      ]
    },
    {
      "category": "capital",
      "source_country": "USA",
      "target_country": "MEX",
      "direction": "United States → Mexico",
      "description": "Payment/capital associated with U.S.-Mexico avocado trade flows from the United States to Mexico.",
      "kind": "direct",
      "confidence": 0.75,
      "evidence": [
        "W2",
        "W4"
      ]
    }
  ]
}
```

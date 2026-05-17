# 08 — Map data (structured output of call #3)

`parsed.map_data` — the structured map signals the renderer consumes. Includes `focal_country`, `adm1_region`, `receiving_countries`, `spillover_countries`, and a `flows` list with canonical ISO-3 codes and Liu 2017 flow categories.

```json
{
  "focal_country": "MEX",
  "adm1_region": "MEX016",
  "mentioned_adm1_regions": [
    "MEX014"
  ],
  "receiving_countries": [
    "USA"
  ],
  "spillover_countries": [
    "CHL",
    "PER"
  ],
  "flows": [
    {
      "category": "matter",
      "source": "MEX",
      "target": "USA",
      "direction": "Mexico → United States",
      "bidirectional": false
    },
    {
      "category": "capital",
      "source": "USA",
      "target": "MEX",
      "direction": "United States → Mexico",
      "bidirectional": false
    }
  ]
}
```

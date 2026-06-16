# 08 — Map data

`parsed.map_data` — structured input to the renderer.

```json
{
  "focal_country": "MEX",
  "adm1_region": null,
  "mentioned_adm1_regions": [
    "MEX016",
    "MEX014"
  ],
  "receiving_countries": [
    "USA",
    "CAN",
    "JPN",
    "SLV",
    "HND"
  ],
  "spillover_countries": [
    "PER",
    "CHL"
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
      "category": "matter",
      "source": "MEX",
      "target": "CAN",
      "direction": "Mexico → Canada",
      "bidirectional": false
    },
    {
      "category": "matter",
      "source": "MEX",
      "target": "JPN",
      "direction": "Mexico → Japan",
      "bidirectional": false
    },
    {
      "category": "matter",
      "source": "MEX",
      "target": "SLV",
      "direction": "Mexico → El Salvador",
      "bidirectional": false
    },
    {
      "category": "matter",
      "source": "MEX",
      "target": "HND",
      "direction": "Mexico → Honduras",
      "bidirectional": false
    },
    {
      "category": "capital",
      "source": "USA",
      "target": "MEX",
      "direction": "United States → Mexico",
      "bidirectional": false
    },
    {
      "category": "capital",
      "source": "CAN",
      "target": "MEX",
      "direction": "Canada → Mexico",
      "bidirectional": false
    },
    {
      "category": "capital",
      "source": "JPN",
      "target": "MEX",
      "direction": "Japan → Mexico",
      "bidirectional": false
    },
    {
      "category": "capital",
      "source": "SLV",
      "target": "MEX",
      "direction": "El Salvador → Mexico",
      "bidirectional": false
    },
    {
      "category": "capital",
      "source": "HND",
      "target": "MEX",
      "direction": "Honduras → Mexico",
      "bidirectional": false
    },
    {
      "category": "information",
      "source": "MEX",
      "target": "USA",
      "direction": "Bidirectional (Mexico ↔ United States)",
      "bidirectional": true
    }
  ]
}
```

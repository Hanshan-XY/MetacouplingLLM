# 08 — Map data

`parsed.map_data` — structured input to the renderer.

```json
{
  "focal_country": "IDN",
  "adm1_region": null,
  "mentioned_adm1_regions": [
    "IDN020"
  ],
  "receiving_countries": [
    "IND",
    "CHN",
    "USA",
    "PAK",
    "CHE"
  ],
  "spillover_countries": [
    "MYS"
  ],
  "flows": [
    {
      "category": "matter",
      "source": "IDN",
      "target": "IND",
      "direction": "Indonesia → India",
      "bidirectional": false
    },
    {
      "category": "matter",
      "source": "IDN",
      "target": "CHN",
      "direction": "Indonesia → China",
      "bidirectional": false
    },
    {
      "category": "matter",
      "source": "IDN",
      "target": null,
      "direction": "Indonesia → European Union",
      "bidirectional": false,
      "target_supranational": "European Union",
      "target_supranational_members": [
        "AUT",
        "BEL",
        "BGR",
        "HRV",
        "CYP",
        "CZE",
        "DNK",
        "EST",
        "FIN",
        "FRA",
        "DEU",
        "GRC",
        "HUN",
        "IRL",
        "ITA",
        "LVA",
        "LTU",
        "LUX",
        "MLT",
        "NLD",
        "POL",
        "PRT",
        "ROU",
        "SVK",
        "SVN",
        "ESP",
        "SWE"
      ]
    },
    {
      "category": "matter",
      "source": "IDN",
      "target": "USA",
      "direction": "Indonesia → United States",
      "bidirectional": false
    },
    {
      "category": "matter",
      "source": "IDN",
      "target": "PAK",
      "direction": "Indonesia → Pakistan",
      "bidirectional": false
    },
    {
      "category": "matter",
      "source": "IDN",
      "target": "CHE",
      "direction": "Indonesia → Switzerland",
      "bidirectional": false
    },
    {
      "category": "capital",
      "source": "IND",
      "target": "IDN",
      "direction": "India → Indonesia",
      "bidirectional": false
    },
    {
      "category": "capital",
      "source": "CHN",
      "target": "IDN",
      "direction": "China → Indonesia",
      "bidirectional": false
    },
    {
      "category": "capital",
      "source": null,
      "target": "IDN",
      "direction": "European Union → Indonesia",
      "bidirectional": false,
      "source_supranational": "European Union",
      "source_supranational_members": [
        "AUT",
        "BEL",
        "BGR",
        "HRV",
        "CYP",
        "CZE",
        "DNK",
        "EST",
        "FIN",
        "FRA",
        "DEU",
        "GRC",
        "HUN",
        "IRL",
        "ITA",
        "LVA",
        "LTU",
        "LUX",
        "MLT",
        "NLD",
        "POL",
        "PRT",
        "ROU",
        "SVK",
        "SVN",
        "ESP",
        "SWE"
      ]
    },
    {
      "category": "capital",
      "source": "USA",
      "target": "IDN",
      "direction": "United States → Indonesia",
      "bidirectional": false
    },
    {
      "category": "capital",
      "source": "PAK",
      "target": "IDN",
      "direction": "Pakistan → Indonesia",
      "bidirectional": false
    },
    {
      "category": "capital",
      "source": "CHE",
      "target": "IDN",
      "direction": "Switzerland → Indonesia",
      "bidirectional": false
    },
    {
      "category": "information",
      "source": "IDN",
      "target": "IND",
      "direction": "Bidirectional (Indonesia ↔ India)",
      "bidirectional": true
    },
    {
      "category": "information",
      "source": "IDN",
      "target": "CHN",
      "direction": "Bidirectional (Indonesia ↔ China)",
      "bidirectional": true
    },
    {
      "category": "information",
      "source": "IDN",
      "target": null,
      "direction": "Bidirectional (Indonesia ↔ European Union)",
      "bidirectional": true,
      "target_supranational": "European Union",
      "target_supranational_members": [
        "AUT",
        "BEL",
        "BGR",
        "HRV",
        "CYP",
        "CZE",
        "DNK",
        "EST",
        "FIN",
        "FRA",
        "DEU",
        "GRC",
        "HUN",
        "IRL",
        "ITA",
        "LVA",
        "LTU",
        "LUX",
        "MLT",
        "NLD",
        "POL",
        "PRT",
        "ROU",
        "SVK",
        "SVN",
        "ESP",
        "SWE"
      ]
    },
    {
      "category": "information",
      "source": "IDN",
      "target": "USA",
      "direction": "Bidirectional (Indonesia ↔ United States)",
      "bidirectional": true
    },
    {
      "category": "information",
      "source": "IDN",
      "target": "PAK",
      "direction": "Indonesia → Pakistan",
      "bidirectional": false
    },
    {
      "category": "information",
      "source": "IDN",
      "target": "CHE",
      "direction": "Bidirectional (Indonesia ↔ Switzerland)",
      "bidirectional": true
    }
  ]
}
```

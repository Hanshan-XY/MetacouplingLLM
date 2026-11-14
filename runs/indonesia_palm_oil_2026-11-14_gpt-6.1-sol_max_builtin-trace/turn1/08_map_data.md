# 08 — Map data

`parsed.map_data` — structured input to the renderer.

```json
{
  "focal_country": "IDN",
  "adm1_region": null,
  "mentioned_adm1_regions": [
    "IDN035"
  ],
  "receiving_countries": [
    "IND",
    "MYS",
    "CHN",
    "PAK",
    "BGD",
    "USA"
  ],
  "spillover_countries": [],
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
      "target": "MYS",
      "direction": "Indonesia → Malaysia",
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
      "target": "PAK",
      "direction": "Indonesia → Pakistan",
      "bidirectional": false
    },
    {
      "category": "matter",
      "source": "IDN",
      "target": "BGD",
      "direction": "Indonesia → Bangladesh",
      "bidirectional": false
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
      "category": "capital",
      "source": "IND",
      "target": "IDN",
      "direction": "India → Indonesia",
      "bidirectional": false
    },
    {
      "category": "capital",
      "source": "MYS",
      "target": "IDN",
      "direction": "Malaysia → Indonesia",
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
      "source": "PAK",
      "target": "IDN",
      "direction": "Pakistan → Indonesia",
      "bidirectional": false
    },
    {
      "category": "capital",
      "source": "BGD",
      "target": "IDN",
      "direction": "Bangladesh → Indonesia",
      "bidirectional": false
    },
    {
      "category": "capital",
      "source": "USA",
      "target": "IDN",
      "direction": "United States → Indonesia",
      "bidirectional": false
    }
  ]
}
```

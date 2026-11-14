# 03 — Web structured signals

Parsed map signals derived from the web results.

```json
{
  "focal_country": "IDN",
  "receiving_systems": [
    {
      "country": "IND",
      "kind": "direct",
      "confidence": 0.99,
      "evidence": [
        "W1",
        "W3"
      ],
      "reason": "Indonesia's largest palm oil export destination in the Trase analysis; also identified as a 2025 export market."
    },
    {
      "country": "PAK",
      "kind": "direct",
      "confidence": 0.99,
      "evidence": [
        "W1",
        "W3"
      ],
      "reason": "A leading Indonesian palm oil destination; exports to Pakistan increased in 2025."
    },
    {
      "country": "CHN",
      "kind": "direct",
      "confidence": 0.99,
      "evidence": [
        "W1",
        "W3"
      ],
      "reason": "A leading Indonesian palm oil destination; exports to China increased in 2025."
    },
    {
      "country": "BGD",
      "kind": "direct",
      "confidence": 0.98,
      "evidence": [
        "W3"
      ],
      "reason": "An explicitly named destination for Indonesian palm-sector exports, with increased shipments in 2025."
    },
    {
      "country": "European Union",
      "kind": "direct",
      "confidence": 0.98,
      "evidence": [
        "W3"
      ],
      "reason": "An explicitly named destination block for Indonesian palm-sector exports, with decreased shipments in 2025.",
      "supranational_members": [
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
      "country": "USA",
      "kind": "direct",
      "confidence": 0.98,
      "evidence": [
        "W3"
      ],
      "reason": "An explicitly named destination for Indonesian palm-sector exports, with decreased shipments in 2025."
    }
  ],
  "spillover_systems": [
    {
      "country": "MYS",
      "kind": "proxy",
      "confidence": 0.75,
      "evidence": [
        "W24"
      ],
      "reason": "An alternative palm oil sourcing origin alongside Indonesia in the Spain-linked EUDR exercise; included as a competing-supplier proxy, not a demonstrated spillover impact."
    }
  ],
  "flows": [
    {
      "category": "matter",
      "source_country": "IDN",
      "target_country": "IND",
      "direction": "Indonesia → India",
      "description": "Palm oil exports from Indonesia to India; palm-sector shipments declined in 2025.",
      "kind": "direct",
      "confidence": 0.99,
      "evidence": [
        "W1",
        "W3"
      ]
    },
    {
      "category": "matter",
      "source_country": "IDN",
      "target_country": "PAK",
      "direction": "Indonesia → Pakistan",
      "description": "Palm oil exports from Indonesia to Pakistan; palm-sector shipments increased in 2025.",
      "kind": "direct",
      "confidence": 0.99,
      "evidence": [
        "W1",
        "W3"
      ]
    },
    {
      "category": "matter",
      "source_country": "IDN",
      "target_country": "CHN",
      "direction": "Indonesia → China",
      "description": "Palm oil exports from Indonesia to China; palm-sector shipments increased in 2025.",
      "kind": "direct",
      "confidence": 0.99,
      "evidence": [
        "W1",
        "W3"
      ]
    },
    {
      "category": "matter",
      "source_country": "IDN",
      "target_country": "BGD",
      "direction": "Indonesia → Bangladesh",
      "description": "Indonesian palm oil and related palm-product exports to Bangladesh increased in 2025.",
      "kind": "direct",
      "confidence": 0.98,
      "evidence": [
        "W3"
      ]
    },
    {
      "category": "matter",
      "source_country": "IDN",
      "target_country": "European Union",
      "direction": "Indonesia → European Union",
      "description": "Indonesian palm oil and related palm-product exports to the European Union decreased in 2025.",
      "kind": "direct",
      "confidence": 0.98,
      "evidence": [
        "W3"
      ],
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
      "source_country": "IDN",
      "target_country": "USA",
      "direction": "Indonesia → United States",
      "description": "Indonesian palm oil and related palm-product exports to the United States decreased in 2025.",
      "kind": "direct",
      "confidence": 0.98,
      "evidence": [
        "W3"
      ]
    }
  ],
  "evidence_cards": [
    {
      "source_id": "W1",
      "claims_supported": [
        "India, Pakistan and China were Indonesia's leading palm oil export destinations.",
        "Domestic use absorbed 53% of 2024 production.",
        "Domestic buyers accounted for 67% of attributed supply-chain deforestation exposure.",
        "Estimated 2024 peat-subsidence emissions approached 148 million tonnes CO2-equivalent."
      ],
      "relevance_score": 1.0,
      "source_type": "other"
    },
    {
      "source_id": "W2",
      "claims_supported": [
        "Official 2025 oil palm area and production statistics are reported by province.",
        "Trade tables classify exports by destination and imports by origin.",
        "Production totals combine establishment surveys with smallholder statistics."
      ],
      "relevance_score": 0.82,
      "source_type": "government"
    },
    {
      "source_id": "W3",
      "claims_supported": [
        "Indonesia's 2025 palm-sector exports reached 32.343 million tonnes and US$35.87 billion.",
        "Exports increased to China, Bangladesh and Pakistan.",
        "Exports decreased to India, the European Union and the United States.",
        "Domestic biodiesel use consumed 12.704 million tonnes in 2025."
      ],
      "relevance_score": 0.93,
      "source_type": "industry"
    },
    {
      "source_id": "W4",
      "claims_supported": [
        "Mapped Indonesian oil palm area reached 16.24 million hectares in 2019.",
        "Same-year forest clearing and oil palm conversion represented 29% of 2001–2019 old-growth forest loss.",
        "Lower palm oil prices correlated with less plantation expansion and forest loss."
      ],
      "relevance_score": 0.97,
      "source_type": "academic"
    },
    {
      "source_id": "W5",
      "claims_supported": [
        "By 2020, 7.8 million hectares of cleared Indonesian primary forest had become oil palm.",
        "Another 8.8 million hectares of cleared forest remained unused in 2020.",
        "Oil palm was the most common eventual productive use of initially idle cleared land."
      ],
      "relevance_score": 0.92,
      "source_type": "academic"
    },
    {
      "source_id": "W6",
      "claims_supported": [
        "Globally, palm oil supplied about 40% of vegetable oil demand using roughly 5–5.5% of oil-crop land.",
        "Forest conversion, peat drainage and fire-related pollution are major environmental harms.",
        "Replacing palm oil with lower-yielding oils could transfer land-use pressures elsewhere."
      ],
      "relevance_score": 0.9,
      "source_type": "academic"
    },
    {
      "source_id": "W7",
      "claims_supported": [
        "Jambi-based life-cycle analysis found no greenhouse gas savings for first rotations established after forest clearance.",
        "Later rotations and cultivation on nonforested land offered greater potential emissions savings.",
        "Open effluent ponds increased methane emissions relative to gas-capture systems."
      ],
      "relevance_score": 0.95,
      "source_type": "academic"
    },
    {
      "source_id": "W8",
      "claims_supported": [
        "Satellite observations identified three methane plumes at Indonesian palm oil mills.",
        "Fewer than 10% of Indonesian mills had biogas systems in the evidence cited.",
        "Some additional satellite methane enhancements could reflect retrieval artifacts."
      ],
      "relevance_score": 0.91,
      "source_type": "academic"
    },
    {
      "source_id": "W9",
      "claims_supported": [
        "Kais watershed modeling estimated 21% higher runoff after oil palm conversion.",
        "Modeled sediment yields increased by 16.9%.",
        "Modeled total nitrogen and phosphorus increased by 78% and 144%, respectively."
      ],
      "relevance_score": 0.95,
      "source_type": "academic"
    },
    {
      "source_id": "W10",
      "claims_supported": [
        "Indonesia's 2015 fires caused estimated economic damages and losses of US$16.1 billion.",
        "Drained peatlands intensified persistent haze during the fire crisis.",
        "The assessment explicitly states that not all fires were started for oil palm."
      ],
      "relevance_score": 0.85,
      "source_type": "international_organization"
    },
    {
      "source_id": "W11",
      "claims_supported": [
        "Across 42 Sumatra plots, industrial estate yields averaged roughly twice smallholder yields.",
        "Reduced management intensity did not necessarily reduce yields and improved ecological outcomes.",
        "Nearby forest generally enhanced plantation biodiversity and ecosystem functioning."
      ],
      "relevance_score": 0.96,
      "source_type": "academic"
    },
    {
      "source_id": "W12",
      "claims_supported": [
        "Historical Indonesian oil palm expansion was associated with faster district poverty reduction and expenditure growth.",
        "Higher returns to agricultural labour were the principal economic mechanism.",
        "Expansion also increased deforestation, fires, certain health problems and conflict."
      ],
      "relevance_score": 0.99,
      "source_type": "academic"
    },
    {
      "source_id": "W13",
      "claims_supported": [
        "Kalimantan oil palm villages with established market livelihoods showed socioeconomic gains relative to matched comparisons.",
        "Remote, heavily forested subsistence villages experienced the opposite socioeconomic pattern.",
        "Social and environmental improvements generally lagged those in comparison villages."
      ],
      "relevance_score": 0.96,
      "source_type": "academic"
    },
    {
      "source_id": "W14",
      "claims_supported": [
        "West Kalimantan investigations documented allegations of inadequate land consultation and compensation.",
        "Peat drainage was linked to subsidence, flooding, fire vulnerability and livelihood disruption.",
        "Two plantation concessions occupied approximately 76% of Olak Olak village land."
      ],
      "relevance_score": 0.97,
      "source_type": "ngo"
    },
    {
      "source_id": "W15",
      "claims_supported": [
        "ILO identifies unsafe work, low wages, excessive hours and discrimination in Indonesia's palm oil sector.",
        "Informality, temporary contracts and remoteness heighten exploitation risks.",
        "The project proposes worker-driven monitoring, remediation and independent verification."
      ],
      "relevance_score": 0.95,
      "source_type": "international_organization"
    },
    {
      "source_id": "W16",
      "claims_supported": [
        "Certification was estimated to reduce deforestation by 33% relative to the uncertified counterfactual.",
        "No detectable certification effect was found on peatland forest loss or active fire detections.",
        "Certified estates contained under 1% of forests remaining within Indonesian oil palm plantations by 2015."
      ],
      "relevance_score": 0.99,
      "source_type": "academic"
    },
    {
      "source_id": "W17",
      "claims_supported": [
        "The analysis covered 36,311 Indonesian villages during 2000–2018.",
        "Certification was associated with poverty reductions in market-based, but not subsistence-based, communities.",
        "Outcomes varied with baseline livelihoods and regional production histories."
      ],
      "relevance_score": 0.97,
      "source_type": "academic"
    },
    {
      "source_id": "W18",
      "claims_supported": [
        "Surveys covered 1,541 Indonesian oil palm farming households.",
        "Independent smallholders were poorly reached by zero-deforestation commitments.",
        "Commitment-focused training and price differentiation scarcely reached producers.",
        "Average village deforestation patterns were similar inside commitment sourcing areas and elsewhere."
      ],
      "relevance_score": 0.99,
      "source_type": "academic"
    },
    {
      "source_id": "W19",
      "claims_supported": [
        "RSPO reported 16.2 million tonnes of certified palm oil sales in 2025, about 20% of global supply.",
        "Members reported protecting 551,637 hectares globally.",
        "An Indonesian example involved legal-protection efforts for more than 8,300 hectares in West Kalimantan.",
        "Independent smallholder credit sales generated US$7.9 million globally in 2025."
      ],
      "relevance_score": 0.85,
      "source_type": "industry"
    },
    {
      "source_id": "W20",
      "claims_supported": [
        "Indonesia raised its biodiesel mandate from B40 to B50 in July 2026.",
        "Palm export levies finance domestic biodiesel subsidies.",
        "Subsidy payments exceeded levy revenue, approximately halving the fund balance during 2025.",
        "USDA forecast annual biodiesel consumption of 17.5 billion litres."
      ],
      "relevance_score": 0.95,
      "source_type": "government"
    },
    {
      "source_id": "W21",
      "claims_supported": [
        "Modeled B50 scenarios required 4.85–8.55 million hectares of land conversion by 2030.",
        "Oil palm scenarios implied 360–3,753 million tonnes CO2-equivalent emissions.",
        "Food and bioenergy production scenarios competed for overlapping suitable land."
      ],
      "relevance_score": 0.95,
      "source_type": "academic"
    },
    {
      "source_id": "W22",
      "claims_supported": [
        "The model examines palm oil climate and trade policy involving Indonesia and Malaysia over 1988–2016.",
        "A modeled 50% production tax reduced emissions by 7.4 gigatonnes relative to business as usual.",
        "Coordinated, credible import tariffs produced a modeled 5.4-gigatonne emissions reduction."
      ],
      "relevance_score": 0.96,
      "source_type": "academic"
    },
    {
      "source_id": "W23",
      "claims_supported": [
        "EUDR covers oil palm and specified derived products on the European Union market.",
        "Covered production must avoid deforestation after December 31, 2020.",
        "Required information includes harvest-plot geolocation.",
        "Application begins December 30, 2026 or June 30, 2027, depending on enterprise size."
      ],
      "relevance_score": 0.86,
      "source_type": "government"
    },
    {
      "source_id": "W24",
      "claims_supported": [
        "A Spain-linked EUDR exercise used palm oil shipment information from Indonesia and Malaysia.",
        "Indonesian restrictions on sharing larger-plot polygons hindered geolocation assessment.",
        "Certification could support, but not replace, operators' due-diligence responsibility.",
        "Exercise findings were learning outputs, not compliance determinations."
      ],
      "relevance_score": 0.98,
      "source_type": "international_organization"
    },
    {
      "source_id": "W25",
      "claims_supported": [
        "Indonesia challenged EU measures affecting palm oil and palm-based biofuels.",
        "The WTO panel found discriminatory administration and shortcomings in low-indirect-land-use-change certification procedures.",
        "Indonesia did not establish that relevant EU limits were more trade-restrictive than necessary.",
        "The dispute does not adjudicate the separate EUDR."
      ],
      "relevance_score": 0.98,
      "source_type": "international_organization"
    }
  ],
  "suggested_followup_queries": [
    "Indonesia palm oil destination specific deforestation footprints 2025",
    "Indonesian palm oil fires transboundary haze impact attribution",
    "Indonesia Malaysia palm oil competition deforestation leakage evidence",
    "Indonesia palm oil smallholder EUDR compliance costs 2027",
    "Indonesian palm oil export earnings distribution workers smallholders"
  ]
}
```

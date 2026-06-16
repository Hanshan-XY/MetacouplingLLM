# 03 — Web structured signals

Parsed map signals derived from the web results.

```json
{
  "focal_country": "MEX",
  "receiving_systems": [
    {
      "country": "USA",
      "kind": "direct",
      "confidence": 0.96,
      "evidence": [
        "W1",
        "W2",
        "W6",
        "W20"
      ],
      "reason": "Largest named destination for Mexican avocado exports; sources report about 80% by volume, US$3.444 billion in 2024, and major U.S. supply-chain dependence."
    },
    {
      "country": "CAN",
      "kind": "direct",
      "confidence": 0.9,
      "evidence": [
        "W1",
        "W2"
      ],
      "reason": "Named as a major destination for Mexican avocado exports and reported at US$257 million in 2024 export value."
    },
    {
      "country": "JPN",
      "kind": "direct",
      "confidence": 0.9,
      "evidence": [
        "W1",
        "W2"
      ],
      "reason": "Named as a major destination for Mexican avocado exports and reported at US$108 million in 2024 export value."
    },
    {
      "country": "SLV",
      "kind": "direct",
      "confidence": 0.82,
      "evidence": [
        "W2"
      ],
      "reason": "Data México lists El Salvador as a principal 2024 destination for Mexican fresh or dried avocado exports."
    },
    {
      "country": "HND",
      "kind": "direct",
      "confidence": 0.82,
      "evidence": [
        "W2"
      ],
      "reason": "Data México lists Honduras as a principal 2024 destination for Mexican fresh or dried avocado exports."
    }
  ],
  "spillover_systems": [],
  "flows": [
    {
      "category": "matter",
      "source_country": "MEX",
      "target_country": "USA",
      "direction": "Mexico → United States",
      "description": "Mexican avocado exports to the United States, the destination for about 80% of export volume and US$3.444 billion in 2024 export value.",
      "kind": "direct",
      "confidence": 0.95,
      "evidence": [
        "W1",
        "W2"
      ]
    },
    {
      "category": "matter",
      "source_country": "MEX",
      "target_country": "CAN",
      "direction": "Mexico → Canada",
      "description": "Mexican avocado exports to Canada, identified as a major destination and valued at US$257 million in 2024.",
      "kind": "direct",
      "confidence": 0.88,
      "evidence": [
        "W1",
        "W2"
      ]
    },
    {
      "category": "matter",
      "source_country": "MEX",
      "target_country": "JPN",
      "direction": "Mexico → Japan",
      "description": "Mexican avocado exports to Japan, identified as a major destination and valued at US$108 million in 2024.",
      "kind": "direct",
      "confidence": 0.88,
      "evidence": [
        "W1",
        "W2"
      ]
    },
    {
      "category": "matter",
      "source_country": "MEX",
      "target_country": "SLV",
      "direction": "Mexico → El Salvador",
      "description": "Mexican fresh or dried avocado exports to El Salvador, valued at US$38 million in 2024.",
      "kind": "direct",
      "confidence": 0.82,
      "evidence": [
        "W2"
      ]
    },
    {
      "category": "matter",
      "source_country": "MEX",
      "target_country": "HND",
      "direction": "Mexico → Honduras",
      "description": "Mexican fresh or dried avocado exports to Honduras, valued at US$31.6 million in 2024.",
      "kind": "direct",
      "confidence": 0.82,
      "evidence": [
        "W2"
      ]
    },
    {
      "category": "capital",
      "source_country": "USA",
      "target_country": "MEX",
      "direction": "United States → Mexico",
      "description": "Payment/value flow associated with United States purchases of Mexican avocados; 2024 Mexican avocado exports to the United States were US$3.444 billion.",
      "kind": "direct",
      "confidence": 0.86,
      "evidence": [
        "W2"
      ]
    },
    {
      "category": "capital",
      "source_country": "CAN",
      "target_country": "MEX",
      "direction": "Canada → Mexico",
      "description": "Payment/value flow associated with Canadian purchases of Mexican avocados; 2024 Mexican avocado exports to Canada were US$257 million.",
      "kind": "direct",
      "confidence": 0.78,
      "evidence": [
        "W2"
      ]
    }
  ],
  "evidence_cards": [
    {
      "source_id": "W1",
      "claims_supported": [
        "Mexico 2025 avocado production forecast at 2.75 million metric tons",
        "Mexico 2025 avocado exports forecast at 1.34 million metric tons",
        "United States receives about 80% of Mexican avocado exports by volume",
        "Canada and Japan follow the United States as Mexican avocado export destinations"
      ],
      "relevance_score": 0.95,
      "source_type": "government"
    },
    {
      "source_id": "W2",
      "claims_supported": [
        "Mexico exported US$3.969 billion of fresh or dried avocados in 2024",
        "Principal 2024 destinations included the United States, Canada, Japan, El Salvador, and Honduras",
        "Michoacán de Ocampo accounted for US$3.525 billion in avocado exports",
        "Mexico was the world’s leading fresh or dried avocado exporter in 2022 ahead of Peru and the Netherlands"
      ],
      "relevance_score": 0.95,
      "source_type": "government"
    },
    {
      "source_id": "W3",
      "claims_supported": [
        "SIAP agricultural statistics cover planted area, harvested area, production volume, and production value",
        "SIAP reports cover all 32 Mexican federal entities",
        "Users can consult statistics for avocado and Hass avocado"
      ],
      "relevance_score": 0.55,
      "source_type": "government"
    },
    {
      "source_id": "W4",
      "claims_supported": [
        "CEC received a February 2023 USMCA/CUSMA submission on avocado production in Michoacán",
        "The submission alleges Mexico is failing to enforce laws protecting forests and water quality",
        "The submission links avocado plantation expansion to deforestation and environmental degradation"
      ],
      "relevance_score": 0.82,
      "source_type": "international_organization"
    },
    {
      "source_id": "W5",
      "claims_supported": [
        "Climate Rights International investigated avocado impacts in Michoacán and Jalisco",
        "The report estimates avocado-driven deforestation likely exceeded 40,000 acres over the preceding decade",
        "About 85% of Michoacán’s avocado production area was certified for U.S. export",
        "The report covers water capture, forest fires, violence, organized crime, and company responsibility"
      ],
      "relevance_score": 0.95,
      "source_type": "ngo"
    },
    {
      "source_id": "W6",
      "claims_supported": [
        "The study connects Mexican avocado exports to U.S. supply chains and deforestation in Michoacán",
        "In 2018, 87% of avocados sold in the United States came from Mexico",
        "The United States consumed roughly three-quarters of Mexican avocado exports",
        "About 20% of Michoacán deforestation from 2001 to 2017 was associated with avocado plantation expansion"
      ],
      "relevance_score": 0.98,
      "source_type": "academic"
    },
    {
      "source_id": "W7",
      "claims_supported": [
        "The review synthesizes environmental and socioeconomic impacts of avocado expansion in Michoacán",
        "Identified impacts include deforestation, forest fragmentation, biodiversity loss, water scarcity, and carbon loss",
        "Socioeconomic effects include job creation, reduced out-migration, inequality, and narco presence"
      ],
      "relevance_score": 0.88,
      "source_type": "academic"
    },
    {
      "source_id": "W8",
      "claims_supported": [
        "The article analyzes avocado frontier expansion in Michoacán before and after NAFTA",
        "Avocado frontiers expanded from 12,909 hectares in 1974 to 152,493 hectares in 2011",
        "The study links tariff reductions and the end of the U.S. ban to increased trade flows from Michoacán to the United States"
      ],
      "relevance_score": 0.84,
      "source_type": "academic"
    },
    {
      "source_id": "W9",
      "claims_supported": [
        "The article models likely future avocado expansion in Michoacán",
        "Demand growth is linked to avocado production expanding into new locations",
        "Proximity to existing agriculture, roads, and localities are strong drivers of expansion"
      ],
      "relevance_score": 0.78,
      "source_type": "academic"
    },
    {
      "source_id": "W10",
      "claims_supported": [
        "The article analyzes future forest-loss risk from avocado expansion in Michoacán",
        "The model projects avocado expansion to 2050",
        "Climate Rights International cites the study as projecting more than 148,200 acres of forest under very high threat of loss"
      ],
      "relevance_score": 0.78,
      "source_type": "academic"
    },
    {
      "source_id": "W11",
      "claims_supported": [
        "The article examines socio-environmental impacts of avocado expansion in the Meseta Purépecha, Michoacán",
        "The avocado boom is framed as driven largely by North American demand in the NAFTA context",
        "Reported impacts include land-use change, water and soil pollution, forest fragmentation, inequality, and weakened community cohesion"
      ],
      "relevance_score": 0.86,
      "source_type": "academic"
    },
    {
      "source_id": "W12",
      "claims_supported": [
        "The study quantifies avocado water footprint in Uruapan, Michoacán from 2012 to 2017",
        "Mean total water footprint was estimated at 744.3 m³ per ton",
        "Irrigated plantations had a much higher footprint than rainfed plantations",
        "In dry years, production can consume up to 120% of legally granted agricultural water volumes"
      ],
      "relevance_score": 0.84,
      "source_type": "academic"
    },
    {
      "source_id": "W13",
      "claims_supported": [
        "The 2024 Water article evaluates avocado water requirements in Ziracuaretiro, Michoacán from 2012 to 2021",
        "Rainfed avocado plantations require 839.03 m³ per ton while irrigated plantations require 2,355.80 m³ per ton",
        "Avocado cultivation can demand up to 124.3% of agricultural water concessions in the municipality"
      ],
      "relevance_score": 0.8,
      "source_type": "academic"
    },
    {
      "source_id": "W14",
      "claims_supported": [
        "The study assesses pesticide risks in the eastern Avocado Belt of Michoacán",
        "It analyzed interviews with 55 smallholder farmers and 16 water samples",
        "Glyphosate, benomyl, and imidacloprid were the most applied pesticides",
        "Water analysis detected 13 pesticides and degradation products"
      ],
      "relevance_score": 0.78,
      "source_type": "academic"
    },
    {
      "source_id": "W15",
      "claims_supported": [
        "The article studies forest fires in the Michoacán Avocado Belt",
        "Fire records from 2000 to 2017 covered 19 municipalities",
        "The presence of avocado orchards was identified as a consistent driver of forest fires across remnant forest patches"
      ],
      "relevance_score": 0.76,
      "source_type": "academic"
    },
    {
      "source_id": "W16",
      "claims_supported": [
        "The article compares carbon storage in avocado orchards and neighboring pine-oak forests in Michoacán",
        "Aboveground carbon is significantly higher in forests than in orchards",
        "The study reports evidence of over-fertilization within orchards"
      ],
      "relevance_score": 0.72,
      "source_type": "academic"
    },
    {
      "source_id": "W17",
      "claims_supported": [
        "Global Forest Watch reports that Mexican avocado production grew 8% annually since 2009",
        "More than three-quarters of Mexico’s avocado production occurs in Michoacán",
        "Avocado production drove 30–40% of recent deforestation in Michoacán",
        "Jalisco production is growing and occurs in regions with high tree-cover loss"
      ],
      "relevance_score": 0.86,
      "source_type": "ngo"
    },
    {
      "source_id": "W18",
      "claims_supported": [
        "AP reports water conflict linked to avocado export-crop expansion in Michoacán",
        "Residents in Villa Madero dismantled illegal water intakes and unlicensed irrigation holding ponds",
        "Drought and rising water use for export crops led by avocados were linked to disappearing rivers and lakes"
      ],
      "relevance_score": 0.72,
      "source_type": "news"
    },
    {
      "source_id": "W19",
      "claims_supported": [
        "AP reports the killing of anti-logging activist Felipe Cisneros in Michoacán",
        "The article states loggers often clear-cut trees to plant avocados",
        "Michoacán was for decades the only Mexican state authorized to export avocados to the U.S. market"
      ],
      "relevance_score": 0.74,
      "source_type": "news"
    },
    {
      "source_id": "W20",
      "claims_supported": [
        "The report page focuses on Mexico’s avocado boom and organized crime in Michoacán",
        "It states avocado production and exports to the United States and European Union show signs of criminal-organization involvement",
        "It links international demand, organized crime groups, human-rights issues, and environmental impacts"
      ],
      "relevance_score": 0.84,
      "source_type": "ngo"
    },
    {
      "source_id": "W21",
      "claims_supported": [
        "The article examines organized criminal expansion into licit commodity markets using Mexican avocados as a central case",
        "It uses Mexican avocado export data and municipal homicide data",
        "Increases in a municipality’s share of avocado export value are associated with homicides where organized criminal groups are present"
      ],
      "relevance_score": 0.76,
      "source_type": "academic"
    },
    {
      "source_id": "W22",
      "claims_supported": [
        "The article examines organized-crime rent extraction from Michoacán’s legal avocado export market from 2001 to 2014",
        "It focuses on the Knights Templar criminal organization",
        "The study assesses how extortion drained money from the legal economy"
      ],
      "relevance_score": 0.72,
      "source_type": "academic"
    },
    {
      "source_id": "W23",
      "claims_supported": [
        "Michoacán state government reported 90% of avocado exports to the United States were free of deforestation under certification",
        "Thirty-seven packing houses in Michoacán and Jalisco had obtained compliance certification",
        "Certification excludes deforestation after January 2018 and properties affected by fires since January 2012",
        "More than 900,000 avocado harvest checks had been conducted using the Guardián Forestal system"
      ],
      "relevance_score": 0.78,
      "source_type": "government"
    },
    {
      "source_id": "W24",
      "claims_supported": [
        "The article studies a community model of avocado production in Indigenous communities in Zitácuaro, Michoacán",
        "Avocado revenues support community institutions such as forest monitoring and water committees",
        "Local rules limit communal forest parcelization and restrict sale or rental of communal lands to outsiders",
        "Sustainability concerns include unequal water access, high agrochemical use, and dependence on technical advisers"
      ],
      "relevance_score": 0.76,
      "source_type": "academic"
    },
    {
      "source_id": "W25",
      "claims_supported": [
        "Avocado Institute page addresses Mexican avocado industry efforts toward deforestation-free exports"
      ],
      "relevance_score": 0.4,
      "source_type": "industry"
    }
  ],
  "suggested_followup_queries": [
    "Mexico avocado exports Europe destination countries",
    "Independent audits Mexican avocado deforestation-free certification",
    "Jalisco avocado deforestation export supply chains",
    "Michoacan avocado water conflicts export orchards",
    "Retailer sustainability policies Mexican avocados"
  ]
}
```

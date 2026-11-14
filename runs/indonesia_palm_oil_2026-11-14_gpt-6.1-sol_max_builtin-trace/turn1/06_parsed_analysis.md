# 06 — Parsed analysis

`result.parsed` re-parsed into structured fields.

```json
{
  "country_pericoupling_info": {
    "focal_country": "Indonesia (IDN)",
    "mode": "national",
    "pair_results": "Indonesia (IDN) ↔ India (IND) (at sea): PERICOUPLED; Indonesia (IDN) ↔ Malaysia (MYS): PERICOUPLED; Indonesia (IDN) ↔ China (CHN): TELECOUPLED; Indonesia (IDN) ↔ Pakistan (PAK): TELECOUPLED; Indonesia (IDN) ↔ Bangladesh (BGD): TELECOUPLED; Indonesia (IDN) ↔ United States (USA): TELECOUPLED; Indonesia (IDN) ↔ European Union: TELECOUPLED",
    "core_subnational_regions": "Jambi, Kalimantan Barat",
    "note": "LLM classification is consistent with the coupling database."
  },
  "coupling_classification": "**Indonesia’s palm oil sector involves all three coupling types.** Its sustainability cannot be explained by exports alone: domestic food and biodiesel demand, neighboring markets, and distant consumers jointly influence production and its consequences.\n\nThis is a **national-level scoping analysis**. Indonesia is the focal coupled human and natural system (CHANS); movements entirely within Indonesia are therefore intracoupling. Findings below come from different periods and methods, not a single harmonized assessment.\n\n- **Intracoupling:** Plantation management, processing, domestic consumption, employment, investment, and interactions with forests, peatlands, and watersheds within Indonesia.\n- **Pericoupling:** Documented palm-product exports to **Malaysia and India**. Under the supplied country-level adjacency convention, shared maritime boundaries count as adjacency, so India is classified here rather than under telecoupling. Actual exports—not adjacency alone—establish these connections. [T1:W3]\n- **Telecoupling:** Palm-product exports to **China, Pakistan, Bangladesh, the European Union, and the United States**, together with reverse payments and international policy and traceability information. [T1:4] [T1:W3]\n\nThe principal telecoupling category is **Trade**. For scale, GAPKI reports approximately **32.343 million tonnes of palm-sector exports and US$35.87 billion in export earnings in 2025**. These totals cover several crude and processed product categories; they are not exclusively crude palm oil or measures of net social benefit. [T1:W3]",
  "cross_coupling_interactions": [
    "**Amplification / offset:** Adjacent and distant demand can amplify domestic plantation and processing activity, while domestic biodiesel demand can compete with exports for feedstock. Export levies also finance domestic biodiesel subsidies, directly connecting cross-border capital with within-country incentives. However, domestic consumption and exports both increased in the supplied 2025 statistics, so greater domestic use should not automatically be interpreted as reducing exports. [T1:W3] [T1:W20]",
    "**Spatial tradeoffs:** Importing systems obtain palm products while many forest, water, and land-conversion burdens occur in Indonesian producing landscapes. Within Indonesia, benefits and costs also differ geographically and socially: market-connected communities and remote forest-dependent communities have experienced contrasting welfare outcomes. Aggregate export earnings therefore cannot establish equitable sustainability. [T1:1] [T1:W13]",
    "**Spillover effects:** International certification can influence clearing both inside certified supply bases and elsewhere in Indonesian landscapes. A review reports positive and negative effects around certified operations in Kalimantan, with an insignificant overall net effect in the summarized analysis. At the national boundary used here, these are within-Indonesia consequences associated with an internationally connected intervention—not proof that reductions inside certified estates equal reductions across the wider landscape. [T1:5]"
  ],
  "evidence_coverage_note": "**Strong multi-source support:** The connection between oil-palm expansion and forest, peat, biodiversity, and emissions impacts is supported across the retrieved study and several environmental studies. Evidence also supports heterogeneous social outcomes rather than uniform poverty reduction or uniform harm. [T1:1] [T1:W4] [T1:W6] [T1:W7] [T1:W13]\n\n**Limited or scale-specific evidence:** The quoted 2025 market totals come from an industry association, not an independent sustainability evaluation. Kais results are modeled catchment outcomes, Jambi biodiesel findings concern particular production conditions, and the certification estimates cover historical periods. These findings should not be combined as interchangeable national or current-year measurements. [T1:W3] [T1:W9] [T1:W7] [T1:W16]\n\n**Inferred without direct measurement:** Bilateral financial distributions, environmental changes within importing countries, destination-specific deforestation footprints, and Malaysia’s competitive response remain unquantified. Malaysia is therefore a **candidate**, not a confirmed spillover system. The preparedness exercise establishes shared sourcing and implementation challenges, not displaced production or forest loss. Some retrieved passages are also summarized or truncated, limiting detailed verification. [T1:W24]",
  "intracoupling": {
    "systems": [
      {
        "role": "focal",
        "name": "Indonesia’s palm oil production–consumption system",
        "human_subsystem": "Independent and scheme smallholders, plantation workers, private and state-owned plantation companies, mills, refineries, domestic consumers, biodiesel producers, and public agencies.",
        "natural_subsystem": "Oil-palm agroecosystems, remaining forests, peatlands, biodiversity, soils, rivers, and groundwater.",
        "geographic_scope": "Indonesia nationally, with differentiated production landscapes, including Sumatra and Kalimantan and the studied Kais watershed in West Papua. [T1:2] [T1:W9]"
      }
    ],
    "flows": [
      {
        "category": "matter",
        "direction": "Plantations → mills → refineries → domestic buyers and export terminals.",
        "description": "Fresh fruit bunches, crude palm oil, and processed palm products. GAPKI reports domestic consumption of 24.772 million tonnes in 2025, including 12.704 million tonnes for biodiesel. [T1:W3]"
      },
      {
        "category": "matter",
        "direction": "Plantation inputs and mill processes → soils, drainage networks, and receiving waters.",
        "description": "Fertilizers, nutrient runoff, sediment, and processing effluent. Water-pollution pathways are documented or modeled, but their magnitude varies by location. [T1:1] [T1:W8] [T1:W9]"
      },
      {
        "category": "capital",
        "direction": "Domestic buyers → producers and processors; firms → workers and suppliers; Indonesian exporters → public funding mechanisms → biodiesel producers.",
        "description": "Product payments, wages, investment, export-levy receipts, and biodiesel subsidies. Export levies financing domestic subsidies connect international trade to domestic production incentives. [T1:W20]"
      },
      {
        "category": "energy",
        "direction": "Domestic biodiesel producers → fuel distribution and transport users.",
        "description": "Palm-derived transport energy. Feedstock tonnage should not be confused with an energy quantity or a demonstrated emissions saving. [T1:W3] [T1:W7]"
      },
      {
        "category": "information",
        "direction": "Farms and mills ↔ cooperatives, companies, auditors, workers’ organizations, and public agencies.",
        "description": "Production records, land information, environmental monitoring, training, and compliance information. Evidence indicates substantial weaknesses in information reaching independent smallholders. [T1:W15] [T1:W18]"
      },
      {
        "category": "people",
        "direction": "Indonesian households and settlements → plantations and processing workplaces.",
        "description": "Labor movement and recruitment within the national focal system. Employment is important, but employment quantity does not establish job quality. [T1:2] [T1:W15]"
      }
    ],
    "agents": [
      {
        "level": "individuals / households",
        "name": "**Smallholders, workers, and local residents**",
        "description": "Decide on cultivation, labor allocation, land transactions, and participation in producer organizations."
      },
      {
        "level": "firms / traders / corporations",
        "name": "**Plantation companies, mills, refineries, and biodiesel producers**",
        "description": "Organize production, purchasing, processing, investment, and employment."
      },
      {
        "level": "governments / policymakers",
        "name": "**National and subnational authorities**",
        "description": "Shape concessions, land governance, labor enforcement, environmental regulation, and biodiesel support."
      },
      {
        "level": "organizations / NGOs",
        "name": "**Cooperatives, worker organizations, NGOs, researchers, and certification bodies**",
        "description": "Facilitate information, monitoring, collective representation, and assistance."
      }
    ],
    "causes": {
      "economic": [
        "Returns from domestic food, industrial, fuel, and export markets encourage cultivation and processing. Historical research links palm oil prices with plantation expansion and forest loss, although correlation alone does not establish every causal pathway. [T1:W3] [T1:W4]"
      ],
      "political / institutional": [
        "Biodiesel mandates and levy-funded subsidies create domestic demand and redistribute sector revenue. [T1:W20]",
        "Concession allocation, tenure arrangements, and enforcement influence where expansion occurs and who controls land. A West Kalimantan investigation documents allegations of inadequate consultation, compensation, and remedies. [T1:W14]"
      ],
      "technological / infrastructural": [
        "Mills, refineries, and processing capacity enable commercialization. The literature documents rapid historical expansion of Indonesian refining capacity. [T1:2]",
        "Effluent treatment and biogas infrastructure influence the environmental consequences of processing. [T1:W7] [T1:W8]"
      ]
    },
    "effects": {
      "economic": [
        "**Indonesia:** Historical district-level evidence identifies higher agricultural labor returns, faster poverty reduction, and increased public revenue associated with expansion. These findings do not imply equal benefits for every household or future expansion. [T1:W12]"
      ],
      "ecological / biological": [
        "**Indonesia:** Forest conversion and habitat simplification reduce ecological integrity. One nationwide study linked same-year conversion to oil palm with approximately **29% of old-growth forest loss during 2001–2019**. This is not equivalent to attributing all Indonesian deforestation to palm oil. [T1:W4]",
        "**Indonesia:** Retaining nearby forest, reducing excessive management intensity, and enriching plantations with native trees can improve ecological outcomes under studied conditions; plantations nevertheless do not replace natural forests. [T1:W6] [T1:W11]"
      ],
      "cultural / social / demographic": [
        "**Indonesia:** Welfare outcomes differ by starting conditions. Kalimantan villages with established market livelihoods experienced different outcomes from remote, forest-dependent communities. [T1:W13]",
        "**Indonesia:** Land-rights disputes and decent-work risks can accompany economic gains. The ILO identifies safety, wages, working hours, discrimination, and organizing rights as important concerns, without estimating their national prevalence. [T1:W14] [T1:W15]"
      ],
      "hydrological": [
        "**Indonesia:** Conversion can change runoff and nutrient loading. In the **Kais watershed model**, runoff increased by 21%, total nitrogen by 78%, and total phosphorus by 144%. These are catchment-specific modeled changes, not national averages. [T1:W9]"
      ],
      "climatic / atmospheric": [
        "**Indonesia:** Forest conversion, peat drainage, fire, and mill-effluent methane create emissions. [T1:W6] [T1:W8]",
        "**Indonesia:** Biodiesel is not automatically climate beneficial. A Jambi-based life-cycle study found no greenhouse-gas savings for first-rotation plantations established following forest clearance under the scenarios examined. [T1:W7]"
      ],
      "geological / geomorphological": [
        "**Indonesia:** Peat subsidence and sediment production can impose persistent costs. Evidence includes the West Kalimantan case investigation and modeled sediment increases in Kais; neither establishes a uniform national effect. [T1:W14] [T1:W9]"
      ]
    }
  },
  "is_parsed": true,
  "map_data": {
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
  },
  "pericoupling": {
    "systems": [
      {
        "role": "sending",
        "system_scope": "adjacent",
        "name": "Indonesia — palm products"
      },
      {
        "role": "receiving",
        "system_scope": "adjacent",
        "name": "Indonesia — payments",
        "human_subsystem": "Producers, processors, exporters, workers, and relevant authorities.",
        "natural_subsystem": "Plantation landscapes and connected forests, peatlands, soils, and watersheds.",
        "geographic_scope": "Indonesia nationally."
      },
      {
        "role": "receiving",
        "system_scope": "adjacent",
        "name": "India — palm products"
      },
      {
        "role": "sending",
        "system_scope": "adjacent",
        "name": "India — payments",
        "human_subsystem": "Vegetable-oil importers, processors, consumers, and trade authorities.",
        "natural_subsystem": "Domestic agricultural land and water resources potentially affected by the balance between imported and domestically produced oils; changes are not quantified here.",
        "geographic_scope": "India nationally; adjacency follows the supplied shared-maritime-boundary classification."
      },
      {
        "role": "receiving",
        "system_scope": "adjacent",
        "name": "Malaysia — palm products"
      },
      {
        "role": "sending",
        "system_scope": "adjacent",
        "name": "Malaysia — payments",
        "human_subsystem": "Importers, traders, processors, consumers, and regulators.",
        "natural_subsystem": "Oil-palm landscapes, forests, peatlands, and associated water resources; the effects specifically attributable to Indonesian imports remain unmeasured.",
        "geographic_scope": "Malaysia nationally, including Peninsular Malaysia and Malaysian Borneo.",
        "indonesian_exports_to_both_countries_are_reported_in_the_supplied_industry_statistics._malaysia’s_role_as_an_importer_should_not_be_confused_with_its_separate_role_as_a_major_producer._[t1": "W3] [T1:W6]"
      }
    ],
    "flows": [
      {
        "category": "matter",
        "direction": "Indonesia → India; Indonesia → Malaysia.",
        "description": "Palm-sector products. The supplied 2025 release reports declining exports to India and increasing exports to Malaysia, but does not provide bilateral volumes in its summary. [T1:W3]"
      },
      {
        "category": "capital",
        "direction": "Indian and Malaysian purchasers → Indonesian exporters.",
        "description": "Trade payments. Their distribution among Indonesian companies, smallholders, workers, and government requires additional evidence."
      }
    ],
    "agents": [
      {
        "level": "individuals / households",
        "name": "**Indonesian producers and workers; purchasing-country consumers**",
        "description": "Participate in production and consumption decisions."
      },
      {
        "level": "firms / traders / corporations",
        "name": "**Exporters, importers, processors, and intermediaries**",
        "description": "Arrange purchases, logistics, and product allocation."
      },
      {
        "level": "governments / policymakers",
        "name": "**Indonesian, Indian, and Malaysian authorities**",
        "description": "Influence trade conditions, domestic market allocation, and sector governance."
      }
    ],
    "causes": {
      "economic": [
        "Demand from adjacent markets connects Indonesian production to regional purchasing decisions. Observed changes in exports establish the flows, but do not by themselves identify why each destination’s purchases changed. [T1:W3]"
      ],
      "political / institutional": [
        "Indonesian export levies and domestic biodiesel support affect the incentives governing allocation between domestic use and exports, including exports to neighbors. Partner-specific effects are not estimated in the supplied evidence. [T1:W20]"
      ],
      "technological / infrastructural": [
        "Processing and logistics capacity enable regional exchange. Indonesian processing expansion is documented historically, but the evidence does not isolate infrastructure’s contribution to either bilateral route. [T1:2]"
      ]
    },
    "effects": {
      "economic": [
        "**Indonesia, sending products:** Receives export revenue.",
        "**India and Malaysia, receiving products:** Obtain supplies for consumption, processing, or onward trade. Final uses and welfare effects require bilateral product and end-use data."
      ],
      "ecological / biological": [
        "**Indonesia, sending products:** Demand from neighboring markets can contribute to plantation-related land pressure. Existing national forest-loss estimates cannot be assigned to India or Malaysia without sourcing information. [T1:1] [T1:W4]"
      ],
      "hydrological": [
        "**Indonesia, sending products:** Water-use and pollution burdens associated with production occur in producing landscapes. Attribution to adjacent buyers requires tracing their purchases to particular locations. [T1:1]"
      ],
      "climatic / atmospheric": [
        "**Indonesia, sending products:** Production may generate land-use, peat, and processing emissions.",
        "**India and Malaysia, receiving products:** Consumption-based accounting can attribute a portion of those emissions to their purchases. This is an accounting relationship—not evidence that emissions physically travel inside the oil. [T1:4]"
      ]
    }
  },
  "pericoupling_info": null,
  "raw_text": "## 1. Coupling Classification\n\n**Indonesia’s palm oil sector involves all three coupling types.** Its sustainability cannot be explained by exports alone: domestic food and biodiesel demand, neighboring markets, and distant consumers jointly influence production and its consequences.\n\nThis is a **national-level scoping analysis**. Indonesia is the focal coupled human and natural system (CHANS); movements entirely within Indonesia are therefore intracoupling. Findings below come from different periods and methods, not a single harmonized assessment.\n\n- **Intracoupling:** Plantation management, processing, domestic consumption, employment, investment, and interactions with forests, peatlands, and watersheds within Indonesia.\n- **Pericoupling:** Documented palm-product exports to **Malaysia and India**. Under the supplied country-level adjacency convention, shared maritime boundaries count as adjacency, so India is classified here rather than under telecoupling. Actual exports—not adjacency alone—establish these connections. [T1:W3]\n- **Telecoupling:** Palm-product exports to **China, Pakistan, Bangladesh, the European Union, and the United States**, together with reverse payments and international policy and traceability information. [T1:4] [T1:W3]\n\nThe principal telecoupling category is **Trade**. For scale, GAPKI reports approximately **32.343 million tonnes of palm-sector exports and US$35.87 billion in export earnings in 2025**. These totals cover several crude and processed product categories; they are not exclusively crude palm oil or measures of net social benefit. [T1:W3]\n\n## 2. Intracoupling Analysis — within Indonesia\n\n### 2.1 Systems Identification\n\n**Focal System: Indonesia’s palm oil production–consumption system**\n\n- **Human subsystem:** Independent and scheme smallholders, plantation workers, private and state-owned plantation companies, mills, refineries, domestic consumers, biodiesel producers, and public agencies.\n- **Natural subsystem:** Oil-palm agroecosystems, remaining forests, peatlands, biodiversity, soils, rivers, and groundwater.\n- **Geographic scope:** Indonesia nationally, with differentiated production landscapes, including Sumatra and Kalimantan and the studied Kais watershed in West Papua. [T1:2] [T1:W9]\n\n### 2.2 Flows Analysis\n\n**Matter Flow**\n- **Direction:** Plantations → mills → refineries → domestic buyers and export terminals.\n- **Description:** Fresh fruit bunches, crude palm oil, and processed palm products. GAPKI reports domestic consumption of 24.772 million tonnes in 2025, including 12.704 million tonnes for biodiesel. [T1:W3]\n\n**Matter Flow**\n- **Direction:** Plantation inputs and mill processes → soils, drainage networks, and receiving waters.\n- **Description:** Fertilizers, nutrient runoff, sediment, and processing effluent. Water-pollution pathways are documented or modeled, but their magnitude varies by location. [T1:1] [T1:W8] [T1:W9]\n\n**Capital Flow**\n- **Direction:** Domestic buyers → producers and processors; firms → workers and suppliers; Indonesian exporters → public funding mechanisms → biodiesel producers.\n- **Description:** Product payments, wages, investment, export-levy receipts, and biodiesel subsidies. Export levies financing domestic subsidies connect international trade to domestic production incentives. [T1:W20]\n\n**Energy Flow**\n- **Direction:** Domestic biodiesel producers → fuel distribution and transport users.\n- **Description:** Palm-derived transport energy. Feedstock tonnage should not be confused with an energy quantity or a demonstrated emissions saving. [T1:W3] [T1:W7]\n\n**Information Flow**\n- **Direction:** Farms and mills ↔ cooperatives, companies, auditors, workers’ organizations, and public agencies.\n- **Description:** Production records, land information, environmental monitoring, training, and compliance information. Evidence indicates substantial weaknesses in information reaching independent smallholders. [T1:W15] [T1:W18]\n\n**People Flow**\n- **Direction:** Indonesian households and settlements → plantations and processing workplaces.\n- **Description:** Labor movement and recruitment within the national focal system. Employment is important, but employment quantity does not establish job quality. [T1:2] [T1:W15]\n\n### 2.3 Agents\n\n- [Individuals / households] **Smallholders, workers, and local residents:** Decide on cultivation, labor allocation, land transactions, and participation in producer organizations.\n- [Firms / traders / corporations] **Plantation companies, mills, refineries, and biodiesel producers:** Organize production, purchasing, processing, investment, and employment.\n- [Governments / policymakers] **National and subnational authorities:** Shape concessions, land governance, labor enforcement, environmental regulation, and biodiesel support.\n- [Organizations / NGOs] **Cooperatives, worker organizations, NGOs, researchers, and certification bodies:** Facilitate information, monitoring, collective representation, and assistance.\n\nOil palms are treated here primarily as crops and resources; the supplied evidence does not establish a distinct non-human-agent mechanism requiring separate analysis.\n\n### 2.4 Causes\n\n**Economic**\n- Returns from domestic food, industrial, fuel, and export markets encourage cultivation and processing. Historical research links palm oil prices with plantation expansion and forest loss, although correlation alone does not establish every causal pathway. [T1:W3] [T1:W4]\n\n**Political / Institutional**\n- Biodiesel mandates and levy-funded subsidies create domestic demand and redistribute sector revenue. [T1:W20]\n- Concession allocation, tenure arrangements, and enforcement influence where expansion occurs and who controls land. A West Kalimantan investigation documents allegations of inadequate consultation, compensation, and remedies. [T1:W14]\n\n**Technological / Infrastructural**\n- Mills, refineries, and processing capacity enable commercialization. The literature documents rapid historical expansion of Indonesian refining capacity. [T1:2]\n- Effluent treatment and biogas infrastructure influence the environmental consequences of processing. [T1:W7] [T1:W8]\n\n### 2.5 Effects\n\n**Economic**\n- **Indonesia:** Historical district-level evidence identifies higher agricultural labor returns, faster poverty reduction, and increased public revenue associated with expansion. These findings do not imply equal benefits for every household or future expansion. [T1:W12]\n\n**Ecological / Biological**\n- **Indonesia:** Forest conversion and habitat simplification reduce ecological integrity. One nationwide study linked same-year conversion to oil palm with approximately **29% of old-growth forest loss during 2001–2019**. This is not equivalent to attributing all Indonesian deforestation to palm oil. [T1:W4]\n- **Indonesia:** Retaining nearby forest, reducing excessive management intensity, and enriching plantations with native trees can improve ecological outcomes under studied conditions; plantations nevertheless do not replace natural forests. [T1:W6] [T1:W11]\n\n**Cultural / Social / Demographic**\n- **Indonesia:** Welfare outcomes differ by starting conditions. Kalimantan villages with established market livelihoods experienced different outcomes from remote, forest-dependent communities. [T1:W13]\n- **Indonesia:** Land-rights disputes and decent-work risks can accompany economic gains. The ILO identifies safety, wages, working hours, discrimination, and organizing rights as important concerns, without estimating their national prevalence. [T1:W14] [T1:W15]\n\n**Hydrological**\n- **Indonesia:** Conversion can change runoff and nutrient loading. In the **Kais watershed model**, runoff increased by 21%, total nitrogen by 78%, and total phosphorus by 144%. These are catchment-specific modeled changes, not national averages. [T1:W9]\n\n**Climatic / Atmospheric**\n- **Indonesia:** Forest conversion, peat drainage, fire, and mill-effluent methane create emissions. [T1:W6] [T1:W8]\n- **Indonesia:** Biodiesel is not automatically climate beneficial. A Jambi-based life-cycle study found no greenhouse-gas savings for first-rotation plantations established following forest clearance under the scenarios examined. [T1:W7]\n\n**Geological / Geomorphological**\n- **Indonesia:** Peat subsidence and sediment production can impose persistent costs. Evidence includes the West Kalimantan case investigation and modeled sediment increases in Kais; neither establishes a uniform national effect. [T1:W14] [T1:W9]\n\n## 3. Pericoupling Analysis — adjacent systems\n\n### 3.1 Systems Identification\n\nThe paired headings identify each system’s role for different flows. Reverse payments are the commercial counterpart of documented exports; bilateral payment amounts are unavailable.\n\n**Sending System (adjacent): Indonesia — palm products**\n**Receiving System (adjacent): Indonesia — payments**\n- **Human subsystem:** Producers, processors, exporters, workers, and relevant authorities.\n- **Natural subsystem:** Plantation landscapes and connected forests, peatlands, soils, and watersheds.\n- **Geographic scope:** Indonesia nationally.\n\n**Receiving System (adjacent): India — palm products**\n**Sending System (adjacent): India — payments**\n- **Human subsystem:** Vegetable-oil importers, processors, consumers, and trade authorities.\n- **Natural subsystem:** Domestic agricultural land and water resources potentially affected by the balance between imported and domestically produced oils; changes are not quantified here.\n- **Geographic scope:** India nationally; adjacency follows the supplied shared-maritime-boundary classification.\n\n**Receiving System (adjacent): Malaysia — palm products**\n**Sending System (adjacent): Malaysia — payments**\n- **Human subsystem:** Importers, traders, processors, consumers, and regulators.\n- **Natural subsystem:** Oil-palm landscapes, forests, peatlands, and associated water resources; the effects specifically attributable to Indonesian imports remain unmeasured.\n- **Geographic scope:** Malaysia nationally, including Peninsular Malaysia and Malaysian Borneo.\n\nIndonesian exports to both countries are reported in the supplied industry statistics. Malaysia’s role as an importer should not be confused with its separate role as a major producer. [T1:W3] [T1:W6]\n\n### 3.2 Flows Analysis\n\n**Matter Flow**\n- **Direction:** Indonesia → India; Indonesia → Malaysia.\n- **Description:** Palm-sector products. The supplied 2025 release reports declining exports to India and increasing exports to Malaysia, but does not provide bilateral volumes in its summary. [T1:W3]\n\n**Capital Flow**\n- **Direction:** Indian and Malaysian purchasers → Indonesian exporters.\n- **Description:** Trade payments. Their distribution among Indonesian companies, smallholders, workers, and government requires additional evidence.\n\n### 3.3 Agents\n\n- [Individuals / households] **Indonesian producers and workers; purchasing-country consumers:** Participate in production and consumption decisions.\n- [Firms / traders / corporations] **Exporters, importers, processors, and intermediaries:** Arrange purchases, logistics, and product allocation.\n- [Governments / policymakers] **Indonesian, Indian, and Malaysian authorities:** Influence trade conditions, domestic market allocation, and sector governance.\n\n### 3.4 Causes\n\n**Economic**\n- Demand from adjacent markets connects Indonesian production to regional purchasing decisions. Observed changes in exports establish the flows, but do not by themselves identify why each destination’s purchases changed. [T1:W3]\n\n**Political / Institutional**\n- Indonesian export levies and domestic biodiesel support affect the incentives governing allocation between domestic use and exports, including exports to neighbors. Partner-specific effects are not estimated in the supplied evidence. [T1:W20]\n\n**Technological / Infrastructural**\n- Processing and logistics capacity enable regional exchange. Indonesian processing expansion is documented historically, but the evidence does not isolate infrastructure’s contribution to either bilateral route. [T1:2]\n\n### 3.5 Effects\n\n**Economic**\n- **Indonesia, sending products:** Receives export revenue.\n- **India and Malaysia, receiving products:** Obtain supplies for consumption, processing, or onward trade. Final uses and welfare effects require bilateral product and end-use data.\n\n**Ecological / Biological**\n- **Indonesia, sending products:** Demand from neighboring markets can contribute to plantation-related land pressure. Existing national forest-loss estimates cannot be assigned to India or Malaysia without sourcing information. [T1:1] [T1:W4]\n\n**Hydrological**\n- **Indonesia, sending products:** Water-use and pollution burdens associated with production occur in producing landscapes. Attribution to adjacent buyers requires tracing their purchases to particular locations. [T1:1]\n\n**Climatic / Atmospheric**\n- **Indonesia, sending products:** Production may generate land-use, peat, and processing emissions.\n- **India and Malaysia, receiving products:** Consumption-based accounting can attribute a portion of those emissions to their purchases. This is an accounting relationship—not evidence that emissions physically travel inside the oil. [T1:4]\n\n## 4. Telecoupling Analysis — distant systems\n\n### 4.1 Systems Identification\n\nFor importing systems, the natural-subsystem descriptions identify relevant assessment domains. The supplied sources generally do **not** quantify environmental changes within those countries caused by Indonesian imports.\n\n**Sending System (distant): Indonesia — palm products and supply-chain information**\n**Receiving System (distant): Indonesia — payments and policy/sourcing information**\n- **Human subsystem:** Producers, workers, exporters, processors, public agencies, and producer organizations.\n- **Natural subsystem:** Plantations, forests, peatlands, biodiversity, soils, and water systems.\n- **Geographic scope:** Indonesia nationally.\n\n**Receiving System (distant): China — palm products**\n**Sending System (distant): China — payments**\n- **Human subsystem:** Importers, processors, food and industrial users, consumers, and regulators.\n- **Natural subsystem:** Domestic agricultural ecosystems, soils, and freshwater associated with vegetable-oil supply.\n- **Geographic scope:** China nationally. China’s role as a destination for processed Indonesian palm oil is also identified in the literature. [T1:2]\n\n**Receiving System (distant): Pakistan — palm products**\n**Sending System (distant): Pakistan — payments**\n- **Human subsystem:** Importers, processors, food-sector businesses, consumers, and trade authorities.\n- **Natural subsystem:** Agricultural land, soils, and freshwater associated with domestic food and oil production.\n- **Geographic scope:** Pakistan nationally.\n\n**Receiving System (distant): Bangladesh — palm products**\n**Sending System (distant): Bangladesh — payments**\n- **Human subsystem:** Importers, food processors, consumers, and relevant authorities.\n- **Natural subsystem:** Agricultural ecosystems and freshwater resources connected to edible-oil supply.\n- **Geographic scope:** Bangladesh nationally.\n\n**Receiving System (distant): European Union — palm products and traceability information**\n**Sending System (distant): European Union — payments and regulatory information**\n- **Human subsystem:** Importers, manufacturers, consumers, the European Commission, and national competent authorities.\n- **Natural subsystem:** Oil-crop agroecosystems, freshwater resources, and environmental processes affected by food and fuel choices.\n- **Geographic scope:** EU member states and their collective market.\n\n**Receiving System (distant): United States — palm products**\n**Sending System (distant): United States — payments and relevant policy information**\n- **Human subsystem:** Importers, manufacturers, consumers, and public agencies.\n- **Natural subsystem:** Oil-crop agroecosystems, freshwater resources, and environmental processes associated with alternative food and fuel supplies.\n- **Geographic scope:** United States nationally.\n\nExports to these five destination systems are reported in the supplied statistics. [T1:W3]\n\n**Spillover System: Malaysia as a competing producer — candidate, not established**\n- **Human subsystem:** Malaysian plantation companies, smallholders, workers, refiners, and exporters that could experience changes in competition.\n- **Natural subsystem:** Existing plantations and potentially affected forests, peatlands, soils, and watersheds.\n- **Geographic scope:** Malaysian production landscapes.\n- **Assessment:** Malaysia’s direct importing role is established above. An additional, indirect effect from changes in Indonesia’s exports to distant buyers is plausible but **not demonstrated**. Evidence that EU operators source from both countries does not itself prove a competitive spillover. [T1:W24]\n\n### 4.2 Flows Analysis\n\n**Matter Flow**\n- **Direction:** Indonesia → China, Pakistan, Bangladesh, EU, and United States.\n- **Description:** Crude and processed palm-sector products. Destination-specific product composition and sourcing locations must be established rather than inferred from national totals. [T1:W3]\n\n**Capital Flow**\n- **Direction:** Distant purchasers → Indonesian exporters.\n- **Description:** Export payments, with subsequent domestic distribution through purchases, wages, investment, and public revenue. International financiers and land investors are also identified qualitatively in the literature, but their origins and amounts are not established by the supplied passages. [T1:4]\n\n**Information Flow**\n- **Direction:** Foreign policymakers and buyers → Indonesian agencies and suppliers; Indonesian suppliers → importing operators.\n- **Description:** Biofuel-policy signals, sourcing requirements, geolocation records, legality information, and due-diligence documentation. Historical EU/US bioenergy demand policies and present deforestation-related requirements should be distinguished rather than treated as one policy mechanism. [T1:4] [T1:W23] [T1:W24]\n\n**Matter Flow**\n- **Direction:** Indonesian production → distant consumption, through embodied-resource accounting.\n- **Description:** Virtual-water requirements and pollution footprints attributed to traded oil. **Grey water is a pollution-assimilation indicator, not a literal volume of polluted water shipped abroad.** Embodied emissions likewise allocate responsibility without implying physical shipment of CO₂ in products. [T1:1] [T1:4]\n\n**Information Flow**\n- **Direction:** Changes in distant-market purchasing → market signals → Malaysian producers.\n- **Description:** A proposed pathway for the candidate spillover system. Its strength, direction, and resulting production response remain unverified.\n\n### 4.3 Agents\n\n- [Individuals / households] **Indonesian smallholders and workers; distant consumers:** Influence production, labor, and purchasing decisions.\n- [Firms / traders / corporations] **Exporters, international traders, importers, manufacturers, financiers, and intermediaries:** Organize trade, investment, sourcing, and processing.\n- [Governments / policymakers] **Indonesia and importing-market authorities:** Shape trade, energy, environmental, and due-diligence conditions.\n- [Organizations / NGOs] **RSPO, certification bodies, NGOs, research organizations, and the ILO:** Develop standards, assess outcomes, support implementation, and represent environmental or social concerns.\n\n### 4.4 Causes\n\n**Economic**\n- International food, industrial, and biofuel demand links distant consumption to Indonesian production. China’s demand includes processed palm products. [T1:4] [T1:2]\n\n**Political / Institutional**\n- Historical foreign renewable-energy policies contributed to feedstock demand. [T1:4]\n- Deforestation-related import requirements instead condition market access. The EUDR requires covered products to meet deforestation-free and legality conditions and provide required information, including harvest-plot geolocation. These legal requirements are not evidence of realized conservation outcomes. [T1:W23]\n\n**Technological / Infrastructural**\n- Processing capacity enables participation in different product markets. [T1:2]\n- Traceability systems and data transfer influence the feasibility of demonstrating compliance. An EUDR preparedness exercise identified practical geolocation and documentation difficulties; it was not an enforcement determination. [T1:W24]\n\n### 4.5 Effects\n\n**Economic**\n- **Indonesia, sending products:** Gains gross export earnings, while benefits depend on production costs, ownership, wages, taxation, and distribution. [T1:W3]\n- **Distant receiving systems:** Obtain industrial and food inputs; net consumer and producer welfare effects are not quantified here.\n- **Malaysia, candidate spillover:** Could gain or lose market share following changes in Indonesian trade. Neither direction nor magnitude is established.\n\n**Political / Institutional**\n- **Indonesia and EU:** Due-diligence requirements create information and administrative responsibilities across the trade relationship. [T1:W23] [T1:W24]\n- **Indonesia:** Certification has produced some improvements, but not universal protection. Historical research estimated lower deforestation within certified estates while finding no detectable effect on peatland forest loss or active fires. [T1:W16]\n\n**Ecological / Biological**\n- **Indonesia, sending products:** Export demand is connected to forest conversion and fragmentation through production expansion; destination-specific attribution remains necessary. [T1:1]\n- **Malaysia, candidate spillover:** Any indirect production response could alter pressure on forests and peatlands, but no such response is demonstrated by the supplied evidence.\n\n**Hydrological**\n- **Indonesia, sending products:** Production-related water scarcity and pollution can be attributed partly to distant consumption through footprint analysis, without relocating the actual environmental damage to importing countries. [T1:1]\n\n**Climatic / Atmospheric**\n- **Indonesia, sending products:** Land conversion, peat processes, and mill methane contribute emissions associated with exported products. [T1:1] [T1:W8]\n- **Receiving systems:** Climate claims about palm-based fuels depend on production history and full accounting. Jambi findings caution against assuming savings when plantations replace forest. [T1:W7]\n\n**Cultural / Social / Demographic**\n- **Indonesia:** Economic benefits coexist with unequal welfare outcomes, labor risks, and land-rights concerns. [T1:W13] [T1:W14] [T1:W15]\n- **Indonesian smallholders:** Corporate commitments do not automatically deliver training, incentives, or meaningful participation. A recent study found particularly limited reach among independent smallholders. [T1:W18]\n\n## 5. Cross-coupling Interactions\n\n- **Amplification / offset:** Adjacent and distant demand can amplify domestic plantation and processing activity, while domestic biodiesel demand can compete with exports for feedstock. Export levies also finance domestic biodiesel subsidies, directly connecting cross-border capital with within-country incentives. However, domestic consumption and exports both increased in the supplied 2025 statistics, so greater domestic use should not automatically be interpreted as reducing exports. [T1:W3] [T1:W20]\n\n- **Spatial tradeoffs:** Importing systems obtain palm products while many forest, water, and land-conversion burdens occur in Indonesian producing landscapes. Within Indonesia, benefits and costs also differ geographically and socially: market-connected communities and remote forest-dependent communities have experienced contrasting welfare outcomes. Aggregate export earnings therefore cannot establish equitable sustainability. [T1:1] [T1:W13]\n\n- **Spillover effects:** International certification can influence clearing both inside certified supply bases and elsewhere in Indonesian landscapes. A review reports positive and negative effects around certified operations in Kalimantan, with an insignificant overall net effect in the summarized analysis. At the national boundary used here, these are within-Indonesia consequences associated with an internationally connected intervention—not proof that reductions inside certified estates equal reductions across the wider landscape. [T1:5]\n\n## 6. Research Gaps and Suggestions\n\n1. **Specify the assessment period and sustainability priorities.** Which years, provinces, and outcomes matter most—household welfare, worker rights, biodiversity, water, climate, or their combination? A national assessment should retain regional and producer-type differences.\n\n2. **Trace production to domestic and export destinations.** Link plantations → mills → processors → traders → destinations, including domestic biodiesel. Separate product categories and identify whether Malaysian purchases are consumed, processed, or re-exported. Official BPS production and trade tables provide a potential baseline, but do not alone establish plantation-to-destination links. [T1:W2]\n\n3. **Strengthen causal attribution and account for delayed conversion.** Distinguish effects of export demand from domestic mandates, prices, infrastructure, and land institutions. Reconstruct land histories: land appearing unused before planting may have been cleared years earlier. [T1:W5]\n\n4. **Measure distribution, not just sector totals.** Collect household income and cost data, wages, occupational risks, tenure histories, compensation, and access to remedies. Disaggregate independent smallholders, scheme participants, workers, and forest-dependent residents. [T1:W13] [T1:W14] [T1:W15]\n\n5. **Validate environmental attribution and candidate spillovers.** Combine forest histories, peat condition, mill emissions, and watershed monitoring with sourcing data. Test Malaysian competitive responses and any transboundary pollution pathways before classifying them as observed effects. Do not attribute Indonesia’s entire fire burden to palm oil; the World Bank explicitly cautions against that attribution. [T1:W8] [T1:W9] [T1:W10]\n\n6. **Evaluate flow-based governance across domestic and foreign markets.** Test coordinated traceability, smallholder support, worker participation, public land information, and financing arrangements—not export standards alone. Include effects beyond participating suppliers and whether domestic-market demand sustains pressures that foreign requirements seek to reduce. [T1:5] [T1:W15] [T1:W18]\n\n## 7. Evidence Coverage\n\n**Strong multi-source support:** The connection between oil-palm expansion and forest, peat, biodiversity, and emissions impacts is supported across the retrieved study and several environmental studies. Evidence also supports heterogeneous social outcomes rather than uniform poverty reduction or uniform harm. [T1:1] [T1:W4] [T1:W6] [T1:W7] [T1:W13]\n\n**Limited or scale-specific evidence:** The quoted 2025 market totals come from an industry association, not an independent sustainability evaluation. Kais results are modeled catchment outcomes, Jambi biodiesel findings concern particular production conditions, and the certification estimates cover historical periods. These findings should not be combined as interchangeable national or current-year measurements. [T1:W3] [T1:W9] [T1:W7] [T1:W16]\n\n**Inferred without direct measurement:** Bilateral financial distributions, environmental changes within importing countries, destination-specific deforestation footprints, and Malaysia’s competitive response remain unquantified. Malaysia is therefore a **candidate**, not a confirmed spillover system. The preparedness exercise establishes shared sourcing and implementation challenges, not displaced production or forest loss. Some retrieved passages are also summarized or truncated, limiting detailed verification. [T1:W24]",
  "research_gaps": [
    "**Specify the assessment period and sustainability priorities.** Which years, provinces, and outcomes matter most—household welfare, worker rights, biodiversity, water, climate, or their combination? A national assessment should retain regional and producer-type differences.",
    "**Trace production to domestic and export destinations.** Link plantations → mills → processors → traders → destinations, including domestic biodiesel. Separate product categories and identify whether Malaysian purchases are consumed, processed, or re-exported. Official BPS production and trade tables provide a potential baseline, but do not alone establish plantation-to-destination links. [T1:W2]",
    "**Strengthen causal attribution and account for delayed conversion.** Distinguish effects of export demand from domestic mandates, prices, infrastructure, and land institutions. Reconstruct land histories: land appearing unused before planting may have been cleared years earlier. [T1:W5]",
    "**Measure distribution, not just sector totals.** Collect household income and cost data, wages, occupational risks, tenure histories, compensation, and access to remedies. Disaggregate independent smallholders, scheme participants, workers, and forest-dependent residents. [T1:W13] [T1:W14] [T1:W15]",
    "**Validate environmental attribution and candidate spillovers.** Combine forest histories, peat condition, mill emissions, and watershed monitoring with sourcing data. Test Malaysian competitive responses and any transboundary pollution pathways before classifying them as observed effects. Do not attribute Indonesia’s entire fire burden to palm oil; the World Bank explicitly cautions against that attribution. [T1:W8] [T1:W9] [T1:W10]",
    "**Evaluate flow-based governance across domestic and foreign markets.** Test coordinated traceability, smallholder support, worker participation, public land information, and financing arrangements—not export standards alone. Include effects beyond participating suppliers and whether domestic-market demand sustains pressures that foreign requirements seek to reduce. [T1:5] [T1:W15] [T1:W18]"
  ],
  "telecoupling": {
    "systems": [
      {
        "role": "sending",
        "system_scope": "distant",
        "name": "Indonesia — palm products and supply-chain information"
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "Indonesia — payments and policy/sourcing information",
        "human_subsystem": "Producers, workers, exporters, processors, public agencies, and producer organizations.",
        "natural_subsystem": "Plantations, forests, peatlands, biodiversity, soils, and water systems.",
        "geographic_scope": "Indonesia nationally."
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "China — palm products"
      },
      {
        "role": "sending",
        "system_scope": "distant",
        "name": "China — payments",
        "human_subsystem": "Importers, processors, food and industrial users, consumers, and regulators.",
        "natural_subsystem": "Domestic agricultural ecosystems, soils, and freshwater associated with vegetable-oil supply.",
        "geographic_scope": "China nationally. China’s role as a destination for processed Indonesian palm oil is also identified in the literature. [T1:2]"
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "Pakistan — palm products"
      },
      {
        "role": "sending",
        "system_scope": "distant",
        "name": "Pakistan — payments",
        "human_subsystem": "Importers, processors, food-sector businesses, consumers, and trade authorities.",
        "natural_subsystem": "Agricultural land, soils, and freshwater associated with domestic food and oil production.",
        "geographic_scope": "Pakistan nationally."
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "Bangladesh — palm products"
      },
      {
        "role": "sending",
        "system_scope": "distant",
        "name": "Bangladesh — payments",
        "human_subsystem": "Importers, food processors, consumers, and relevant authorities.",
        "natural_subsystem": "Agricultural ecosystems and freshwater resources connected to edible-oil supply.",
        "geographic_scope": "Bangladesh nationally."
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "European Union — palm products and traceability information"
      },
      {
        "role": "sending",
        "system_scope": "distant",
        "name": "European Union — payments and regulatory information",
        "human_subsystem": "Importers, manufacturers, consumers, the European Commission, and national competent authorities.",
        "natural_subsystem": "Oil-crop agroecosystems, freshwater resources, and environmental processes affected by food and fuel choices.",
        "geographic_scope": "EU member states and their collective market."
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "United States — palm products"
      },
      {
        "role": "sending",
        "system_scope": "distant",
        "name": "United States — payments and relevant policy information",
        "human_subsystem": "Importers, manufacturers, consumers, and public agencies.",
        "natural_subsystem": "Oil-crop agroecosystems, freshwater resources, and environmental processes associated with alternative food and fuel supplies.",
        "geographic_scope": "United States nationally.",
        "exports_to_these_five_destination_systems_are_reported_in_the_supplied_statistics._[t1": "W3]"
      },
      {
        "role": "spillover",
        "name": "Malaysia as a competing producer — candidate, not established",
        "human_subsystem": "Malaysian plantation companies, smallholders, workers, refiners, and exporters that could experience changes in competition.",
        "natural_subsystem": "Existing plantations and potentially affected forests, peatlands, soils, and watersheds.",
        "geographic_scope": "Malaysian production landscapes.",
        "assessment": "Malaysia’s direct importing role is established above. An additional, indirect effect from changes in Indonesia’s exports to distant buyers is plausible but **not demonstrated**. Evidence that EU operators source from both countries does not itself prove a competitive spillover. [T1:W24]"
      }
    ],
    "flows": [
      {
        "category": "matter",
        "direction": "Indonesia → China, Pakistan, Bangladesh, EU, and United States.",
        "description": "Crude and processed palm-sector products. Destination-specific product composition and sourcing locations must be established rather than inferred from national totals. [T1:W3]"
      },
      {
        "category": "capital",
        "direction": "Distant purchasers → Indonesian exporters.",
        "description": "Export payments, with subsequent domestic distribution through purchases, wages, investment, and public revenue. International financiers and land investors are also identified qualitatively in the literature, but their origins and amounts are not established by the supplied passages. [T1:4]"
      },
      {
        "category": "information",
        "direction": "Foreign policymakers and buyers → Indonesian agencies and suppliers; Indonesian suppliers → importing operators.",
        "description": "Biofuel-policy signals, sourcing requirements, geolocation records, legality information, and due-diligence documentation. Historical EU/US bioenergy demand policies and present deforestation-related requirements should be distinguished rather than treated as one policy mechanism. [T1:4] [T1:W23] [T1:W24]"
      },
      {
        "category": "matter",
        "direction": "Indonesian production → distant consumption, through embodied-resource accounting.",
        "description": "Virtual-water requirements and pollution footprints attributed to traded oil. **Grey water is a pollution-assimilation indicator, not a literal volume of polluted water shipped abroad.** Embodied emissions likewise allocate responsibility without implying physical shipment of CO₂ in products. [T1:1] [T1:4]"
      },
      {
        "category": "information",
        "direction": "Changes in distant-market purchasing → market signals → Malaysian producers.",
        "description": "A proposed pathway for the candidate spillover system. Its strength, direction, and resulting production response remain unverified."
      }
    ],
    "agents": [
      {
        "level": "individuals / households",
        "name": "**Indonesian smallholders and workers; distant consumers**",
        "description": "Influence production, labor, and purchasing decisions."
      },
      {
        "level": "firms / traders / corporations",
        "name": "**Exporters, international traders, importers, manufacturers, financiers, and intermediaries**",
        "description": "Organize trade, investment, sourcing, and processing."
      },
      {
        "level": "governments / policymakers",
        "name": "**Indonesia and importing-market authorities**",
        "description": "Shape trade, energy, environmental, and due-diligence conditions."
      },
      {
        "level": "organizations / NGOs",
        "name": "**RSPO, certification bodies, NGOs, research organizations, and the ILO**",
        "description": "Develop standards, assess outcomes, support implementation, and represent environmental or social concerns."
      }
    ],
    "causes": {
      "economic": [
        "International food, industrial, and biofuel demand links distant consumption to Indonesian production. China’s demand includes processed palm products. [T1:4] [T1:2]"
      ],
      "political / institutional": [
        "Historical foreign renewable-energy policies contributed to feedstock demand. [T1:4]",
        "Deforestation-related import requirements instead condition market access. The EUDR requires covered products to meet deforestation-free and legality conditions and provide required information, including harvest-plot geolocation. These legal requirements are not evidence of realized conservation outcomes. [T1:W23]"
      ],
      "technological / infrastructural": [
        "Processing capacity enables participation in different product markets. [T1:2]",
        "Traceability systems and data transfer influence the feasibility of demonstrating compliance. An EUDR preparedness exercise identified practical geolocation and documentation difficulties; it was not an enforcement determination. [T1:W24]"
      ]
    },
    "effects": {
      "economic": [
        "**Indonesia, sending products:** Gains gross export earnings, while benefits depend on production costs, ownership, wages, taxation, and distribution. [T1:W3]",
        "**Distant receiving systems:** Obtain industrial and food inputs; net consumer and producer welfare effects are not quantified here.",
        "**Malaysia, candidate spillover:** Could gain or lose market share following changes in Indonesian trade. Neither direction nor magnitude is established."
      ],
      "political / institutional": [
        "**Indonesia and EU:** Due-diligence requirements create information and administrative responsibilities across the trade relationship. [T1:W23] [T1:W24]",
        "**Indonesia:** Certification has produced some improvements, but not universal protection. Historical research estimated lower deforestation within certified estates while finding no detectable effect on peatland forest loss or active fires. [T1:W16]"
      ],
      "ecological / biological": [
        "**Indonesia, sending products:** Export demand is connected to forest conversion and fragmentation through production expansion; destination-specific attribution remains necessary. [T1:1]",
        "**Malaysia, candidate spillover:** Any indirect production response could alter pressure on forests and peatlands, but no such response is demonstrated by the supplied evidence."
      ],
      "hydrological": [
        "**Indonesia, sending products:** Production-related water scarcity and pollution can be attributed partly to distant consumption through footprint analysis, without relocating the actual environmental damage to importing countries. [T1:1]"
      ],
      "climatic / atmospheric": [
        "**Indonesia, sending products:** Land conversion, peat processes, and mill methane contribute emissions associated with exported products. [T1:1] [T1:W8]",
        "**Receiving systems:** Climate claims about palm-based fuels depend on production history and full accounting. Jambi findings caution against assuming savings when plantations replace forest. [T1:W7]"
      ],
      "cultural / social / demographic": [
        "**Indonesia:** Economic benefits coexist with unequal welfare outcomes, labor risks, and land-rights concerns. [T1:W13] [T1:W14] [T1:W15]",
        "**Indonesian smallholders:** Corporate commitments do not automatically deliver training, incentives, or meaningful participation. A recent study found particularly limited reach among independent smallholders. [T1:W18]"
      ]
    }
  }
}
```

# 06 — Parsed analysis

`result.parsed` re-parsed into structured fields.

```json
{
  "country_pericoupling_info": {
    "focal_country": "Indonesia (IDN)",
    "mode": "national",
    "pair_results": "Indonesia (IDN) ↔ India (IND) (at sea): PERICOUPLED; Indonesia (IDN) ↔ China (CHN): TELECOUPLED; Indonesia (IDN) ↔ United States (USA): TELECOUPLED; Indonesia (IDN) ↔ Pakistan (PAK): TELECOUPLED; Indonesia (IDN) ↔ Switzerland (CHE): TELECOUPLED; Indonesia (IDN) ↔ European Union: TELECOUPLED",
    "core_subnational_regions": "Kalimantan Barat",
    "note": "LLM classification is consistent with the coupling database."
  },
  "coupling_classification": "Indonesia’s palm oil sector is best analyzed as a **metacoupled system**: production, domestic consumption, neighboring-country trade, and distant exports jointly influence sustainability. Following Liu et al. (2013) and Liu (2017), the analysis below separates these spatial interactions before examining their connections.\n\n**Scope:** Indonesia is the national focal system. Consequently, interactions between Indonesian provinces are treated as intracoupling here; a province-level study would classify some of those same interactions differently. Findings are dated where possible, and unverified mechanisms are identified as hypotheses.\n\n- **Intracoupling:** Interactions among Indonesian plantations, mills, workers, households, governments, forests, peatlands, and water resources. These include domestic food and biodiesel demand—not only production for export.\n- **Pericoupling:** Palm oil exports to **India**, an actual destination identified in the supplied evidence. Under the supplied country-pair lookup, Indonesia and India are maritime-adjacent, so their interaction is classified as pericoupling. [T1:W2] Although one retrieved summary groups India with distant importers, that classification is not adopted under this explicit adjacency convention. [T1:4]\n- **Telecoupling:** Trade with distant markets, including China, the European Union, the United States, Pakistan, and Switzerland, together with associated payments and policy information. Evidence strength and observation periods differ across these markets. [T1:2] [T1:4] [T1:W21]\n\nThe principal category from Liu et al.’s (2013) fourteen telecoupling categories is **trade**. Investment and technology transfer may also matter, but their specific origins, destinations, and magnitudes require further evidence.",
  "cross_coupling_interactions": [
    "**Amplification:** Distant demand can strengthen incentives for plantation investment, processing, and land conversion within Indonesia. Global price associations with expansion support this connection, but do not establish that exports caused every observed instance of clearing. [T1:1] [T1:W8]",
    "**Offset:** At a given production level, increased Indonesian biodiesel consumption can reduce exportable supply. Bank Indonesia’s higher-blend scenarios project foregone export earnings, while the retrieved synthesis also describes situations in which domestic use and exports grow together. The sustainability outcome therefore depends on production responses, not merely the domestic–export allocation. [T1:W24] [T1:8]",
    "**Spatial tradeoffs:** Distant consumers obtain useful products while Indonesian producing landscapes and communities bear much of the associated land, water, labor, and emissions burden. Benefits and costs also differ within Indonesia: higher revenues for some actors can coexist with losses for forest-dependent communities or workers. [T1:1] [T1:W4] [T1:W7]",
    "**Displacement:** Restrictions or purchasing changes in one distant market may redirect sales toward other markets rather than reduce Indonesian production. The supplied synthesis reports such redirection, but does not quantify bilateral changes. Whether this affects adjacent India, distant buyers, or competing Malaysian suppliers requires tracing actual orders and production responses. [T1:8]"
  ],
  "evidence_coverage_note": "**Strongest support within the supplied material:** Plantation expansion, forest conversion, and environmental pressures are supported by the Rulli summary and independent satellite-based research. [T1:1] [T1:W8] Heterogeneous livelihood outcomes are supported by village-level research and detailed partnership investigations, although these sources use different methods and populations. [T1:W4] [T1:W5]\n\n**Direct but uneven trade evidence:** India is identified in the supplied Trase analysis, China in the retrieved literature, and Switzerland in OECD reporting. [T1:W2] [T1:2] [T1:W21] The US and Pakistan roles rely principally on one older summary. [T1:4] The broad regional description does not establish named country-to-country investment pathways. [T1:7]\n\n**Inferred or unverified:** Detailed payment distribution, bilateral information networks, importing-country land-use effects, and Malaysia’s proposed spillover response are analytical mappings or hypotheses—not measured findings. Malaysia’s importance as a producer does not independently prove an effect from Indonesian exports. [T1:W1]\n\n**Interpretive limitations:** The supplied evidence consists of passages and web-result summaries rather than independently checked full texts. Historical case investigations, footprint models, scenarios, and causal evaluations should not be treated as interchangeable. In particular, Trase’s historical deforestation exposure is not annual clearing, and modeled water pressures do not negate site-specific measurements showing low nutrient impacts under particular conditions. [T1:W2] [T1:W14]",
  "intracoupling": {
    "systems": [
      {
        "role": "focal",
        "name": "Indonesia",
        "human_subsystem": "Independent and contract smallholders, plantation workers, estate companies, mills, refiners, domestic consumers, biodiesel producers, financial institutions, and national and subnational governments.",
        "natural_subsystem": "Oil palm cropland, remaining tropical forests, peatlands, mineral soils, rivers, groundwater, and associated biodiversity.",
        "geographic_scope": "Indonesia nationally, with production concentrated historically in Sumatra and Kalimantan. Jambi and West Kalimantan provide important examples of smallholder and community-level interactions. [T1:2] [T1:3] [T1:W6]"
      }
    ],
    "flows": [
      {
        "category": "matter",
        "direction": "Indonesian plantations → domestic mills → refineries, food manufacturers, biodiesel plants, and export terminals.",
        "description": "Fresh fruit bunches, crude palm oil, and processed products. Domestic consumption is substantial: the supplied Trase analysis reports that it absorbed approximately 53% of production in 2024. [T1:W2]"
      },
      {
        "category": "matter",
        "direction": "Water sources and fertilizer applications → plantations → crops, soils, groundwater, and streams.",
        "description": "Water and nutrient movements connect plantation management with surrounding ecosystems. Their environmental consequences depend on soils, fertilizer practices, and hydrological conditions. [T1:1] [T1:W14]"
      },
      {
        "category": "capital",
        "direction": "Domestic buyers, companies, lenders, and public programs → producers and workers; participating households → lenders or company-managed partnership accounts.",
        "description": "Crop payments, wages, credit, subsidies, investment, and debt repayments. Investigated plasma partnerships demonstrate that participation can involve substantial debt and limited household control over financial calculations. These cases are not representative estimates for all smallholders. [T1:W5]"
      },
      {
        "category": "energy",
        "direction": "Indonesian biodiesel producers → domestic transport and other fuel users.",
        "description": "Energy supplied through palm-based biodiesel. This is the energy content of a material product and should not be added to commodity tonnage as an independent physical quantity. [T1:W24]"
      },
      {
        "category": "people",
        "direction": "Indonesian residences and communities → plantations and mills.",
        "description": "Workers travel or relocate to production sites. Employment is documented, but the supplied evidence does not quantify national origin–destination labor movements. [T1:W7]"
      },
      {
        "category": "information",
        "direction": "Indonesian regulators, companies, producer groups, and researchers ↔ farms and mills.",
        "description": "Production advice, certification requirements, land documentation, monitoring information, and purchasing conditions. Detailed network volumes remain unmeasured. [T1:W21] [T1:W23]"
      }
    ],
    "agents": [
      {
        "level": "individuals / households",
        "name": "**Independent and contract smallholders**",
        "description": "Choose production practices, negotiate sales, manage household finances, and decide whether to replant. Their market access and contractual autonomy differ substantially. [T1:W23]"
      },
      {
        "level": "individuals / households",
        "name": "**Workers and affected communities**",
        "description": "Supply labor, negotiate land access, contest plantation expansion, and experience livelihood and environmental changes."
      },
      {
        "level": "firms / traders / corporations",
        "name": "**Plantation companies, mills, refiners, traders, and lenders**",
        "description": "Organize production, processing, purchasing, finance, and contractual benefit-sharing."
      },
      {
        "level": "governments / policymakers",
        "name": "**National, provincial, and district authorities**",
        "description": "Shape licensing, land allocation, labor enforcement, environmental protection, certification, and biodiesel policy."
      },
      {
        "level": "organizations / NGOs",
        "name": "**Producer organizations, research institutions, unions, and civil society organizations**",
        "description": "Support producer participation, generate evidence, monitor conditions, and represent affected groups."
      }
    ],
    "causes": {
      "economic": [
        "Expected income from domestic and export sales encourages cultivation, processing, and investment. Historical palm oil prices were associated with industrial plantation expansion and forest loss. [T1:W8]",
        "Limited liquidity and income losses during replanting can discourage renewal of aging plantations. [T1:W17]"
      ],
      "political / institutional": [
        "Land concessions, tenure arrangements, enforcement, certification, and biodiesel policy shape which activities are profitable and legally possible.",
        "Restrictions on new licenses do not necessarily prevent conversion under existing concessions. [T1:W21]"
      ],
      "technological / infrastructural": [
        "Milling and refining capacity, agronomic knowledge, planting material, and plantation management influence productivity.",
        "Research identifies substantial yield gaps and opportunities to improve ecological performance within existing plantations. [T1:W15] [T1:W16]"
      ],
      "cultural / social / demographic": [
        "Initial livelihood systems, market integration, bargaining power, and dependence on forests influence participation and outcomes. These help explain why plantation development benefits some villages but disadvantages others. [T1:W4]"
      ],
      "hydrological": [
        "Plantation water requirements, peat drainage, fertilizer transport, and seasonal water availability influence production and environmental pressures. [T1:1] [T1:W6]"
      ]
    },
    "effects": {
      "economic": [
        "**Indonesia:** Plantation development can increase income and economic activity, but benefits are uneven. Market-oriented Kalimantan villages experienced better socioeconomic outcomes than comparable villages, whereas remote, heavily forested subsistence-oriented villages experienced worse outcomes. [T1:W4]",
        "**Indonesia:** Some investigated plasma partnerships generated low household receipts and persistent debt, demonstrating that formal participation does not guarantee equitable benefits. [T1:W5]"
      ],
      "ecological / biological": [
        "**Indonesia:** Forest conversion and fragmentation reduce habitat and biodiversity. A national satellite study estimated that approximately 3.09 million hectares of primary forest lost during 2001–2019 had become oil palm by 2019; this does not mean all Indonesian deforestation was caused by palm oil. [T1:W8]",
        "**Indonesia:** Reduced management intensity and native-tree enrichment improved ecological outcomes without detected yield penalties in the studied Sumatran systems. These results do not make plantations equivalent to forests. [T1:W15]"
      ],
      "cultural / social / demographic": [
        "**Indonesia:** Historical investigations documented labor abuses, insecure employment, inadequate consultation, and livelihood disruption in particular plantation settings. These are important findings, but not current sector-wide prevalence estimates. [T1:W6] [T1:W7]"
      ],
      "hydrological": [
        "**Indonesia:** Water scarcity and nutrient pollution pressures are identified in footprint modeling. However, measured mature plantations in one Sumatran landscape did not cause nearby eutrophication under sampled conditions. Water effects therefore require site-specific assessment rather than a uniform pollution assumption. [T1:1] [T1:W14]"
      ],
      "climatic / atmospheric": [
        "**Indonesia:** Forest conversion and peat drainage generate emissions. Biodiesel’s climate performance depends strongly on prior land use: an Indonesian assessment found no greenhouse gas savings for a first rotation following forest conversion on mineral soils, while modeled subsequent rotations performed substantially better. [T1:W12]"
      ],
      "geological / geomorphological": [
        "**Indonesia:** Peat drainage can cause subsidence, with consequences for land condition and livelihoods, as documented in the West Kalimantan investigation. [T1:W6]"
      ]
    }
  },
  "is_parsed": true,
  "map_data": {
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
  },
  "pericoupling": {
    "systems": [
      {
        "role": "sending",
        "system_scope": "adjacent",
        "name": "Indonesia — palm oil"
      },
      {
        "role": "receiving",
        "system_scope": "adjacent",
        "name": "Indonesia — buyer payments",
        "human_subsystem": "Producers, mills, refiners, exporters, workers, and trade authorities serving Indian demand.",
        "natural_subsystem": "Supplying plantations, forests, peatlands, soils, and watersheds.",
        "geographic_scope": "Indonesia, particularly the production areas supplying India; these origins require shipment-to-plantation tracing."
      },
      {
        "role": "receiving",
        "system_scope": "adjacent",
        "name": "India — palm oil"
      },
      {
        "role": "sending",
        "system_scope": "adjacent",
        "name": "India — buyer payments",
        "human_subsystem": "Importers, refiners, food and industrial manufacturers, consumers, and trade authorities.",
        "natural_subsystem": "Domestic oilseed cropland, freshwater resources, and processing environments relevant to vegetable-oil provision. Changes in these resources caused by Indonesian imports are not established in the supplied evidence.",
        "geographic_scope": "India nationally, using the supplied maritime-adjacency convention.",
        "description": "The commodity relationship is documented; reverse payment flows are economically expected but not quantified in the supplied material."
      }
    ],
    "flows": [
      {
        "category": "matter",
        "direction": "Indonesia → India.",
        "description": "Palm oil and related products. India is identified as Indonesia’s largest export destination in the supplied Trase analysis, but no bilateral tonnage is provided here. [T1:W2]"
      },
      {
        "category": "capital",
        "direction": "Indian buyers → Indonesian exporters and suppliers.",
        "description": "Payments for traded products. Their distribution among traders, companies, workers, and smallholders requires financial tracing rather than assuming export receipts reach producers proportionately."
      },
      {
        "category": "information",
        "direction": "Indian buyers ↔ Indonesian exporters and processors.",
        "description": "Orders, prices, product specifications, and purchasing decisions are expected trade-related information flows. Their detailed pathways and influence have not been directly documented in the supplied evidence."
      },
      {
        "category": "matter",
        "direction": "Indonesian production → Indian consumption, through embodied-water accounting.",
        "description": "Water use can be attributed to the palm oil consumed in India. This is an accounting relationship—not evidence that irrigation water or polluted water physically travels to India. Grey water footprints represent calculated pollution-related dilution requirements. [T1:1] [T1:4]"
      }
    ],
    "agents": [
      {
        "level": "individuals / households",
        "name": "**Indonesian producers and workers; Indian consumers**",
        "description": "Participate in production, employment, and consumption."
      },
      {
        "level": "firms / traders / corporations",
        "name": "**Indonesian exporters and Indian importers, refiners, and manufacturers**",
        "description": "Determine procurement, processing, contractual conditions, and market access."
      },
      {
        "level": "governments / policymakers",
        "name": "**Indonesian and Indian authorities**",
        "description": "Govern the trade relationship. Specific policy effects should be established using dated bilateral evidence, not inferred merely from their regulatory roles."
      }
    ],
    "causes": {
      "economic": [
        "Indian demand and Indonesian suppliers’ expected returns sustain the commodity relationship. Rising dietary and industrial demand is identified among the broader drivers of Indonesian palm oil trade. [T1:4]"
      ],
      "cultural / social / demographic": [
        "Food consumption patterns and population-related demand help sustain vegetable-oil consumption. The supplied evidence does not quantify how individual dietary changes affect Indonesian land use. [T1:4]"
      ],
      "technological / infrastructural": [
        "Plantation processing, export terminals, shipping, and importing-country refining enable the trade. Their importance is clear analytically, but route-specific capacity and cost data remain missing."
      ]
    },
    "effects": {
      "economic": [
        "**Indonesia:** Indian purchases generate sales revenue and connect producers to external demand; the net income and distributional effects remain unquantified.",
        "**India:** Imports provide food and industrial inputs. Effects on consumer prices, domestic producers, and import dependence require a market counterfactual."
      ],
      "ecological / biological": [
        "**Indonesia:** Part of the production-related forest and biodiversity footprint can be attributed to Indian consumption. This allocation is not equivalent to proving how much additional clearing Indian demand caused. [T1:1] [T1:4]",
        "**India:** Avoided domestic land conversion should not be assumed. It depends on which oils, production locations, and consumption levels would otherwise prevail."
      ],
      "climatic / atmospheric": [
        "**Indonesia and India:** Emissions associated with producing India-bound oil connect Indonesian land management with Indian consumption. Consumption-based attribution does not change where emissions physically occur, and a bilateral emissions share is not supplied. [T1:4]"
      ]
    }
  },
  "pericoupling_info": null,
  "raw_text": "## 1. Coupling Classification\n\nIndonesia’s palm oil sector is best analyzed as a **metacoupled system**: production, domestic consumption, neighboring-country trade, and distant exports jointly influence sustainability. Following Liu et al. (2013) and Liu (2017), the analysis below separates these spatial interactions before examining their connections.\n\n**Scope:** Indonesia is the national focal system. Consequently, interactions between Indonesian provinces are treated as intracoupling here; a province-level study would classify some of those same interactions differently. Findings are dated where possible, and unverified mechanisms are identified as hypotheses.\n\n- **Intracoupling:** Interactions among Indonesian plantations, mills, workers, households, governments, forests, peatlands, and water resources. These include domestic food and biodiesel demand—not only production for export.\n- **Pericoupling:** Palm oil exports to **India**, an actual destination identified in the supplied evidence. Under the supplied country-pair lookup, Indonesia and India are maritime-adjacent, so their interaction is classified as pericoupling. [T1:W2] Although one retrieved summary groups India with distant importers, that classification is not adopted under this explicit adjacency convention. [T1:4]\n- **Telecoupling:** Trade with distant markets, including China, the European Union, the United States, Pakistan, and Switzerland, together with associated payments and policy information. Evidence strength and observation periods differ across these markets. [T1:2] [T1:4] [T1:W21]\n\nThe principal category from Liu et al.’s (2013) fourteen telecoupling categories is **trade**. Investment and technology transfer may also matter, but their specific origins, destinations, and magnitudes require further evidence.\n\n## 2. Intracoupling Analysis — within Indonesia\n\n### 2.1 Systems Identification\n\n**Focal System**: Indonesia\n- **Human subsystem:** Independent and contract smallholders, plantation workers, estate companies, mills, refiners, domestic consumers, biodiesel producers, financial institutions, and national and subnational governments.\n- **Natural subsystem:** Oil palm cropland, remaining tropical forests, peatlands, mineral soils, rivers, groundwater, and associated biodiversity.\n- **Geographic scope:** Indonesia nationally, with production concentrated historically in Sumatra and Kalimantan. Jambi and West Kalimantan provide important examples of smallholder and community-level interactions. [T1:2] [T1:3] [T1:W6]\n\n### 2.2 Flows Analysis\n\n**Matter Flow**\n- **Direction:** Indonesian plantations → domestic mills → refineries, food manufacturers, biodiesel plants, and export terminals.\n- **Description:** Fresh fruit bunches, crude palm oil, and processed products. Domestic consumption is substantial: the supplied Trase analysis reports that it absorbed approximately 53% of production in 2024. [T1:W2]\n\n**Matter Flow**\n- **Direction:** Water sources and fertilizer applications → plantations → crops, soils, groundwater, and streams.\n- **Description:** Water and nutrient movements connect plantation management with surrounding ecosystems. Their environmental consequences depend on soils, fertilizer practices, and hydrological conditions. [T1:1] [T1:W14]\n\n**Capital Flow**\n- **Direction:** Domestic buyers, companies, lenders, and public programs → producers and workers; participating households → lenders or company-managed partnership accounts.\n- **Description:** Crop payments, wages, credit, subsidies, investment, and debt repayments. Investigated plasma partnerships demonstrate that participation can involve substantial debt and limited household control over financial calculations. These cases are not representative estimates for all smallholders. [T1:W5]\n\n**Energy Flow**\n- **Direction:** Indonesian biodiesel producers → domestic transport and other fuel users.\n- **Description:** Energy supplied through palm-based biodiesel. This is the energy content of a material product and should not be added to commodity tonnage as an independent physical quantity. [T1:W24]\n\n**People Flow**\n- **Direction:** Indonesian residences and communities → plantations and mills.\n- **Description:** Workers travel or relocate to production sites. Employment is documented, but the supplied evidence does not quantify national origin–destination labor movements. [T1:W7]\n\n**Information Flow**\n- **Direction:** Indonesian regulators, companies, producer groups, and researchers ↔ farms and mills.\n- **Description:** Production advice, certification requirements, land documentation, monitoring information, and purchasing conditions. Detailed network volumes remain unmeasured. [T1:W21] [T1:W23]\n\n### 2.3 Agents\n\n- [Individuals / households] **Independent and contract smallholders:** Choose production practices, negotiate sales, manage household finances, and decide whether to replant. Their market access and contractual autonomy differ substantially. [T1:W23]\n- [Individuals / households] **Workers and affected communities:** Supply labor, negotiate land access, contest plantation expansion, and experience livelihood and environmental changes.\n- [Firms / traders / corporations] **Plantation companies, mills, refiners, traders, and lenders:** Organize production, processing, purchasing, finance, and contractual benefit-sharing.\n- [Governments / policymakers] **National, provincial, and district authorities:** Shape licensing, land allocation, labor enforcement, environmental protection, certification, and biodiesel policy.\n- [Organizations / NGOs] **Producer organizations, research institutions, unions, and civil society organizations:** Support producer participation, generate evidence, monitor conditions, and represent affected groups.\n\nOil palms are central biological components, but their presence alone does not require treating them as autonomous agents. They can instead be modeled through ecological processes such as growth and nutrient uptake.\n\n### 2.4 Causes\n\n**Economic**\n- Expected income from domestic and export sales encourages cultivation, processing, and investment. Historical palm oil prices were associated with industrial plantation expansion and forest loss. [T1:W8]\n- Limited liquidity and income losses during replanting can discourage renewal of aging plantations. [T1:W17]\n\n**Political / Institutional**\n- Land concessions, tenure arrangements, enforcement, certification, and biodiesel policy shape which activities are profitable and legally possible.\n- Restrictions on new licenses do not necessarily prevent conversion under existing concessions. [T1:W21]\n\n**Technological / Infrastructural**\n- Milling and refining capacity, agronomic knowledge, planting material, and plantation management influence productivity.\n- Research identifies substantial yield gaps and opportunities to improve ecological performance within existing plantations. [T1:W15] [T1:W16]\n\n**Cultural / Social / Demographic**\n- Initial livelihood systems, market integration, bargaining power, and dependence on forests influence participation and outcomes. These help explain why plantation development benefits some villages but disadvantages others. [T1:W4]\n\n**Hydrological**\n- Plantation water requirements, peat drainage, fertilizer transport, and seasonal water availability influence production and environmental pressures. [T1:1] [T1:W6]\n\n### 2.5 Effects\n\n**Economic**\n- **Indonesia:** Plantation development can increase income and economic activity, but benefits are uneven. Market-oriented Kalimantan villages experienced better socioeconomic outcomes than comparable villages, whereas remote, heavily forested subsistence-oriented villages experienced worse outcomes. [T1:W4]\n- **Indonesia:** Some investigated plasma partnerships generated low household receipts and persistent debt, demonstrating that formal participation does not guarantee equitable benefits. [T1:W5]\n\n**Ecological / Biological**\n- **Indonesia:** Forest conversion and fragmentation reduce habitat and biodiversity. A national satellite study estimated that approximately 3.09 million hectares of primary forest lost during 2001–2019 had become oil palm by 2019; this does not mean all Indonesian deforestation was caused by palm oil. [T1:W8]\n- **Indonesia:** Reduced management intensity and native-tree enrichment improved ecological outcomes without detected yield penalties in the studied Sumatran systems. These results do not make plantations equivalent to forests. [T1:W15]\n\n**Cultural / Social / Demographic**\n- **Indonesia:** Historical investigations documented labor abuses, insecure employment, inadequate consultation, and livelihood disruption in particular plantation settings. These are important findings, but not current sector-wide prevalence estimates. [T1:W6] [T1:W7]\n\n**Hydrological**\n- **Indonesia:** Water scarcity and nutrient pollution pressures are identified in footprint modeling. However, measured mature plantations in one Sumatran landscape did not cause nearby eutrophication under sampled conditions. Water effects therefore require site-specific assessment rather than a uniform pollution assumption. [T1:1] [T1:W14]\n\n**Climatic / Atmospheric**\n- **Indonesia:** Forest conversion and peat drainage generate emissions. Biodiesel’s climate performance depends strongly on prior land use: an Indonesian assessment found no greenhouse gas savings for a first rotation following forest conversion on mineral soils, while modeled subsequent rotations performed substantially better. [T1:W12]\n\n**Geological / Geomorphological**\n- **Indonesia:** Peat drainage can cause subsidence, with consequences for land condition and livelihoods, as documented in the West Kalimantan investigation. [T1:W6]\n\n## 3. Pericoupling Analysis — Indonesia and adjacent India\n\n### 3.1 Systems Identification\n\n**Sending System (adjacent)**: Indonesia — palm oil\n**Receiving System (adjacent)**: Indonesia — buyer payments\n- **Human subsystem:** Producers, mills, refiners, exporters, workers, and trade authorities serving Indian demand.\n- **Natural subsystem:** Supplying plantations, forests, peatlands, soils, and watersheds.\n- **Geographic scope:** Indonesia, particularly the production areas supplying India; these origins require shipment-to-plantation tracing.\n\n**Receiving System (adjacent)**: India — palm oil\n**Sending System (adjacent)**: India — buyer payments\n- **Human subsystem:** Importers, refiners, food and industrial manufacturers, consumers, and trade authorities.\n- **Natural subsystem:** Domestic oilseed cropland, freshwater resources, and processing environments relevant to vegetable-oil provision. Changes in these resources caused by Indonesian imports are not established in the supplied evidence.\n- **Geographic scope:** India nationally, using the supplied maritime-adjacency convention.\n\nThe commodity relationship is documented; reverse payment flows are economically expected but not quantified in the supplied material.\n\n### 3.2 Flows Analysis\n\n**Matter Flow**\n- **Direction:** Indonesia → India.\n- **Description:** Palm oil and related products. India is identified as Indonesia’s largest export destination in the supplied Trase analysis, but no bilateral tonnage is provided here. [T1:W2]\n\n**Capital Flow**\n- **Direction:** Indian buyers → Indonesian exporters and suppliers.\n- **Description:** Payments for traded products. Their distribution among traders, companies, workers, and smallholders requires financial tracing rather than assuming export receipts reach producers proportionately.\n\n**Information Flow**\n- **Direction:** Indian buyers ↔ Indonesian exporters and processors.\n- **Description:** Orders, prices, product specifications, and purchasing decisions are expected trade-related information flows. Their detailed pathways and influence have not been directly documented in the supplied evidence.\n\n**Matter Flow**\n- **Direction:** Indonesian production → Indian consumption, through embodied-water accounting.\n- **Description:** Water use can be attributed to the palm oil consumed in India. This is an accounting relationship—not evidence that irrigation water or polluted water physically travels to India. Grey water footprints represent calculated pollution-related dilution requirements. [T1:1] [T1:4]\n\n### 3.3 Agents\n\n- [Individuals / households] **Indonesian producers and workers; Indian consumers:** Participate in production, employment, and consumption.\n- [Firms / traders / corporations] **Indonesian exporters and Indian importers, refiners, and manufacturers:** Determine procurement, processing, contractual conditions, and market access.\n- [Governments / policymakers] **Indonesian and Indian authorities:** Govern the trade relationship. Specific policy effects should be established using dated bilateral evidence, not inferred merely from their regulatory roles.\n\n### 3.4 Causes\n\n**Economic**\n- Indian demand and Indonesian suppliers’ expected returns sustain the commodity relationship. Rising dietary and industrial demand is identified among the broader drivers of Indonesian palm oil trade. [T1:4]\n\n**Cultural / Social / Demographic**\n- Food consumption patterns and population-related demand help sustain vegetable-oil consumption. The supplied evidence does not quantify how individual dietary changes affect Indonesian land use. [T1:4]\n\n**Technological / Infrastructural**\n- Plantation processing, export terminals, shipping, and importing-country refining enable the trade. Their importance is clear analytically, but route-specific capacity and cost data remain missing.\n\n### 3.5 Effects\n\n**Economic**\n- **Indonesia:** Indian purchases generate sales revenue and connect producers to external demand; the net income and distributional effects remain unquantified.\n- **India:** Imports provide food and industrial inputs. Effects on consumer prices, domestic producers, and import dependence require a market counterfactual.\n\n**Ecological / Biological**\n- **Indonesia:** Part of the production-related forest and biodiversity footprint can be attributed to Indian consumption. This allocation is not equivalent to proving how much additional clearing Indian demand caused. [T1:1] [T1:4]\n- **India:** Avoided domestic land conversion should not be assumed. It depends on which oils, production locations, and consumption levels would otherwise prevail.\n\n**Climatic / Atmospheric**\n- **Indonesia and India:** Emissions associated with producing India-bound oil connect Indonesian land management with Indian consumption. Consumption-based attribution does not change where emissions physically occur, and a bilateral emissions share is not supplied. [T1:4]\n\n## 4. Telecoupling Analysis — Indonesia and distant markets\n\n### 4.1 Systems Identification\n\nThe following importer natural subsystems identify what a complete CHANS analysis should examine. They are **not claims of measured ecological change in those countries**.\n\n**Sending System (distant)**: Indonesia — palm oil\n**Receiving System (distant)**: Indonesia — payments and policy information\n- **Human subsystem:** Producers, workers, processors, exporters, financial institutions, and regulators.\n- **Natural subsystem:** Supplying plantations, tropical forests, peatlands, soils, and watersheds.\n- **Geographic scope:** Indonesia nationally, with origins disaggregated by producing landscape wherever possible.\n\n**Receiving System (distant)**: China — palm oil\n**Sending System (distant)**: China — buyer payments and orders\n- **Human subsystem:** Importers, processors, manufacturers, consumers, and trade authorities.\n- **Natural subsystem:** Oilseed cropland, freshwater resources, and environments affected by processing and alternative oil provision.\n- **Geographic scope:** China. The retrieved literature identifies China as an important destination for processed Indonesian palm oil. [T1:2]\n\n**Receiving System (distant)**: European Union — palm oil\n**Sending System (distant)**: European Union — payments and regulatory information\n- **Human subsystem:** Importers, manufacturers, consumers, EU institutions, and member-state authorities.\n- **Natural subsystem:** Agricultural landscapes and ecosystems involved in alternative vegetable-oil production and processing.\n- **Geographic scope:** EU member states; regulatory authority also operates at bloc level. [T1:4] [T1:W22]\n\n**Receiving System (distant)**: United States — palm oil\n**Sending System (distant)**: United States — payments and policy signals\n- **Human subsystem:** Importers, food and industrial manufacturers, consumers, and policymakers.\n- **Natural subsystem:** Oilseed-producing landscapes, soils, water resources, and processing environments.\n- **Geographic scope:** United States. Its receiving role and bioenergy-policy influence are identified in the supplied 2019 summary, not established here with current bilateral quantities. [T1:4]\n\n**Receiving System (distant)**: Pakistan — palm oil\n**Sending System (distant)**: Pakistan — buyer payments\n- **Human subsystem:** Importers, edible-oil processors, consumers, and trade institutions.\n- **Natural subsystem:** Domestic oilseed cropland and water resources relevant to alternative oil supply.\n- **Geographic scope:** Pakistan; identified as a destination in the supplied 2019 summary. [T1:4]\n\n**Receiving System (distant)**: Switzerland — palm oil\n**Sending System (distant)**: Switzerland — payments and sustainability-related purchasing requirements\n- **Human subsystem:** Importers, manufacturers, consumers, and authorities administering preferential trade conditions.\n- **Natural subsystem:** Agricultural land, freshwater resources, and environments associated with alternative food-oil supply.\n- **Geographic scope:** Switzerland. OECD reports 21 tonnes of qualifying Indonesian palm oil imports under sustainability-linked preferences in 2022. [T1:W21]\n\n**Spillover System**: Malaysia — **candidate, not a demonstrated spillover**\n- **Human subsystem:** Malaysian plantation companies, smallholders, refiners, exporters, workers, and relevant authorities.\n- **Natural subsystem:** Oil palm landscapes and remaining forest and peat habitats that could experience altered production pressure.\n- **Geographic scope:** Malaysia, including producing landscapes in Peninsular Malaysia and Malaysian Borneo.\n- **Potential mechanism:** Changes in buyers’ sourcing from Indonesia could affect Malaysian orders, prices, investment, and land-use incentives. Malaysia’s importance as a co-producer supports investigating this possibility, but does not establish that such effects occurred. [T1:W1]\n\nMalaysia’s spillover role concerns its **indirect relationship to an Indonesia–distant-market interaction**. It is not classified as a direct Indonesian trading or investment partner solely because the countries are adjacent.\n\n### 4.2 Flows Analysis\n\n**Matter Flow**\n- **Direction:** Indonesia → China, EU member states, United States, Pakistan, and Switzerland.\n- **Description:** Crude or processed palm oil and feedstocks, with product composition differing by market. China’s processed-oil relationship is specifically identified; other historical destinations appear in the Rulli summary. [T1:2] [T1:4]\n- **Quantity qualification:** Switzerland’s 21 tonnes concern qualifying preferential imports in 2022—not its entire palm oil market. No such preferential imports were recorded in 2023–2024; that does not demonstrate that all Indonesian–Swiss palm oil trade ceased. [T1:W21]\n\n**Capital Flow**\n- **Direction:** Distant buyers → Indonesian exporters → domestic suppliers and investment.\n- **Description:** Export payments and subsequent domestic allocation. The international payment belongs here; its internal reinvestment and distribution also belong in intracoupling. Neither the detailed payment network nor benefit shares are supplied.\n\n**Information Flow**\n- **Direction:** EU and US policymakers → Indonesian authorities and firms; Indonesian suppliers → distant buyers.\n- **Description:** Bioenergy-policy signals, sustainability requirements, production-location information, and due-diligence documentation. EU legal requirements establish obligations, not evidence that compliance has already reduced Indonesian deforestation. [T1:4] [T1:W22]\n\n**Matter Flow**\n- **Direction:** Indonesian production → distant consumption, through embodied-resource accounting.\n- **Description:** Water use and pollution-related footprints are allocated to traded products. Associated emissions should also be accounted for, but “embodied emissions” are not a physical shipment of the gases emitted during production. [T1:1] [T1:4]\n\n**Information Flow**\n- **Direction:** Shared importing markets → Malaysian firms and producers.\n- **Description:** Possible changes in prices, orders, and sourcing expectations constitute a **candidate** connection to Malaysia. The supplied evidence does not establish its direction, magnitude, or land-use consequences.\n\n### 4.3 Agents\n\n- [Individuals / households] **Indonesian producers, workers, and affected communities; distant consumers:** Make production, labor, livelihood, and purchasing decisions.\n- [Firms / traders / corporations] **Exporters, multinational traders, refiners, manufacturers, and importing firms:** Connect production locations with distant demand and distribute purchasing requirements.\n- [Governments / policymakers] **Indonesian authorities and distant-market policymakers:** Influence production incentives, bioenergy demand, market access, and legal obligations.\n- [Organizations / NGOs] **Certification organizations, researchers, labor organizations, and environmental NGOs:** Set or assess standards, investigate impacts, and support accountability.\n- [Firms / traders / corporations] **Malaysian producers and processors:** Potentially respond to altered sourcing, but this response remains to be demonstrated.\n\n### 4.4 Causes\n\n**Economic**\n- Global food, industrial, and biofuel demand creates incentives for Indonesian production and exports. International prices influence expected plantation returns. [T1:4] [T1:W8]\n\n**Political / Institutional**\n- Foreign renewable-energy policies can increase feedstock demand, while sustainability requirements alter market-access conditions. [T1:4]\n- EU rules require covered products to be deforestation-free, legally produced, and supported by due diligence, including production-location information. [T1:W22]\n\n**Cultural / Social / Demographic**\n- Dietary patterns, population-related consumption, and demand for manufactured products contribute to distant demand. [T1:4]\n\n**Technological / Infrastructural**\n- Refining capacity, processing capabilities, and logistics enable Indonesia to supply differentiated products. The retrieved literature documents a major historical increase in Indonesian processing capacity. [T1:2]\n\n### 4.5 Effects\n\n**Economic**\n- **Indonesia:** Distant markets provide sales opportunities but also expose suppliers to changing prices and purchasing conditions. Export earnings are not equivalent to net social benefits.\n- **Receiving systems:** Buyers obtain food, manufacturing, and energy inputs; compliance can introduce additional costs.\n- **Malaysia, candidate spillover:** Procurement changes could increase or decrease business opportunities. No net effect is established.\n\n**Political / Institutional**\n- **EU and Indonesian suppliers:** Due-diligence requirements create traceability and documentation obligations. [T1:W22]\n- **Switzerland and Indonesian suppliers:** Sustainability-linked preferences illustrate conditional market access, but observed uptake was very small during the reported period. [T1:W21]\n\n**Ecological / Biological**\n- **Indonesia:** Export-linked demand is associated with land-conversion and habitat pressures, although the additional effect of exports must be separated from domestic demand and other land-use drivers. [T1:1]\n- **Indonesia:** An evaluation of RSPO certification during 2001–2015 estimated reduced deforestation within certified plantations, but detected no causal effect on peatland forest loss or active fires. [T1:W18]\n- **Malaysia, candidate spillover:** Any change in forest pressure depends on whether altered demand affects yields, existing plantation output, or new expansion. None is demonstrated here.\n\n**Cultural / Social / Demographic**\n- **Indonesia:** International supply chains connect distant buyers with documented labor and livelihood harms in particular producing locations. [T1:W6] [T1:W7]\n- **Indonesia:** Certified-market participation is uneven. Independent smallholders were underrepresented, but the relevant study found no evidence that they exited markets after mills became certified; underrepresentation should not automatically be called certification-caused exclusion. [T1:W23]\n\n**Climatic / Atmospheric**\n- **Indonesia and receiving systems:** Production-related emissions can offset intended biofuel climate benefits. Outcomes depend on forest conversion, soil type, plantation age, and processing practices—not simply whether the fuel is biological. [T1:W12]\n- **All named systems:** Emissions enter a shared atmosphere, but country-specific climate damages attributable to these trade relationships are not quantified in the supplied evidence.\n\n## 5. Cross-coupling Interactions\n\n- **Amplification:** Distant demand can strengthen incentives for plantation investment, processing, and land conversion within Indonesia. Global price associations with expansion support this connection, but do not establish that exports caused every observed instance of clearing. [T1:1] [T1:W8]\n\n- **Offset:** At a given production level, increased Indonesian biodiesel consumption can reduce exportable supply. Bank Indonesia’s higher-blend scenarios project foregone export earnings, while the retrieved synthesis also describes situations in which domestic use and exports grow together. The sustainability outcome therefore depends on production responses, not merely the domestic–export allocation. [T1:W24] [T1:8]\n\n- **Spatial tradeoffs:** Distant consumers obtain useful products while Indonesian producing landscapes and communities bear much of the associated land, water, labor, and emissions burden. Benefits and costs also differ within Indonesia: higher revenues for some actors can coexist with losses for forest-dependent communities or workers. [T1:1] [T1:W4] [T1:W7]\n\n- **Displacement:** Restrictions or purchasing changes in one distant market may redirect sales toward other markets rather than reduce Indonesian production. The supplied synthesis reports such redirection, but does not quantify bilateral changes. Whether this affects adjacent India, distant buyers, or competing Malaysian suppliers requires tracing actual orders and production responses. [T1:8]\n\n## 6. Research Gaps and Suggestions\n\n1. **Specify the study boundaries and period.** Which years, provinces, export destinations, and sustainability outcomes are priorities? A national assessment can conceal major differences among districts, communities, and production systems.\n\n2. **Build an origin–destination commodity and financial account.** Connect plantations and mills to domestic food demand, biodiesel demand, India, and distant markets. Distinguish crude oil, refined products, re-exports, and payment recipients. Missing shipment records identified by Trase make this especially consequential. [T1:W2]\n\n3. **Separate footprint attribution from causal impact.** Consumption-based footprints answer “whose demand is associated with production?” They do not automatically answer “how much clearing would disappear without that demand?” Use dated land-use histories and credible counterfactuals. Delayed plantation establishment can complicate commodity attribution. [T1:W9]\n\n4. **Measure distributional outcomes directly.** Combine household and worker data with contracts, land tenure, wages, debt, gender, and livelihood dependence. Separate independent smallholders, contract smallholders, workers, and people losing access to land.\n\n5. **Verify additional neighboring and spillover interactions.** Investigate named investment, trade, labor, and pollution pathways involving Malaysia and other adjacent systems. Do not infer flows from adjacency, or attribute all fires and haze to palm oil; the World Bank explicitly cautions that not all fires were set for oil palm. [T1:W13]\n\n6. **Evaluate governance against counterfactual outcomes.** Compare certification, buyer commitments, public enforcement, and productivity support using both direct and indirect effects. A study of commitments during 2018–2020 found high compliance but no additional deforestation reduction, illustrating why pledge coverage alone is insufficient. [T1:W19] Flow-based governance could connect importer requirements with traceability finance, fair purchasing arrangements, producer support, and Indonesian land-use enforcement.\n\n## 7. Evidence Coverage\n\n**Strongest support within the supplied material:** Plantation expansion, forest conversion, and environmental pressures are supported by the Rulli summary and independent satellite-based research. [T1:1] [T1:W8] Heterogeneous livelihood outcomes are supported by village-level research and detailed partnership investigations, although these sources use different methods and populations. [T1:W4] [T1:W5]\n\n**Direct but uneven trade evidence:** India is identified in the supplied Trase analysis, China in the retrieved literature, and Switzerland in OECD reporting. [T1:W2] [T1:2] [T1:W21] The US and Pakistan roles rely principally on one older summary. [T1:4] The broad regional description does not establish named country-to-country investment pathways. [T1:7]\n\n**Inferred or unverified:** Detailed payment distribution, bilateral information networks, importing-country land-use effects, and Malaysia’s proposed spillover response are analytical mappings or hypotheses—not measured findings. Malaysia’s importance as a producer does not independently prove an effect from Indonesian exports. [T1:W1]\n\n**Interpretive limitations:** The supplied evidence consists of passages and web-result summaries rather than independently checked full texts. Historical case investigations, footprint models, scenarios, and causal evaluations should not be treated as interchangeable. In particular, Trase’s historical deforestation exposure is not annual clearing, and modeled water pressures do not negate site-specific measurements showing low nutrient impacts under particular conditions. [T1:W2] [T1:W14]",
  "research_gaps": [
    "**Specify the study boundaries and period.** Which years, provinces, export destinations, and sustainability outcomes are priorities? A national assessment can conceal major differences among districts, communities, and production systems.",
    "**Build an origin–destination commodity and financial account.** Connect plantations and mills to domestic food demand, biodiesel demand, India, and distant markets. Distinguish crude oil, refined products, re-exports, and payment recipients. Missing shipment records identified by Trase make this especially consequential. [T1:W2]",
    "**Separate footprint attribution from causal impact.** Consumption-based footprints answer “whose demand is associated with production?” They do not automatically answer “how much clearing would disappear without that demand?” Use dated land-use histories and credible counterfactuals. Delayed plantation establishment can complicate commodity attribution. [T1:W9]",
    "**Measure distributional outcomes directly.** Combine household and worker data with contracts, land tenure, wages, debt, gender, and livelihood dependence. Separate independent smallholders, contract smallholders, workers, and people losing access to land.",
    "**Verify additional neighboring and spillover interactions.** Investigate named investment, trade, labor, and pollution pathways involving Malaysia and other adjacent systems. Do not infer flows from adjacency, or attribute all fires and haze to palm oil; the World Bank explicitly cautions that not all fires were set for oil palm. [T1:W13]",
    "**Evaluate governance against counterfactual outcomes.** Compare certification, buyer commitments, public enforcement, and productivity support using both direct and indirect effects. A study of commitments during 2018–2020 found high compliance but no additional deforestation reduction, illustrating why pledge coverage alone is insufficient. [T1:W19] Flow-based governance could connect importer requirements with traceability finance, fair purchasing arrangements, producer support, and Indonesian land-use enforcement."
  ],
  "telecoupling": {
    "systems": [
      {
        "role": "sending",
        "system_scope": "distant",
        "name": "Indonesia — palm oil"
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "Indonesia — payments and policy information",
        "human_subsystem": "Producers, workers, processors, exporters, financial institutions, and regulators.",
        "natural_subsystem": "Supplying plantations, tropical forests, peatlands, soils, and watersheds.",
        "geographic_scope": "Indonesia nationally, with origins disaggregated by producing landscape wherever possible."
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "China — palm oil"
      },
      {
        "role": "sending",
        "system_scope": "distant",
        "name": "China — buyer payments and orders",
        "human_subsystem": "Importers, processors, manufacturers, consumers, and trade authorities.",
        "natural_subsystem": "Oilseed cropland, freshwater resources, and environments affected by processing and alternative oil provision.",
        "geographic_scope": "China. The retrieved literature identifies China as an important destination for processed Indonesian palm oil. [T1:2]"
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "European Union — palm oil"
      },
      {
        "role": "sending",
        "system_scope": "distant",
        "name": "European Union — payments and regulatory information",
        "human_subsystem": "Importers, manufacturers, consumers, EU institutions, and member-state authorities.",
        "natural_subsystem": "Agricultural landscapes and ecosystems involved in alternative vegetable-oil production and processing.",
        "geographic_scope": "EU member states; regulatory authority also operates at bloc level. [T1:4] [T1:W22]"
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "United States — palm oil"
      },
      {
        "role": "sending",
        "system_scope": "distant",
        "name": "United States — payments and policy signals",
        "human_subsystem": "Importers, food and industrial manufacturers, consumers, and policymakers.",
        "natural_subsystem": "Oilseed-producing landscapes, soils, water resources, and processing environments.",
        "geographic_scope": "United States. Its receiving role and bioenergy-policy influence are identified in the supplied 2019 summary, not established here with current bilateral quantities. [T1:4]"
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "Pakistan — palm oil"
      },
      {
        "role": "sending",
        "system_scope": "distant",
        "name": "Pakistan — buyer payments",
        "human_subsystem": "Importers, edible-oil processors, consumers, and trade institutions.",
        "natural_subsystem": "Domestic oilseed cropland and water resources relevant to alternative oil supply.",
        "geographic_scope": "Pakistan; identified as a destination in the supplied 2019 summary. [T1:4]"
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "Switzerland — palm oil"
      },
      {
        "role": "sending",
        "system_scope": "distant",
        "name": "Switzerland — payments and sustainability-related purchasing requirements",
        "human_subsystem": "Importers, manufacturers, consumers, and authorities administering preferential trade conditions.",
        "natural_subsystem": "Agricultural land, freshwater resources, and environments associated with alternative food-oil supply.",
        "geographic_scope": "Switzerland. OECD reports 21 tonnes of qualifying Indonesian palm oil imports under sustainability-linked preferences in 2022. [T1:W21]"
      },
      {
        "role": "spillover",
        "name": "Malaysia — **candidate, not a demonstrated spillover",
        "human_subsystem": "Malaysian plantation companies, smallholders, refiners, exporters, workers, and relevant authorities.",
        "natural_subsystem": "Oil palm landscapes and remaining forest and peat habitats that could experience altered production pressure.",
        "geographic_scope": "Malaysia, including producing landscapes in Peninsular Malaysia and Malaysian Borneo.",
        "potential_mechanism": "Changes in buyers’ sourcing from Indonesia could affect Malaysian orders, prices, investment, and land-use incentives. Malaysia’s importance as a co-producer supports investigating this possibility, but does not establish that such effects occurred. [T1:W1]",
        "description": "Malaysia’s spillover role concerns its **indirect relationship to an Indonesia–distant-market interaction**. It is not classified as a direct Indonesian trading or investment partner solely because the countries are adjacent."
      }
    ],
    "flows": [
      {
        "category": "matter",
        "direction": "Indonesia → China, EU member states, United States, Pakistan, and Switzerland.",
        "description": "Crude or processed palm oil and feedstocks, with product composition differing by market. China’s processed-oil relationship is specifically identified; other historical destinations appear in the Rulli summary. [T1:2] [T1:4] **Quantity qualification:** Switzerland’s 21 tonnes concern qualifying preferential imports in 2022—not its entire palm oil market. No such preferential imports were recorded in 2023–2024; that does not demonstrate that all Indonesian–Swiss palm oil trade ceased. [T1:W21]"
      },
      {
        "category": "capital",
        "direction": "Distant buyers → Indonesian exporters → domestic suppliers and investment.",
        "description": "Export payments and subsequent domestic allocation. The international payment belongs here; its internal reinvestment and distribution also belong in intracoupling. Neither the detailed payment network nor benefit shares are supplied."
      },
      {
        "category": "information",
        "direction": "EU and US policymakers → Indonesian authorities and firms; Indonesian suppliers → distant buyers.",
        "description": "Bioenergy-policy signals, sustainability requirements, production-location information, and due-diligence documentation. EU legal requirements establish obligations, not evidence that compliance has already reduced Indonesian deforestation. [T1:4] [T1:W22]"
      },
      {
        "category": "matter",
        "direction": "Indonesian production → distant consumption, through embodied-resource accounting.",
        "description": "Water use and pollution-related footprints are allocated to traded products. Associated emissions should also be accounted for, but “embodied emissions” are not a physical shipment of the gases emitted during production. [T1:1] [T1:4]"
      },
      {
        "category": "information",
        "direction": "Shared importing markets → Malaysian firms and producers.",
        "description": "Possible changes in prices, orders, and sourcing expectations constitute a **candidate** connection to Malaysia. The supplied evidence does not establish its direction, magnitude, or land-use consequences."
      }
    ],
    "agents": [
      {
        "level": "individuals / households",
        "name": "**Indonesian producers, workers, and affected communities; distant consumers**",
        "description": "Make production, labor, livelihood, and purchasing decisions."
      },
      {
        "level": "firms / traders / corporations",
        "name": "**Exporters, multinational traders, refiners, manufacturers, and importing firms**",
        "description": "Connect production locations with distant demand and distribute purchasing requirements."
      },
      {
        "level": "governments / policymakers",
        "name": "**Indonesian authorities and distant-market policymakers**",
        "description": "Influence production incentives, bioenergy demand, market access, and legal obligations."
      },
      {
        "level": "organizations / NGOs",
        "name": "**Certification organizations, researchers, labor organizations, and environmental NGOs**",
        "description": "Set or assess standards, investigate impacts, and support accountability."
      },
      {
        "level": "firms / traders / corporations",
        "name": "**Malaysian producers and processors**",
        "description": "Potentially respond to altered sourcing, but this response remains to be demonstrated."
      }
    ],
    "causes": {
      "economic": [
        "Global food, industrial, and biofuel demand creates incentives for Indonesian production and exports. International prices influence expected plantation returns. [T1:4] [T1:W8]"
      ],
      "political / institutional": [
        "Foreign renewable-energy policies can increase feedstock demand, while sustainability requirements alter market-access conditions. [T1:4]",
        "EU rules require covered products to be deforestation-free, legally produced, and supported by due diligence, including production-location information. [T1:W22]"
      ],
      "cultural / social / demographic": [
        "Dietary patterns, population-related consumption, and demand for manufactured products contribute to distant demand. [T1:4]"
      ],
      "technological / infrastructural": [
        "Refining capacity, processing capabilities, and logistics enable Indonesia to supply differentiated products. The retrieved literature documents a major historical increase in Indonesian processing capacity. [T1:2]"
      ]
    },
    "effects": {
      "economic": [
        "**Indonesia:** Distant markets provide sales opportunities but also expose suppliers to changing prices and purchasing conditions. Export earnings are not equivalent to net social benefits.",
        "**Receiving systems:** Buyers obtain food, manufacturing, and energy inputs; compliance can introduce additional costs.",
        "**Malaysia, candidate spillover:** Procurement changes could increase or decrease business opportunities. No net effect is established."
      ],
      "political / institutional": [
        "**EU and Indonesian suppliers:** Due-diligence requirements create traceability and documentation obligations. [T1:W22]",
        "**Switzerland and Indonesian suppliers:** Sustainability-linked preferences illustrate conditional market access, but observed uptake was very small during the reported period. [T1:W21]"
      ],
      "ecological / biological": [
        "**Indonesia:** Export-linked demand is associated with land-conversion and habitat pressures, although the additional effect of exports must be separated from domestic demand and other land-use drivers. [T1:1]",
        "**Indonesia:** An evaluation of RSPO certification during 2001–2015 estimated reduced deforestation within certified plantations, but detected no causal effect on peatland forest loss or active fires. [T1:W18]",
        "**Malaysia, candidate spillover:** Any change in forest pressure depends on whether altered demand affects yields, existing plantation output, or new expansion. None is demonstrated here."
      ],
      "cultural / social / demographic": [
        "**Indonesia:** International supply chains connect distant buyers with documented labor and livelihood harms in particular producing locations. [T1:W6] [T1:W7]",
        "**Indonesia:** Certified-market participation is uneven. Independent smallholders were underrepresented, but the relevant study found no evidence that they exited markets after mills became certified; underrepresentation should not automatically be called certification-caused exclusion. [T1:W23]"
      ],
      "climatic / atmospheric": [
        "**Indonesia and receiving systems:** Production-related emissions can offset intended biofuel climate benefits. Outcomes depend on forest conversion, soil type, plantation age, and processing practices—not simply whether the fuel is biological. [T1:W12]",
        "**All named systems:** Emissions enter a shared atmosphere, but country-specific climate damages attributable to these trade relationships are not quantified in the supplied evidence."
      ]
    }
  }
}
```

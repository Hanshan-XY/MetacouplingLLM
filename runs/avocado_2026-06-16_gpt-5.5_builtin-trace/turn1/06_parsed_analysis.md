# 06 — Parsed analysis

`result.parsed` re-parsed into structured fields.

```json
{
  "country_pericoupling_info": {
    "focal_country": "Mexico (MEX)",
    "mode": "national",
    "pair_results": "Mexico (MEX) ↔ United States (USA): PERICOUPLED; Mexico (MEX) ↔ Canada (CAN): TELECOUPLED; Mexico (MEX) ↔ Japan (JPN): TELECOUPLED; Mexico (MEX) ↔ El Salvador (SLV): TELECOUPLED; Mexico (MEX) ↔ Honduras (HND): TELECOUPLED",
    "core_subnational_regions": "Jalisco, México, Michoacán de Ocampo",
    "note": "LLM classification is consistent with the coupling database."
  },
  "coupling_classification": "- **Intracoupling**: Present. Avocado production creates within-system interactions in Mexico, especially in Michoacán and Jalisco, among orchards, forests, water resources, rural communities, labor markets, packers, certification systems, and local governance. Evidence links avocado expansion to land-use change, water use, agrochemical pollution, employment, inequality, organized-crime pressures, and community governance within producing regions [T1:W5, T1:W7, T1:W11, T1:W12].\n\n- **Pericoupling**: Present. Mexico and the United States are adjacent systems, and the U.S. is the dominant destination for Mexican avocado exports: USDA reports about 80% of Mexican avocado exports go to the United States, while Data México reports US$3.444 billion in 2024 avocado exports to the U.S. [T1:W1, T1:W2]. This makes the Mexico–U.S. avocado relationship a major adjacent-system coupling at the country scale.\n\n- **Telecoupling**: Present. Mexican avocado production is also connected to distant receiving systems, including Canada, Japan, El Salvador, Honduras, and distant U.S. urban consumer landscapes at the subnational scale. Data México identifies Canada, Japan, El Salvador, and Honduras as major 2024 destinations after the United States [T1:W2], and the literature explicitly frames avocado production and trade as a telecoupled system driven by international demand [T1:1, T1:3, T1:5].\n\n---",
  "cross_coupling_interactions": [
    "**Amplification between intracoupling, pericoupling, and telecoupling**: U.S. and global demand amplify within-Mexico land-use decisions by increasing the profitability of avocado orchards relative to forests, subsistence crops, or mixed farming. Capital and information from adjacent and distant markets strengthen local processes such as orchard expansion, irrigation investment, packing-house development, and certification uptake. The strongest evidence is for U.S.-linked demand and Michoacán deforestation, including the finding that about 20% of Michoacán deforestation from 2001 to 2017 was associated with avocado plantation expansion [T1:W6].",
    "**Spatial tradeoffs**: Receiving systems gain food access, retail profits, and consumer benefits, while many ecological and social costs remain in Mexican producing regions. These costs include deforestation, water scarcity, forest-carbon loss, agrochemical contamination, and risks to rural and Indigenous communities [T1:W5, T1:W7, T1:W12, T1:W14, T1:W16]. This is consistent with the landscape-sustainability framing that U.S. urban consumption can mask offstage environmental burdens in Michoacán [T1:5].",
    "**Displacement**: If certification, enforcement, water scarcity, or reputational risk constrain production in parts of Michoacán, avocado expansion may shift to Jalisco, other Mexican states, or competitor countries such as Peru and Chile. Evidence already shows Jalisco’s growing role in Mexican production and exports [T1:W1, T1:W2, T1:W17], while Ortiz et al. identify Mexico, Chile, and Peru as interconnected avocado producers whose trade dynamics need further investigation [T1:4].",
    "**Feedback loops**: Environmental and social effects in Mexico are beginning to feed back into governance and market flows. Deforestation, water conflict, and NGO or academic documentation have contributed to CEC scrutiny, retailer attention, and deforestation-free certification efforts [T1:W4, T1:W5, T1:W6, T1:W23]. If certification becomes credible and enforceable, it could reduce flows from recently deforested orchards; if it remains weak or partial, export demand may continue reinforcing land conversion and water capture.",
    "**Coupling transformations**: Mexican avocado production moved from more local and domestic coupling toward strong pericoupling and telecoupling as trade liberalization, U.S. market access, and global demand expanded export chains [T1:2, T1:3, T1:W8]. Potential decoupling could occur through drought, phytosanitary suspension, violence, consumer backlash, or import restrictions. Recoupling could occur through traceability, credible deforestation-free certification, water-footprint verification, community governance models, or geographic shifts to new producing regions."
  ],
  "evidence_coverage_note": "Strong evidence base: The trade structure is well supported by government sources showing Mexico’s production scale, export value, and destination concentration, especially the dominance of the U.S. market [T1:W1, T1:W2]. The environmental-impact claims are also well supported by peer-reviewed and NGO sources linking avocado expansion to deforestation, forest fragmentation, water scarcity, agrochemical risks, and carbon loss in Michoacán and Jalisco [T1:W5, T1:W6, T1:W7, T1:W12, T1:W14, T1:W16, T1:W17].\n\nModerate evidence: Social and governance effects are supported by a mix of peer-reviewed studies, NGO reports, government communications, and news reporting. The strongest systematic evidence concerns organized crime and homicide risk where export value increases under criminal presence [T1:W21, T1:W22]. Water conflict and environmental-defender risks are documented through AP reporting and should be paired with more local fieldwork and official records [T1:W18, T1:W19].\n\nTelecoupling evidence is conceptually strong. The retrieved literature explicitly frames avocado production and trade as telecoupled, with international demand shaping local production and sustainability burdens [T1:1, T1:3, T1:5]. Mexico-specific telecoupling to U.S. consumption is directly supported by the Journal of Environmental Management study and landscape-sustainability synthesis [T1:W6, T1:5].\n\nLimited evidence: Energy flows, full life-cycle greenhouse-gas emissions, detailed labor migration, and exact capital distribution among growers, packers, retailers, and criminal actors are inferred from the structure of the avocado supply chain rather than directly quantified in the provided sources. Spillover effects on Peru, Chile, and distant forest-supply regions are plausible and partly supported by Ortiz et al. and Dade et al., but require more direct comparative trade and land-use evidence [T1:1, T1:4, T1:5].",
  "intracoupling": {
    "systems": [
      {
        "role": "focal",
        "name": "Mexico’s avocado-producing regions, with emphasis on Michoacán and Jalisco",
        "human_subsystem": "Avocado growers, smallholders, large landowners, agribusiness firms, packers, exporters, farmworkers, Indigenous and rural communities, municipal authorities, state and federal agencies such as SIAP, SENASICA, environmental regulators, water authorities, certification programs, and in some areas organized-crime actors.",
        "natural_subsystem": "Pine, pine-oak, and temperate forests; agricultural soils; surface water and groundwater; watersheds; biodiversity areas including landscapes near the Monarch Butterfly Biosphere Reserve; avocado orchards; pollinators and pest species; local climate and fire regimes.",
        "geographic_scope": "Mexico, especially the Michoacán Avocado Belt and expanding production areas in Jalisco. Michoacán accounts for the largest share of production and export value: USDA reports Michoacán accounts for 68% of production and Jalisco 12% [T1:W1], while Data México reports Michoacán exported US$3.525 billion in avocados in 2024 and Jalisco US$333 million [T1:W2]."
      }
    ],
    "flows": [
      {
        "category": "matter",
        "direction": "Forests / mixed agricultural lands → avocado orchards within Michoacán and Jalisco",
        "description": "Land cover and land use shift from forests, subsistence crops, or mixed farming systems toward avocado orchards. Peer-reviewed and NGO sources link avocado expansion to deforestation, forest fragmentation, and conversion of forested land [T1:W6, T1:W7, T1:W11, T1:W17]."
      },
      {
        "category": "matter",
        "direction": "Surface water and groundwater → avocado orchards",
        "description": "Irrigation water is withdrawn from springs, streams, wells, reservoirs, or illegal water intakes to support avocado production. Water-footprint studies in Uruapan and Ziracuaretiro show much higher water demand in irrigated orchards than rainfed orchards and estimate that avocado cultivation can exceed legally granted agricultural water volumes in dry years or specific municipalities [T1:W12, T1:W13]."
      },
      {
        "category": "matter",
        "direction": "Agrochemical suppliers / farms → soils, runoff, rivers, wells, and orchard ecosystems",
        "description": "Fertilizers, herbicides, insecticides, and fungicides are applied within production landscapes. A 2024 study in the eastern Avocado Belt detected pesticide residues and degradation products in water samples and identified glyphosate, benomyl, and imidacloprid among the most applied pesticides [T1:W14]."
      },
      {
        "category": "capital",
        "direction": "Avocado sales and export revenues → growers, packers, laborers, landowners, local businesses, and sometimes criminal rent extraction",
        "description": "Avocado revenues circulate locally through employment, land rents, orchard investment, packing operations, and municipal economic growth. However, benefits are unevenly distributed; profits may concentrate among agribusinesses and larger landowners, while workers and smallholders capture less value [T1:4, T1:W7, T1:W11]."
      },
      {
        "category": "information",
        "direction": "Government agencies, certification programs, technical advisers, packers, and growers → orchard-management decisions",
        "description": "Information on export eligibility, phytosanitary standards, deforestation-free certification, pesticide use, irrigation techniques, and market requirements shapes local production practices. USDA notes that only Michoacán and Jalisco are eligible for U.S. avocado exports under the SENASICA–USDA APHIS binational export program [T1:W1]."
      },
      {
        "category": "people",
        "direction": "Rural communities → orchards, packing houses, transport, and monitoring activities",
        "description": "Labor moves within producing regions for orchard establishment, harvesting, packing, transport, water infrastructure construction, and forest monitoring. The avocado sector has created employment, though much is seasonal and benefits are uneven [T1:4, T1:W7]."
      },
      {
        "category": "energy",
        "direction": "Fuel and electricity → irrigation pumps, transport, cold-chain logistics, packing houses",
        "description": "Energy supports irrigation, orchard machinery, packing, refrigeration, and transport within producing regions. Direct quantitative evidence in the provided sources is limited, but these are necessary production and logistics inputs for an export-oriented fresh-fruit chain."
      }
    ],
    "agents": [
      {
        "level": "individuals / households",
        "name": "**Smallholder avocado growers**",
        "description": "Decide whether to plant, intensify, irrigate, or shift away from other crops; may benefit from high prices but face water scarcity, market standards, and dependence on intermediaries."
      },
      {
        "level": "individuals / households",
        "name": "**Farmworkers and rural households**",
        "description": "Provide labor for orchards and packing; experience employment gains but may also face precarious seasonal work, water scarcity, insecurity, and exposure to agrochemicals."
      },
      {
        "level": "individuals / households",
        "name": "**Indigenous and communal landholders**",
        "description": "Manage forest commons, water access, and local land-use rules; some communities have resisted forest conversion or created community-based avocado models [T1:W24]."
      },
      {
        "level": "firms / traders / corporations",
        "name": "**Packers, exporters, and agribusiness firms**",
        "description": "Coordinate harvesting, packing, certification, export logistics, and market access; can concentrate profits and shape growers’ production practices."
      },
      {
        "level": "firms / traders / corporations",
        "name": "**Agrochemical and irrigation suppliers**",
        "description": "Provide inputs and technical advice that influence pesticide use, fertilizer use, irrigation intensity, and production costs."
      },
      {
        "level": "governments / policymakers",
        "name": "**SENASICA, USDA APHIS partner agencies, SIAP, water authorities, and environmental regulators**",
        "description": "Set phytosanitary, statistical, water, land-use, and export-eligibility rules; SIAP provides official crop statistics across Mexico [T1:W3]."
      },
      {
        "level": "governments / policymakers",
        "name": "**Michoacán state environmental authorities**",
        "description": "Implement or promote deforestation-free certification and traceability systems such as Pro Forest Avocado / Guardián Forestal [T1:W23]."
      },
      {
        "level": "organizations / NGOs",
        "name": "**Climate Rights International, Global Forest Watch / WRI, CEC, universities, and research organizations**",
        "description": "Produce evidence, monitoring, legal submissions, and policy recommendations on deforestation, water, violence, and supply-chain responsibility [T1:W4, T1:W5, T1:W6, T1:W17]."
      },
      {
        "level": "non-human agents",
        "name": "**Avocado trees**",
        "description": "Actively structure water demand, land-use permanence, agrochemical regimes, and orchard expansion dynamics as a perennial crop."
      },
      {
        "level": "non-human agents",
        "name": "**Pests, pathogens, pollinators, and forest species**",
        "description": "Influence orchard management, pesticide use, biodiversity outcomes, and ecosystem-service dynamics."
      }
    ],
    "causes": {
      "economic": [
        "High avocado profitability compared with many subsistence or traditional crops encourages land conversion and orchard expansion.",
        "Export revenues create strong incentives for growers, packers, and landowners to meet international standards and expand production.",
        "Profit concentration may favor larger growers and agribusiness firms, while smallholders and workers face unequal benefit distribution [T1:4, T1:W7].",
        "Organized-crime rent seeking is encouraged where avocado export value is high and territorial criminal groups are present [T1:W21, T1:W22]."
      ],
      "political / institutional": [
        "Trade liberalization and export-oriented agricultural policy helped transform avocado from a more local production system into a globalized value chain [T1:2, T1:3, T1:W8].",
        "The SENASICA–USDA APHIS export program determines which Mexican states and orchards can access the U.S. market [T1:W1].",
        "Weak or uneven enforcement of forest, water, and land-use laws is alleged in the CEC submission and NGO reporting [T1:W4, T1:W5].",
        "New deforestation-free certification and traceability programs are emerging as institutional responses to sustainability concerns [T1:W1, T1:W23]."
      ],
      "ecological / biological": [
        "Avocado’s ecological suitability overlaps with temperate forest zones in Michoacán, increasing the risk that forested lands become attractive for orchard conversion [T1:W11].",
        "Forest fragmentation, fire dynamics, and biodiversity vulnerability shape where environmental impacts are most severe [T1:W15, T1:W17].",
        "Pest and disease risks encourage pesticide use and technical input dependence."
      ],
      "technological / infrastructural": [
        "Roads, packing houses, cold-chain logistics, irrigation infrastructure, and proximity to settlements facilitate expansion and export commercialization. Studies identify proximity to roads, existing agriculture, and localities as drivers of expansion [T1:W9].",
        "Remote sensing, supply-chain reconstruction, and traceability systems increasingly enable monitoring of deforestation and export-chain risk [T1:W6, T1:W23]."
      ],
      "cultural / social / demographic": [
        "Domestic and international marketing of avocado as a healthy “superfood” has helped increase demand [T1:2, T1:3].",
        "Rural livelihood aspirations, reduced out-migration, and community employment opportunities motivate participation in avocado production [T1:W7].",
        "Community norms and Indigenous governance institutions can either constrain unsustainable expansion or be weakened by land speculation, inequality, and external market pressure [T1:W11, T1:W24]."
      ],
      "hydrological": [
        "Avocado orchards require substantial water, especially under irrigated production. In Uruapan, irrigated orchards had a far higher water footprint than rainfed orchards, while providing only a small yield increase [T1:W12].",
        "Drought and water scarcity intensify competition among orchards, households, subsistence agriculture, and ecosystems [T1:W12, T1:W18]."
      ],
      "climatic / atmospheric": [
        "Climate change may alter avocado suitability, water stress, and expansion pressure, as shown in Chile and in modeling studies of future Mexican avocado expansion under climate conditions [T1:6, T1:W10].",
        "Forest-to-orchard conversion reduces aboveground carbon storage, contributing to climate-related sustainability concerns [T1:W16]."
      ],
      "geological / geomorphological": [
        "Soil type, slope, elevation, and terrain influence where avocado expansion is profitable and ecologically suitable. Studies identify slope, elevation, and dwindling availability of suitable Andosol soils as relevant determinants [T1:W8, T1:W9].",
        "Soil disturbance, erosion, and nutrient change are linked to land conversion and intensive orchard management [T1:W7, T1:W16]."
      ]
    },
    "effects": {
      "economic": [
        "Avocado production generates major export value: Mexico exported US$3.969 billion in fresh or dried avocados in 2024, with Michoacán accounting for US$3.525 billion [T1:W2].",
        "Employment and local income increase in producing regions, but benefits may be seasonal and unevenly distributed [T1:4, T1:W7].",
        "Extortion and criminal rent extraction can drain money from the legal economy and distort local development [T1:W22].",
        "Land values and production costs may rise, making participation more difficult for smaller producers."
      ],
      "political / institutional": [
        "Sustainability concerns have triggered cross-border environmental scrutiny, including a CEC submission under the USMCA/CUSMA environmental cooperation framework alleging weak enforcement of forest and water laws [T1:W4].",
        "Certification and traceability programs are becoming more important in state and export governance; Michoacán reported certification mechanisms excluding orchards deforested after January 2018 and properties affected by fires after January 2012 [T1:W23].",
        "Local governance may be strengthened where communities use avocado revenues to support forest monitoring and water committees, as reported in community-based models in Zitácuaro [T1:W24].",
        "In other areas, governance may be weakened by illegal logging, unauthorized water capture, violence, and organized-crime control [T1:W18, T1:W19, T1:W20]."
      ],
      "ecological / biological": [
        "Deforestation and forest fragmentation are major effects. One peer-reviewed study estimated that about 20% of Michoacán deforestation from 2001 to 2017 was associated with avocado plantation expansion [T1:W6].",
        "Biodiversity risks arise where orchards overlap with Key Biodiversity Areas or expand near the Monarch Butterfly Biosphere Reserve [T1:W6, T1:W14, T1:W17].",
        "Forest fires may be associated with conversion pathways; one study identified avocado orchards as a consistent driver of forest fires across remnant forest patches in the Michoacán Avocado Belt [T1:W15].",
        "Agrochemical use may affect aquatic ecosystems and non-target organisms; pesticide residues were detected in water samples in the eastern Avocado Belt [T1:W14]."
      ],
      "technological / infrastructural": [
        "Expansion stimulates packing, transport, cold-chain, irrigation, and monitoring infrastructure.",
        "Traceability and remote-sensing tools may improve detection of deforestation and support deforestation-free sourcing [T1:W6, T1:W23].",
        "Irrigation infrastructure, including unlicensed intakes and holding ponds, can intensify conflict over local water access [T1:W18]."
      ],
      "cultural / social / demographic": [
        "Employment and income can reduce poverty and out-migration in some regions, but benefits are uneven and may reinforce inequality [T1:W7].",
        "Indigenous and rural communities may experience weakened cohesion when land-use change, privatization, external investors, or criminal pressure undermine communal institutions [T1:W11, T1:W19].",
        "Environmental defenders and community leaders may face threats or violence in contexts where illegal logging and avocado expansion overlap [T1:W19].",
        "Public health concerns may arise from pesticide exposure, water scarcity, and insecurity [T1:W14, T1:W18]."
      ],
      "hydrological": [
        "Avocado irrigation can intensify water scarcity, reduce streamflow, deplete legal water concessions, and create water-rights conflicts [T1:W12, T1:W13, T1:W18].",
        "Forest conversion can reduce water filtration, aquifer recharge, and watershed regulation services, as alleged in the CEC submission and described in forest-service claims [T1:W4].",
        "Water impacts are spatially variable: warmer or drier municipalities may have much higher water footprints than rainfed or cooler regions [T1:W12, T1:W13]."
      ],
      "climatic / atmospheric": [
        "Conversion of pine-oak forests to orchards reduces aboveground carbon storage [T1:W16].",
        "Deforestation, fires, fertilizer use, transport, refrigeration, and export logistics can contribute to greenhouse-gas emissions, though the provided sources are stronger on land-carbon effects than full life-cycle emissions.",
        "Climate change may increase future production suitability in some areas while worsening water stress in others, creating uneven risks."
      ],
      "geological / geomorphological": [
        "Soil-health degradation, erosion, and nutrient imbalance can result from land conversion and intensive orchard management [T1:W7, T1:W16].",
        "Over-fertilization may alter soil nutrient dynamics and complicate assessment of orchard sustainability [T1:W16]."
      ]
    }
  },
  "is_parsed": true,
  "map_data": {
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
  },
  "pericoupling": {
    "systems": [
      {
        "role": "sending",
        "system_scope": "adjacent",
        "name": "Mexico, especially Michoacán and Jalisco",
        "human_subsystem": "Growers, farmworkers, packers, exporters, SENASICA, state governments, environmental and water agencies, local communities, agribusinesses, transport firms, and certification programs.",
        "natural_subsystem": "Avocado orchards, pine-oak forests, watersheds, soils, biodiversity areas, surface water, groundwater, and forest carbon stocks.",
        "geographic_scope": "Mexico’s avocado-producing and export-certified regions, especially Michoacán and Jalisco."
      },
      {
        "role": "receiving",
        "system_scope": "adjacent",
        "name": "United States",
        "human_subsystem": "Consumers, retailers, restaurants, importers, wholesalers, marketing boards, USDA APHIS, food-service firms, sustainability teams, and trade-policy actors.",
        "natural_subsystem": "U.S. food-consumption landscapes, domestic agricultural land that may be partially spared or reallocated because avocados are imported, and environmental burdens embedded in imported food consumption.",
        "geographic_scope": "United States national market; major urban consumer regions; import and distribution networks."
      }
    ],
    "flows": [
      {
        "category": "matter",
        "direction": "Mexico → United States",
        "description": "Fresh or dried avocados move from Mexican production regions to U.S. consumers, retailers, and food-service markets. The U.S. receives about 80% of Mexican avocado exports by volume [T1:W1], and Data México reports US$3.444 billion in 2024 exports to the U.S. [T1:W2]."
      },
      {
        "category": "capital",
        "direction": "United States → Mexico",
        "description": "Payments for avocado imports flow from U.S. importers, retailers, food-service firms, and consumers to Mexican exporters, packers, growers, and intermediaries. These revenues create incentives for orchard expansion and intensification."
      },
      {
        "category": "information",
        "direction": "United States ↔ Mexico",
        "description": "Market signals, prices, retailer sourcing requirements, phytosanitary rules, sustainability expectations, and deforestation-free certification requirements move between the U.S. and Mexican avocado systems. USDA notes that only Michoacán and Jalisco are eligible for U.S. export under the binational program [T1:W1]."
      },
      {
        "category": "energy",
        "direction": "Mexico → United States, embodied in traded avocados and logistics",
        "description": "Energy is embodied in irrigation, fertilizer, harvesting, packing, refrigeration, and transport required to supply the U.S. market. Direct energy accounting is not provided in the sources, but it is part of the export logistics system."
      },
      {
        "category": "people",
        "direction": "Mexico ↔ United States, indirectly through trade governance and inspection systems",
        "description": "Inspectors, trade officials, company representatives, auditors, technical consultants, and supply-chain managers move or coordinate across the Mexico–U.S. boundary. This flow is less central than matter and capital flows but supports export eligibility and compliance."
      }
    ],
    "agents": [
      {
        "level": "individuals / households",
        "name": "**U.S. consumers**",
        "description": "Drive demand through avocado consumption, including demand associated with healthy diets, “superfood” narratives, restaurants, and retail purchases [T1:3]."
      },
      {
        "level": "individuals / households",
        "name": "**Mexican growers and farmworkers**",
        "description": "Produce and harvest avocados for the U.S. market; experience local benefits and costs."
      },
      {
        "level": "firms / traders / corporations",
        "name": "**U.S. retailers, importers, restaurants, and wholesalers**",
        "description": "Purchase, market, distribute, and sell Mexican avocados; can influence sourcing standards and traceability."
      },
      {
        "level": "firms / traders / corporations",
        "name": "**Mexican packers and exporters**",
        "description": "Aggregate, certify, pack, and ship fruit to the U.S. market."
      },
      {
        "level": "governments / policymakers",
        "name": "**SENASICA and USDA APHIS**",
        "description": "Operate the binational export program that determines export eligibility and phytosanitary compliance [T1:W1]."
      },
      {
        "level": "governments / policymakers",
        "name": "**Trade and environmental agencies under USMCA/CUSMA**",
        "description": "Shape the institutional context for environmental enforcement concerns, including the CEC submission on Michoacán avocado production [T1:W4]."
      },
      {
        "level": "organizations / NGOs",
        "name": "**CEC, Climate Rights International, Global Forest Watch / WRI, academic researchers**",
        "description": "Document and communicate sustainability impacts, deforestation risks, and governance gaps [T1:W4, T1:W5, T1:W6, T1:W17]."
      },
      {
        "level": "non-human agents",
        "name": "**Avocado trees and pests**",
        "description": "Mediate water demand, pesticide use, orchard permanence, and phytosanitary concerns."
      }
    ],
    "causes": {
      "economic": [
        "U.S. demand is the dominant adjacent-system driver of Mexican avocado exports, with the U.S. receiving about 80% of export volume [T1:W1].",
        "High U.S. market value creates strong incentives for Mexican growers and packers to expand production and comply with export requirements.",
        "U.S. consumption growth has been linked to land-use pressure in Michoacán; one study notes U.S. per-capita avocado consumption more than tripled between 2001 and 2017 [T1:W6]."
      ],
      "political / institutional": [
        "Trade liberalization, including the NAFTA-era opening and end of U.S. restrictions on Mexican avocado imports, helped increase Mexico–U.S. avocado flows [T1:W8].",
        "SENASICA–USDA APHIS phytosanitary governance structures determine which regions can export to the U.S. market [T1:W1].",
        "The USMCA/CUSMA environmental cooperation framework provides a venue for allegations that Mexico is not enforcing environmental laws related to avocado expansion [T1:W4]."
      ],
      "ecological / biological": [
        "Avocado suitability in forested highland regions makes export-driven expansion ecologically consequential.",
        "Pest and disease management requirements affect phytosanitary rules, pesticide use, and trade eligibility."
      ],
      "technological / infrastructural": [
        "Cold-chain logistics, highways, packing houses, border-crossing systems, and inspection infrastructure enable high-volume fresh-fruit trade.",
        "Remote sensing and supply-chain reconstruction methods can link U.S. consumption to Mexican deforestation risk [T1:W6]."
      ],
      "cultural / social / demographic": [
        "U.S. dietary trends and avocado marketing as a healthy food or “superfood” increase consumer demand [T1:3, T1:5].",
        "Restaurant, sports-event, and retail cultures normalize high avocado consumption, indirectly shaping land-use incentives in Mexican producing regions."
      ],
      "hydrological": [
        "U.S. demand indirectly increases irrigation demand in Mexican producing regions, especially where growers intensify production to meet export markets.",
        "Drought and local water scarcity in Michoacán make the water burden of export production socially contentious [T1:W12, T1:W18]."
      ],
      "climatic / atmospheric": [
        "Climate-related drought can amplify water conflicts in Mexico while demand from the adjacent U.S. market remains high.",
        "Refrigerated transport and long supply chains may contribute to emissions, though direct life-cycle estimates are not provided in the retrieved evidence."
      ],
      "geological / geomorphological": [
        "Suitable soils, terrain, and elevation in Michoacán and Jalisco shape where U.S.-oriented production expands [T1:W8, T1:W9].",
        "Soil degradation from intensification may reduce long-term productivity and increase management costs."
      ]
    },
    "effects": {
      "economic": [
        "**Sending system — Mexico**: Receives substantial export revenue and employment benefits; Michoacán dominates export value [T1:W2].",
        "**Receiving system — United States**: Gains reliable year-round avocado supply for consumers, retailers, restaurants, and food-service companies.",
        "**Sending system — Mexico**: Profit concentration, extortion, and unequal value capture can reduce local sustainability benefits [T1:4, T1:W21, T1:W22]."
      ],
      "political / institutional": [
        "**Sending system — Mexico**: U.S. market access creates pressure for phytosanitary compliance, traceability, and deforestation-free certification [T1:W1, T1:W23].",
        "**Receiving system — United States**: Retailers and regulators face growing pressure to address imported deforestation and human-rights risks.",
        "**Both systems**: The CEC submission makes avocado sustainability a cross-border environmental governance issue [T1:W4]."
      ],
      "ecological / biological": [
        "**Sending system — Mexico**: U.S.-linked demand contributes to orchard expansion, deforestation, forest fragmentation, and biodiversity risks [T1:W6, T1:W17].",
        "**Receiving system — United States**: Environmental costs of avocado consumption are partly externalized to Mexican production landscapes, consistent with telecoupled ecosystem-service burden arguments [T1:5].",
        "**Sending system — Mexico**: Pesticide use and agrochemical runoff can affect aquatic ecosystems and biodiversity near production areas [T1:W14]."
      ],
      "technological / infrastructural": [
        "**Sending system — Mexico**: Export trade stimulates packing-house development, transport infrastructure, irrigation infrastructure, and certification systems.",
        "**Receiving system — United States**: Import logistics and retail distribution networks expand to support consumer demand."
      ],
      "cultural / social / demographic": [
        "**Sending system — Mexico**: Employment can reduce out-migration and poverty in some areas, but inequality, seasonal work, and violence reduce social sustainability [T1:W7, T1:W20].",
        "**Receiving system — United States**: Consumers benefit from dietary diversity and convenient access, but may remain unaware of environmental and social burdens embedded in supply chains [T1:5, T1:W15].",
        "**Sending system — Mexico**: Rural and Indigenous communities may bear water scarcity, land pressure, and security risks."
      ],
      "hydrological": [
        "**Sending system — Mexico**: Export-driven production increases pressure on springs, streams, groundwater, and legal water concessions [T1:W12, T1:W13, T1:W18].",
        "**Receiving system — United States**: U.S. consumers receive avocados while much of the water burden remains embedded in Mexican production landscapes."
      ],
      "climatic / atmospheric": [
        "**Sending system — Mexico**: Forest conversion reduces aboveground carbon storage [T1:W16].",
        "**Receiving system — United States**: Consumption is linked to embodied land-use emissions and refrigerated supply-chain energy use, though direct quantified emissions are not available in the provided evidence."
      ],
      "geological / geomorphological": [
        "**Sending system — Mexico**: Soil disturbance, erosion, and over-fertilization may follow intensive orchard expansion [T1:W7, T1:W16].",
        "**Receiving system — United States**: Soil impacts are indirect and embodied in imported avocado consumption."
      ]
    }
  },
  "pericoupling_info": null,
  "raw_text": "### 1. Coupling Classification\n\n- **Intracoupling**: Present. Avocado production creates within-system interactions in Mexico, especially in Michoacán and Jalisco, among orchards, forests, water resources, rural communities, labor markets, packers, certification systems, and local governance. Evidence links avocado expansion to land-use change, water use, agrochemical pollution, employment, inequality, organized-crime pressures, and community governance within producing regions [T1:W5, T1:W7, T1:W11, T1:W12].\n\n- **Pericoupling**: Present. Mexico and the United States are adjacent systems, and the U.S. is the dominant destination for Mexican avocado exports: USDA reports about 80% of Mexican avocado exports go to the United States, while Data México reports US$3.444 billion in 2024 avocado exports to the U.S. [T1:W1, T1:W2]. This makes the Mexico–U.S. avocado relationship a major adjacent-system coupling at the country scale.\n\n- **Telecoupling**: Present. Mexican avocado production is also connected to distant receiving systems, including Canada, Japan, El Salvador, Honduras, and distant U.S. urban consumer landscapes at the subnational scale. Data México identifies Canada, Japan, El Salvador, and Honduras as major 2024 destinations after the United States [T1:W2], and the literature explicitly frames avocado production and trade as a telecoupled system driven by international demand [T1:1, T1:3, T1:5].\n\n---\n\n### 2. Intracoupling Analysis — within the focal system\n\n#### 2.1 Systems Identification\n\n**Focal System**: Mexico’s avocado-producing regions, with emphasis on Michoacán and Jalisco\n- **Human subsystem**: Avocado growers, smallholders, large landowners, agribusiness firms, packers, exporters, farmworkers, Indigenous and rural communities, municipal authorities, state and federal agencies such as SIAP, SENASICA, environmental regulators, water authorities, certification programs, and in some areas organized-crime actors.\n- **Natural subsystem**: Pine, pine-oak, and temperate forests; agricultural soils; surface water and groundwater; watersheds; biodiversity areas including landscapes near the Monarch Butterfly Biosphere Reserve; avocado orchards; pollinators and pest species; local climate and fire regimes.\n- **Geographic scope**: Mexico, especially the Michoacán Avocado Belt and expanding production areas in Jalisco. Michoacán accounts for the largest share of production and export value: USDA reports Michoacán accounts for 68% of production and Jalisco 12% [T1:W1], while Data México reports Michoacán exported US$3.525 billion in avocados in 2024 and Jalisco US$333 million [T1:W2].\n\n#### 2.2 Flows Analysis\n\n**Matter Flow**\n- **Direction**: Forests / mixed agricultural lands → avocado orchards within Michoacán and Jalisco\n- **Description**: Land cover and land use shift from forests, subsistence crops, or mixed farming systems toward avocado orchards. Peer-reviewed and NGO sources link avocado expansion to deforestation, forest fragmentation, and conversion of forested land [T1:W6, T1:W7, T1:W11, T1:W17].\n\n**Matter Flow**\n- **Direction**: Surface water and groundwater → avocado orchards\n- **Description**: Irrigation water is withdrawn from springs, streams, wells, reservoirs, or illegal water intakes to support avocado production. Water-footprint studies in Uruapan and Ziracuaretiro show much higher water demand in irrigated orchards than rainfed orchards and estimate that avocado cultivation can exceed legally granted agricultural water volumes in dry years or specific municipalities [T1:W12, T1:W13].\n\n**Matter Flow**\n- **Direction**: Agrochemical suppliers / farms → soils, runoff, rivers, wells, and orchard ecosystems\n- **Description**: Fertilizers, herbicides, insecticides, and fungicides are applied within production landscapes. A 2024 study in the eastern Avocado Belt detected pesticide residues and degradation products in water samples and identified glyphosate, benomyl, and imidacloprid among the most applied pesticides [T1:W14].\n\n**Capital Flow**\n- **Direction**: Avocado sales and export revenues → growers, packers, laborers, landowners, local businesses, and sometimes criminal rent extraction\n- **Description**: Avocado revenues circulate locally through employment, land rents, orchard investment, packing operations, and municipal economic growth. However, benefits are unevenly distributed; profits may concentrate among agribusinesses and larger landowners, while workers and smallholders capture less value [T1:4, T1:W7, T1:W11].\n\n**Information Flow**\n- **Direction**: Government agencies, certification programs, technical advisers, packers, and growers → orchard-management decisions\n- **Description**: Information on export eligibility, phytosanitary standards, deforestation-free certification, pesticide use, irrigation techniques, and market requirements shapes local production practices. USDA notes that only Michoacán and Jalisco are eligible for U.S. avocado exports under the SENASICA–USDA APHIS binational export program [T1:W1].\n\n**People Flow**\n- **Direction**: Rural communities → orchards, packing houses, transport, and monitoring activities\n- **Description**: Labor moves within producing regions for orchard establishment, harvesting, packing, transport, water infrastructure construction, and forest monitoring. The avocado sector has created employment, though much is seasonal and benefits are uneven [T1:4, T1:W7].\n\n**Energy Flow**\n- **Direction**: Fuel and electricity → irrigation pumps, transport, cold-chain logistics, packing houses\n- **Description**: Energy supports irrigation, orchard machinery, packing, refrigeration, and transport within producing regions. Direct quantitative evidence in the provided sources is limited, but these are necessary production and logistics inputs for an export-oriented fresh-fruit chain.\n\n#### 2.3 Agents\n\n- [Individuals / households] **Smallholder avocado growers**: Decide whether to plant, intensify, irrigate, or shift away from other crops; may benefit from high prices but face water scarcity, market standards, and dependence on intermediaries.\n- [Individuals / households] **Farmworkers and rural households**: Provide labor for orchards and packing; experience employment gains but may also face precarious seasonal work, water scarcity, insecurity, and exposure to agrochemicals.\n- [Individuals / households] **Indigenous and communal landholders**: Manage forest commons, water access, and local land-use rules; some communities have resisted forest conversion or created community-based avocado models [T1:W24].\n- [Firms / traders / corporations] **Packers, exporters, and agribusiness firms**: Coordinate harvesting, packing, certification, export logistics, and market access; can concentrate profits and shape growers’ production practices.\n- [Firms / traders / corporations] **Agrochemical and irrigation suppliers**: Provide inputs and technical advice that influence pesticide use, fertilizer use, irrigation intensity, and production costs.\n- [Governments / policymakers] **SENASICA, USDA APHIS partner agencies, SIAP, water authorities, and environmental regulators**: Set phytosanitary, statistical, water, land-use, and export-eligibility rules; SIAP provides official crop statistics across Mexico [T1:W3].\n- [Governments / policymakers] **Michoacán state environmental authorities**: Implement or promote deforestation-free certification and traceability systems such as Pro Forest Avocado / Guardián Forestal [T1:W23].\n- [Organizations / NGOs] **Climate Rights International, Global Forest Watch / WRI, CEC, universities, and research organizations**: Produce evidence, monitoring, legal submissions, and policy recommendations on deforestation, water, violence, and supply-chain responsibility [T1:W4, T1:W5, T1:W6, T1:W17].\n- [Non-human agents] **Avocado trees**: Actively structure water demand, land-use permanence, agrochemical regimes, and orchard expansion dynamics as a perennial crop.\n- [Non-human agents] **Pests, pathogens, pollinators, and forest species**: Influence orchard management, pesticide use, biodiversity outcomes, and ecosystem-service dynamics.\n\n#### 2.4 Causes\n\n**Economic**\n- High avocado profitability compared with many subsistence or traditional crops encourages land conversion and orchard expansion.\n- Export revenues create strong incentives for growers, packers, and landowners to meet international standards and expand production.\n- Profit concentration may favor larger growers and agribusiness firms, while smallholders and workers face unequal benefit distribution [T1:4, T1:W7].\n- Organized-crime rent seeking is encouraged where avocado export value is high and territorial criminal groups are present [T1:W21, T1:W22].\n\n**Political / Institutional**\n- Trade liberalization and export-oriented agricultural policy helped transform avocado from a more local production system into a globalized value chain [T1:2, T1:3, T1:W8].\n- The SENASICA–USDA APHIS export program determines which Mexican states and orchards can access the U.S. market [T1:W1].\n- Weak or uneven enforcement of forest, water, and land-use laws is alleged in the CEC submission and NGO reporting [T1:W4, T1:W5].\n- New deforestation-free certification and traceability programs are emerging as institutional responses to sustainability concerns [T1:W1, T1:W23].\n\n**Ecological / Biological**\n- Avocado’s ecological suitability overlaps with temperate forest zones in Michoacán, increasing the risk that forested lands become attractive for orchard conversion [T1:W11].\n- Forest fragmentation, fire dynamics, and biodiversity vulnerability shape where environmental impacts are most severe [T1:W15, T1:W17].\n- Pest and disease risks encourage pesticide use and technical input dependence.\n\n**Technological / Infrastructural**\n- Roads, packing houses, cold-chain logistics, irrigation infrastructure, and proximity to settlements facilitate expansion and export commercialization. Studies identify proximity to roads, existing agriculture, and localities as drivers of expansion [T1:W9].\n- Remote sensing, supply-chain reconstruction, and traceability systems increasingly enable monitoring of deforestation and export-chain risk [T1:W6, T1:W23].\n\n**Cultural / Social / Demographic**\n- Domestic and international marketing of avocado as a healthy “superfood” has helped increase demand [T1:2, T1:3].\n- Rural livelihood aspirations, reduced out-migration, and community employment opportunities motivate participation in avocado production [T1:W7].\n- Community norms and Indigenous governance institutions can either constrain unsustainable expansion or be weakened by land speculation, inequality, and external market pressure [T1:W11, T1:W24].\n\n**Hydrological**\n- Avocado orchards require substantial water, especially under irrigated production. In Uruapan, irrigated orchards had a far higher water footprint than rainfed orchards, while providing only a small yield increase [T1:W12].\n- Drought and water scarcity intensify competition among orchards, households, subsistence agriculture, and ecosystems [T1:W12, T1:W18].\n\n**Climatic / Atmospheric**\n- Climate change may alter avocado suitability, water stress, and expansion pressure, as shown in Chile and in modeling studies of future Mexican avocado expansion under climate conditions [T1:6, T1:W10].\n- Forest-to-orchard conversion reduces aboveground carbon storage, contributing to climate-related sustainability concerns [T1:W16].\n\n**Geological / Geomorphological**\n- Soil type, slope, elevation, and terrain influence where avocado expansion is profitable and ecologically suitable. Studies identify slope, elevation, and dwindling availability of suitable Andosol soils as relevant determinants [T1:W8, T1:W9].\n- Soil disturbance, erosion, and nutrient change are linked to land conversion and intensive orchard management [T1:W7, T1:W16].\n\n#### 2.5 Effects\n\n**Economic**\n- Avocado production generates major export value: Mexico exported US$3.969 billion in fresh or dried avocados in 2024, with Michoacán accounting for US$3.525 billion [T1:W2].\n- Employment and local income increase in producing regions, but benefits may be seasonal and unevenly distributed [T1:4, T1:W7].\n- Extortion and criminal rent extraction can drain money from the legal economy and distort local development [T1:W22].\n- Land values and production costs may rise, making participation more difficult for smaller producers.\n\n**Political / Institutional**\n- Sustainability concerns have triggered cross-border environmental scrutiny, including a CEC submission under the USMCA/CUSMA environmental cooperation framework alleging weak enforcement of forest and water laws [T1:W4].\n- Certification and traceability programs are becoming more important in state and export governance; Michoacán reported certification mechanisms excluding orchards deforested after January 2018 and properties affected by fires after January 2012 [T1:W23].\n- Local governance may be strengthened where communities use avocado revenues to support forest monitoring and water committees, as reported in community-based models in Zitácuaro [T1:W24].\n- In other areas, governance may be weakened by illegal logging, unauthorized water capture, violence, and organized-crime control [T1:W18, T1:W19, T1:W20].\n\n**Ecological / Biological**\n- Deforestation and forest fragmentation are major effects. One peer-reviewed study estimated that about 20% of Michoacán deforestation from 2001 to 2017 was associated with avocado plantation expansion [T1:W6].\n- Biodiversity risks arise where orchards overlap with Key Biodiversity Areas or expand near the Monarch Butterfly Biosphere Reserve [T1:W6, T1:W14, T1:W17].\n- Forest fires may be associated with conversion pathways; one study identified avocado orchards as a consistent driver of forest fires across remnant forest patches in the Michoacán Avocado Belt [T1:W15].\n- Agrochemical use may affect aquatic ecosystems and non-target organisms; pesticide residues were detected in water samples in the eastern Avocado Belt [T1:W14].\n\n**Technological / Infrastructural**\n- Expansion stimulates packing, transport, cold-chain, irrigation, and monitoring infrastructure.\n- Traceability and remote-sensing tools may improve detection of deforestation and support deforestation-free sourcing [T1:W6, T1:W23].\n- Irrigation infrastructure, including unlicensed intakes and holding ponds, can intensify conflict over local water access [T1:W18].\n\n**Cultural / Social / Demographic**\n- Employment and income can reduce poverty and out-migration in some regions, but benefits are uneven and may reinforce inequality [T1:W7].\n- Indigenous and rural communities may experience weakened cohesion when land-use change, privatization, external investors, or criminal pressure undermine communal institutions [T1:W11, T1:W19].\n- Environmental defenders and community leaders may face threats or violence in contexts where illegal logging and avocado expansion overlap [T1:W19].\n- Public health concerns may arise from pesticide exposure, water scarcity, and insecurity [T1:W14, T1:W18].\n\n**Hydrological**\n- Avocado irrigation can intensify water scarcity, reduce streamflow, deplete legal water concessions, and create water-rights conflicts [T1:W12, T1:W13, T1:W18].\n- Forest conversion can reduce water filtration, aquifer recharge, and watershed regulation services, as alleged in the CEC submission and described in forest-service claims [T1:W4].\n- Water impacts are spatially variable: warmer or drier municipalities may have much higher water footprints than rainfed or cooler regions [T1:W12, T1:W13].\n\n**Climatic / Atmospheric**\n- Conversion of pine-oak forests to orchards reduces aboveground carbon storage [T1:W16].\n- Deforestation, fires, fertilizer use, transport, refrigeration, and export logistics can contribute to greenhouse-gas emissions, though the provided sources are stronger on land-carbon effects than full life-cycle emissions.\n- Climate change may increase future production suitability in some areas while worsening water stress in others, creating uneven risks.\n\n**Geological / Geomorphological**\n- Soil-health degradation, erosion, and nutrient imbalance can result from land conversion and intensive orchard management [T1:W7, T1:W16].\n- Over-fertilization may alter soil nutrient dynamics and complicate assessment of orchard sustainability [T1:W16].\n\n---\n\n### 3. Pericoupling Analysis — adjacent systems\n\n#### 3.1 Systems Identification\n\n**Sending System (adjacent)**: Mexico, especially Michoacán and Jalisco\n- **Human subsystem**: Growers, farmworkers, packers, exporters, SENASICA, state governments, environmental and water agencies, local communities, agribusinesses, transport firms, and certification programs.\n- **Natural subsystem**: Avocado orchards, pine-oak forests, watersheds, soils, biodiversity areas, surface water, groundwater, and forest carbon stocks.\n- **Geographic scope**: Mexico’s avocado-producing and export-certified regions, especially Michoacán and Jalisco.\n\n**Receiving System (adjacent)**: United States\n- **Human subsystem**: Consumers, retailers, restaurants, importers, wholesalers, marketing boards, USDA APHIS, food-service firms, sustainability teams, and trade-policy actors.\n- **Natural subsystem**: U.S. food-consumption landscapes, domestic agricultural land that may be partially spared or reallocated because avocados are imported, and environmental burdens embedded in imported food consumption.\n- **Geographic scope**: United States national market; major urban consumer regions; import and distribution networks.\n\n#### 3.2 Flows Analysis\n\n**Matter Flow**\n- **Direction**: Mexico → United States\n- **Description**: Fresh or dried avocados move from Mexican production regions to U.S. consumers, retailers, and food-service markets. The U.S. receives about 80% of Mexican avocado exports by volume [T1:W1], and Data México reports US$3.444 billion in 2024 exports to the U.S. [T1:W2].\n\n**Capital Flow**\n- **Direction**: United States → Mexico\n- **Description**: Payments for avocado imports flow from U.S. importers, retailers, food-service firms, and consumers to Mexican exporters, packers, growers, and intermediaries. These revenues create incentives for orchard expansion and intensification.\n\n**Information Flow**\n- **Direction**: United States ↔ Mexico\n- **Description**: Market signals, prices, retailer sourcing requirements, phytosanitary rules, sustainability expectations, and deforestation-free certification requirements move between the U.S. and Mexican avocado systems. USDA notes that only Michoacán and Jalisco are eligible for U.S. export under the binational program [T1:W1].\n\n**Energy Flow**\n- **Direction**: Mexico → United States, embodied in traded avocados and logistics\n- **Description**: Energy is embodied in irrigation, fertilizer, harvesting, packing, refrigeration, and transport required to supply the U.S. market. Direct energy accounting is not provided in the sources, but it is part of the export logistics system.\n\n**People Flow**\n- **Direction**: Mexico ↔ United States, indirectly through trade governance and inspection systems\n- **Description**: Inspectors, trade officials, company representatives, auditors, technical consultants, and supply-chain managers move or coordinate across the Mexico–U.S. boundary. This flow is less central than matter and capital flows but supports export eligibility and compliance.\n\n#### 3.3 Agents\n\n- [Individuals / households] **U.S. consumers**: Drive demand through avocado consumption, including demand associated with healthy diets, “superfood” narratives, restaurants, and retail purchases [T1:3].\n- [Individuals / households] **Mexican growers and farmworkers**: Produce and harvest avocados for the U.S. market; experience local benefits and costs.\n- [Firms / traders / corporations] **U.S. retailers, importers, restaurants, and wholesalers**: Purchase, market, distribute, and sell Mexican avocados; can influence sourcing standards and traceability.\n- [Firms / traders / corporations] **Mexican packers and exporters**: Aggregate, certify, pack, and ship fruit to the U.S. market.\n- [Governments / policymakers] **SENASICA and USDA APHIS**: Operate the binational export program that determines export eligibility and phytosanitary compliance [T1:W1].\n- [Governments / policymakers] **Trade and environmental agencies under USMCA/CUSMA**: Shape the institutional context for environmental enforcement concerns, including the CEC submission on Michoacán avocado production [T1:W4].\n- [Organizations / NGOs] **CEC, Climate Rights International, Global Forest Watch / WRI, academic researchers**: Document and communicate sustainability impacts, deforestation risks, and governance gaps [T1:W4, T1:W5, T1:W6, T1:W17].\n- [Non-human agents] **Avocado trees and pests**: Mediate water demand, pesticide use, orchard permanence, and phytosanitary concerns.\n\n#### 3.4 Causes\n\n**Economic**\n- U.S. demand is the dominant adjacent-system driver of Mexican avocado exports, with the U.S. receiving about 80% of export volume [T1:W1].\n- High U.S. market value creates strong incentives for Mexican growers and packers to expand production and comply with export requirements.\n- U.S. consumption growth has been linked to land-use pressure in Michoacán; one study notes U.S. per-capita avocado consumption more than tripled between 2001 and 2017 [T1:W6].\n\n**Political / Institutional**\n- Trade liberalization, including the NAFTA-era opening and end of U.S. restrictions on Mexican avocado imports, helped increase Mexico–U.S. avocado flows [T1:W8].\n- SENASICA–USDA APHIS phytosanitary governance structures determine which regions can export to the U.S. market [T1:W1].\n- The USMCA/CUSMA environmental cooperation framework provides a venue for allegations that Mexico is not enforcing environmental laws related to avocado expansion [T1:W4].\n\n**Ecological / Biological**\n- Avocado suitability in forested highland regions makes export-driven expansion ecologically consequential.\n- Pest and disease management requirements affect phytosanitary rules, pesticide use, and trade eligibility.\n\n**Technological / Infrastructural**\n- Cold-chain logistics, highways, packing houses, border-crossing systems, and inspection infrastructure enable high-volume fresh-fruit trade.\n- Remote sensing and supply-chain reconstruction methods can link U.S. consumption to Mexican deforestation risk [T1:W6].\n\n**Cultural / Social / Demographic**\n- U.S. dietary trends and avocado marketing as a healthy food or “superfood” increase consumer demand [T1:3, T1:5].\n- Restaurant, sports-event, and retail cultures normalize high avocado consumption, indirectly shaping land-use incentives in Mexican producing regions.\n\n**Hydrological**\n- U.S. demand indirectly increases irrigation demand in Mexican producing regions, especially where growers intensify production to meet export markets.\n- Drought and local water scarcity in Michoacán make the water burden of export production socially contentious [T1:W12, T1:W18].\n\n**Climatic / Atmospheric**\n- Climate-related drought can amplify water conflicts in Mexico while demand from the adjacent U.S. market remains high.\n- Refrigerated transport and long supply chains may contribute to emissions, though direct life-cycle estimates are not provided in the retrieved evidence.\n\n**Geological / Geomorphological**\n- Suitable soils, terrain, and elevation in Michoacán and Jalisco shape where U.S.-oriented production expands [T1:W8, T1:W9].\n- Soil degradation from intensification may reduce long-term productivity and increase management costs.\n\n#### 3.5 Effects\n\n**Economic**\n- **Sending system — Mexico**: Receives substantial export revenue and employment benefits; Michoacán dominates export value [T1:W2].\n- **Receiving system — United States**: Gains reliable year-round avocado supply for consumers, retailers, restaurants, and food-service companies.\n- **Sending system — Mexico**: Profit concentration, extortion, and unequal value capture can reduce local sustainability benefits [T1:4, T1:W21, T1:W22].\n\n**Political / Institutional**\n- **Sending system — Mexico**: U.S. market access creates pressure for phytosanitary compliance, traceability, and deforestation-free certification [T1:W1, T1:W23].\n- **Receiving system — United States**: Retailers and regulators face growing pressure to address imported deforestation and human-rights risks.\n- **Both systems**: The CEC submission makes avocado sustainability a cross-border environmental governance issue [T1:W4].\n\n**Ecological / Biological**\n- **Sending system — Mexico**: U.S.-linked demand contributes to orchard expansion, deforestation, forest fragmentation, and biodiversity risks [T1:W6, T1:W17].\n- **Receiving system — United States**: Environmental costs of avocado consumption are partly externalized to Mexican production landscapes, consistent with telecoupled ecosystem-service burden arguments [T1:5].\n- **Sending system — Mexico**: Pesticide use and agrochemical runoff can affect aquatic ecosystems and biodiversity near production areas [T1:W14].\n\n**Technological / Infrastructural**\n- **Sending system — Mexico**: Export trade stimulates packing-house development, transport infrastructure, irrigation infrastructure, and certification systems.\n- **Receiving system — United States**: Import logistics and retail distribution networks expand to support consumer demand.\n\n**Cultural / Social / Demographic**\n- **Sending system — Mexico**: Employment can reduce out-migration and poverty in some areas, but inequality, seasonal work, and violence reduce social sustainability [T1:W7, T1:W20].\n- **Receiving system — United States**: Consumers benefit from dietary diversity and convenient access, but may remain unaware of environmental and social burdens embedded in supply chains [T1:5, T1:W15].\n- **Sending system — Mexico**: Rural and Indigenous communities may bear water scarcity, land pressure, and security risks.\n\n**Hydrological**\n- **Sending system — Mexico**: Export-driven production increases pressure on springs, streams, groundwater, and legal water concessions [T1:W12, T1:W13, T1:W18].\n- **Receiving system — United States**: U.S. consumers receive avocados while much of the water burden remains embedded in Mexican production landscapes.\n\n**Climatic / Atmospheric**\n- **Sending system — Mexico**: Forest conversion reduces aboveground carbon storage [T1:W16].\n- **Receiving system — United States**: Consumption is linked to embodied land-use emissions and refrigerated supply-chain energy use, though direct quantified emissions are not available in the provided evidence.\n\n**Geological / Geomorphological**\n- **Sending system — Mexico**: Soil disturbance, erosion, and over-fertilization may follow intensive orchard expansion [T1:W7, T1:W16].\n- **Receiving system — United States**: Soil impacts are indirect and embodied in imported avocado consumption.\n\n---\n\n### 4. Telecoupling Analysis — distant systems\n\n#### 4.1 Systems Identification\n\n**Sending System (distant)**: Mexico, especially Michoacán and Jalisco\n- **Human subsystem**: Avocado producers, laborers, packers, exporters, certification bodies, state and federal agencies, communities, and intermediaries.\n- **Natural subsystem**: Forests, orchards, water resources, soils, biodiversity areas, and carbon stocks.\n- **Geographic scope**: Mexican avocado-producing regions supplying distant national and urban markets.\n\n**Receiving System (distant)**: Canada\n- **Human subsystem**: Consumers, importers, retailers, restaurants, food distributors, and trade agencies.\n- **Natural subsystem**: Canadian consumption landscapes and domestic food systems that receive imported avocados rather than producing them locally at scale.\n- **Geographic scope**: Canada; Data México reports US$257 million in 2024 Mexican avocado exports to Canada [T1:W2].\n\n**Receiving System (distant)**: Japan\n- **Human subsystem**: Consumers, retailers, food-service firms, importers, and trade regulators.\n- **Natural subsystem**: Japanese food-consumption landscapes and indirect environmental footprints of imported fruit.\n- **Geographic scope**: Japan; Data México reports US$108 million in 2024 Mexican avocado exports to Japan [T1:W2].\n\n**Receiving System (distant)**: El Salvador and Honduras\n- **Human subsystem**: Importers, retailers, consumers, food-service firms, and trade agencies.\n- **Natural subsystem**: Domestic food-consumption landscapes indirectly linked to Mexican production impacts.\n- **Geographic scope**: Central American receiving markets; Data México reports US$38 million in 2024 exports to El Salvador and US$31.6 million to Honduras [T1:W2].\n\n**Receiving System (distant)**: Distant U.S. urban consumer landscapes, treated at subnational scale\n- **Human subsystem**: Urban consumers, restaurants, supermarkets, marketers, importers, and retailers.\n- **Natural subsystem**: Urban food-consumption systems that draw on distant Mexican ecosystem services and externalize water, forest, biodiversity, and carbon burdens.\n- **Geographic scope**: Major U.S. cities and metropolitan markets far from Michoacán and Jalisco. A landscape-sustainability source explicitly identifies U.S. urban consumer landscapes and Michoacán avocado-producing landscapes as telecoupled systems [T1:5].\n\n**Spillover System**: Other avocado-producing and exporting countries, especially Peru and Chile\n- **Human subsystem**: Avocado growers, exporters, irrigation users, local communities, water-rights holders, agribusiness firms, and national trade agencies in competitor countries.\n- **Natural subsystem**: Water-stressed agricultural valleys, biodiversity, forests or semi-arid ecosystems, soils, and watersheds affected by avocado expansion.\n- **Geographic scope**: Chilean production regions such as Valparaíso and Metropolitana, Peruvian avocado regions, and other emerging avocado producers. The literature identifies Mexico, Chile, and Peru as major avocado producers and exporters and calls for further investigation of trade dynamics among them [T1:3, T1:4].\n\n**Spillover System**: Distant timber or forest supply regions\n- **Human subsystem**: Forestry producers, communities, timber markets, conservation agencies, and land managers affected when forest-protection policies in importing regions displace production burdens abroad.\n- **Natural subsystem**: Forest ecosystems, biodiversity, carbon stocks, and regulation services in regions indirectly affected by global commodity demand and forest policy displacement.\n- **Geographic scope**: Not precisely identified in the provided evidence; included because one landscape-sustainability synthesis names distant timber or forest supply regions as part of the telecoupled burden structure [T1:5].\n\n#### 4.2 Flows Analysis\n\n**Matter Flow**\n- **Direction**: Mexico → Canada, Japan, El Salvador, Honduras, and distant U.S. urban markets\n- **Description**: Fresh or dried avocados flow from Mexican producing regions to distant consumer systems. Data México documents 2024 export values to Canada, Japan, El Salvador, and Honduras, while USDA identifies Canada and Japan as leading destinations after the U.S. [T1:W1, T1:W2].\n\n**Capital Flow**\n- **Direction**: Canada, Japan, El Salvador, Honduras, and distant U.S. urban markets → Mexico\n- **Description**: Payments for imported avocados flow back to Mexican exporters, packers, growers, logistics firms, and intermediaries. These capital flows reinforce orchard expansion and intensification incentives.\n\n**Information Flow**\n- **Direction**: Distant consumer markets, retailers, marketers, NGOs, and regulators ↔ Mexican production regions\n- **Description**: Consumer preferences, price signals, health narratives, sustainability concerns, certification requirements, NGO reports, and media attention move across distance and shape production decisions. Global demand for avocado as a “superfood” is identified as a key cause of production growth [T1:2, T1:3].\n\n**Energy Flow**\n- **Direction**: Mexico → distant markets, embodied in refrigerated transport and logistics\n- **Description**: Energy is embodied in harvesting, packing, cooling, long-distance transport, retail storage, and waste management. The provided evidence does not quantify this flow but supports the existence of export supply chains.\n\n**Matter Flow — Virtual Water**\n- **Direction**: Mexican watersheds → distant consumer markets, embodied in exported avocados\n- **Description**: Water used in irrigation and crop evapotranspiration is embodied in exported avocados. Studies show high water footprints in Michoacán, especially for irrigated orchards [T1:W12, T1:W13].\n\n**Matter Flow — Embodied Land and Carbon**\n- **Direction**: Mexican forests / soils / carbon stocks → distant consumer demand, embodied in avocado imports\n- **Description**: Land conversion, forest-carbon loss, and soil impacts are embedded in avocado supply chains. Forests store more aboveground carbon than avocado orchards [T1:W16], and deforestation has been linked to avocado expansion [T1:W6].\n\n**Information Flow**\n- **Direction**: Mexico, Chile, Peru, and other producers ↔ global avocado markets\n- **Description**: Competitive market signals, export standards, prices, and lessons about sustainability practices circulate among producer countries. Ortiz et al. argue that trade dynamics among Mexico, Chile, and Peru should be investigated further for safeguards for local producers [T1:4].\n\n#### 4.3 Agents\n\n- [Individuals / households] **Consumers in Canada, Japan, El Salvador, Honduras, and distant U.S. cities**: Generate demand for imported avocados and respond to marketing, price, health narratives, and sustainability information.\n- [Individuals / households] **Mexican growers, farmworkers, and rural households**: Bear many local production impacts while supplying distant demand.\n- [Firms / traders / corporations] **International retailers, importers, wholesalers, restaurants, and marketers**: Shape product standards, volumes, prices, branding, and consumer access.\n- [Firms / traders / corporations] **Mexican exporters and packing firms**: Connect orchards to distant markets and mediate compliance with export and quality standards.\n- [Firms / traders / corporations] **Competitor-country agribusinesses in Peru and Chile**: Respond to global demand and market competition, potentially expanding production in their own water-stressed or ecologically sensitive regions.\n- [Governments / policymakers] **Mexican federal and state agencies**: Regulate land use, water concessions, forest protection, phytosanitary access, and export certification.\n- [Governments / policymakers] **Importing-country trade and food-safety agencies**: Set import rules and may increase scrutiny of deforestation, water, or human-rights risks.\n- [Organizations / NGOs] **Academic researchers, NGOs, CEC, Global Forest Watch / WRI, Climate Rights International**: Produce knowledge and political pressure around telecoupled sustainability burdens [T1:W4, T1:W5, T1:W6, T1:W17].\n- [Non-human agents] **Avocado trees**: Mediate water, land, carbon, and agrochemical effects through perennial orchard systems.\n- [Non-human agents] **Forest species, pollinators, pests, pathogens, and monarch butterflies**: Shape biodiversity stakes, pesticide decisions, and conservation concerns.\n\n#### 4.4 Causes\n\n**Economic**\n- Global demand for avocados has risen substantially over the past two decades, driven partly by “superfood” narratives and Western dietary trends [T1:3].\n- Mexico’s export dominance and high prices create incentives for expansion, intensification, and market competition with Chile and Peru [T1:3, T1:4].\n- Retail and food-service demand in distant markets links consumer purchasing to land-use decisions in Mexican producing regions [T1:5].\n\n**Political / Institutional**\n- Economic globalization and free trade agreements promoted the shift from domestic production toward international markets [T1:3].\n- Export-oriented agribusiness policies and trade institutions create conditions for distant market demand to shape local land-use decisions [T1:2].\n- Deforestation-free certification and traceability initiatives are emerging partly because distant markets, NGOs, and trade institutions are scrutinizing avocado supply chains [T1:W23].\n\n**Ecological / Biological**\n- Avocado’s high water demand makes production especially consequential in water-stressed regions and during drought. This concern is documented for Chile and Mexico [T1:3, T1:W12, T1:W13].\n- Forest and biodiversity overlap with suitable avocado-growing zones increases ecological sensitivity [T1:W6, T1:W17].\n- Competition among producer countries may shift expansion into new ecological frontiers.\n\n**Technological / Infrastructural**\n- Global cold chains, shipping, highways, packing systems, phytosanitary inspection, and export certification enable long-distance fresh-fruit trade.\n- Remote sensing, machine learning, and supply-chain reconstruction allow researchers to trace distant consumption to local deforestation risk [T1:W6].\n- Irrigation technology enables production in drier regions but can intensify water stress.\n\n**Cultural / Social / Demographic**\n- Avocado consumption is promoted by health, lifestyle, and “superfood” narratives in distant consumer systems [T1:3].\n- Consumer awareness campaigns and sustainability branding may reshape demand, but can also mask offstage environmental burdens [T1:5].\n- Rural livelihood expectations in producing systems encourage participation in avocado markets despite social risks.\n\n**Hydrological**\n- Water scarcity and drought interact with export-oriented production to create conflicts, especially where irrigation demand exceeds legal water concessions [T1:W12, T1:W13].\n- Chilean evidence shows how international avocado demand can worsen water injustice under neoliberal water policies and climate-induced water stress, offering a spillover comparison for Mexico [T1:1, T1:3].\n\n**Climatic / Atmospheric**\n- Climate change may alter where avocado cultivation is viable, potentially expanding suitability in some areas and increasing water stress in others [T1:6, T1:W10].\n- Forest loss reduces carbon storage and may create climate feedbacks [T1:W16].\n\n**Geological / Geomorphological**\n- Soil, elevation, slope, and terrain suitability influence where avocado frontiers expand [T1:W8, T1:W9].\n- As optimal soils become scarce, expansion pressure may move into less suitable or more fragile landscapes.\n\n#### 4.5 Effects\n\n**Economic**\n- **Sending system — Mexico**: Export revenues are large and globally significant; Mexico was the world’s leading fresh or dried avocado exporter in 2022, and exports totaled US$3.969 billion in 2024 [T1:W2].\n- **Receiving systems — Canada, Japan, El Salvador, Honduras, distant U.S. cities**: Consumers and firms gain access to imported avocados.\n- **Spillover systems — Chile and Peru**: Competitive global demand may encourage expansion and intensification in other producer countries; Ortiz et al. identify Mexico, Chile, and Peru as key producers whose trade dynamics require further study [T1:4].\n- **Sending system — Mexico**: Local economic benefits may be undermined by inequality, precarious labor, and organized-crime rent extraction [T1:W7, T1:W21, T1:W22].\n\n**Political / Institutional**\n- **Sending system — Mexico**: Sustainability concerns create pressure for stronger enforcement of forest, water, and land-use laws [T1:W4].\n- **Receiving systems**: Retailers, governments, and consumers may face pressure to adopt deforestation-free sourcing and supply-chain transparency.\n- **Spillover systems — Chile and Peru**: Lessons from Mexico may inform safeguards, but competition may also intensify environmental pressure if markets reward volume without adequate regulation [T1:4].\n\n**Ecological / Biological**\n- **Sending system — Mexico**: Deforestation, fragmentation, biodiversity loss, fire-related forest conversion, and agrochemical contamination are central sustainability effects [T1:W6, T1:W14, T1:W15, T1:W17].\n- **Receiving systems**: Ecological impacts are mostly embodied and displaced to producing landscapes rather than occurring directly in consumer regions [T1:5].\n- **Spillover systems — Chile**: International demand and trade are linked to pressures on water resources, biodiversity, and local communities [T1:1].\n- **Spillover systems — distant forest regions**: Sustainability gains in importing regions may mask burdens shifted to other production landscapes [T1:5].\n\n**Technological / Infrastructural**\n- **Sending system — Mexico**: Export demand increases packing, refrigeration, irrigation, traceability, and road infrastructure.\n- **Receiving systems**: Import logistics, ripening facilities, supermarket distribution, and food-service supply chains expand.\n- **Spillover systems**: Competitor countries may invest in similar export infrastructure to capture global avocado demand.\n\n**Cultural / Social / Demographic**\n- **Sending system — Mexico**: Communities experience mixed effects: employment and reduced out-migration in some places, but inequality, insecurity, and weakened cohesion in others [T1:W7, T1:W11].\n- **Receiving systems**: Consumers benefit from dietary options but may remain disconnected from environmental and human-rights burdens in producing regions [T1:5].\n- **Spillover systems — Chile**: Water injustice and socioecological conflict can emerge where export-oriented avocado expansion competes with local water needs [T1:1, T1:3].\n\n**Hydrological**\n- **Sending system — Mexico**: Virtual water is exported through avocados while water scarcity remains local; studies show irrigated avocado production can exceed legal agricultural water concessions in some settings [T1:W12, T1:W13].\n- **Receiving systems**: Distant consumers import water-intensive fruit without directly experiencing the water stress embedded in production.\n- **Spillover systems — Chile**: International demand has been linked to severe water scarcity, community reliance on cistern trucks, and socioecological conflict in Petorca and Central Chile [T1:3].\n\n**Climatic / Atmospheric**\n- **Sending system — Mexico**: Forest conversion reduces aboveground carbon storage [T1:W16].\n- **Receiving systems**: Embodied emissions and carbon losses are associated with consumption, though direct consumer-region impacts are indirect.\n- **Spillover systems**: If production expands in Chile, Peru, or other countries, land-carbon and water-stress impacts may be redistributed across the global avocado frontier.\n\n**Geological / Geomorphological**\n- **Sending system — Mexico**: Soil degradation, erosion, and nutrient imbalance may follow intensive orchard management [T1:W7, T1:W16].\n- **Spillover systems**: Expansion into less suitable soils or steep terrain in other producer countries may create erosion and land-degradation risks.\n\n---\n\n### 5. Cross-coupling Interactions\n\n- **Amplification between intracoupling, pericoupling, and telecoupling**: U.S. and global demand amplify within-Mexico land-use decisions by increasing the profitability of avocado orchards relative to forests, subsistence crops, or mixed farming. Capital and information from adjacent and distant markets strengthen local processes such as orchard expansion, irrigation investment, packing-house development, and certification uptake. The strongest evidence is for U.S.-linked demand and Michoacán deforestation, including the finding that about 20% of Michoacán deforestation from 2001 to 2017 was associated with avocado plantation expansion [T1:W6].\n\n- **Spatial tradeoffs**: Receiving systems gain food access, retail profits, and consumer benefits, while many ecological and social costs remain in Mexican producing regions. These costs include deforestation, water scarcity, forest-carbon loss, agrochemical contamination, and risks to rural and Indigenous communities [T1:W5, T1:W7, T1:W12, T1:W14, T1:W16]. This is consistent with the landscape-sustainability framing that U.S. urban consumption can mask offstage environmental burdens in Michoacán [T1:5].\n\n- **Displacement**: If certification, enforcement, water scarcity, or reputational risk constrain production in parts of Michoacán, avocado expansion may shift to Jalisco, other Mexican states, or competitor countries such as Peru and Chile. Evidence already shows Jalisco’s growing role in Mexican production and exports [T1:W1, T1:W2, T1:W17], while Ortiz et al. identify Mexico, Chile, and Peru as interconnected avocado producers whose trade dynamics need further investigation [T1:4].\n\n- **Feedback loops**: Environmental and social effects in Mexico are beginning to feed back into governance and market flows. Deforestation, water conflict, and NGO or academic documentation have contributed to CEC scrutiny, retailer attention, and deforestation-free certification efforts [T1:W4, T1:W5, T1:W6, T1:W23]. If certification becomes credible and enforceable, it could reduce flows from recently deforested orchards; if it remains weak or partial, export demand may continue reinforcing land conversion and water capture.\n\n- **Coupling transformations**: Mexican avocado production moved from more local and domestic coupling toward strong pericoupling and telecoupling as trade liberalization, U.S. market access, and global demand expanded export chains [T1:2, T1:3, T1:W8]. Potential decoupling could occur through drought, phytosanitary suspension, violence, consumer backlash, or import restrictions. Recoupling could occur through traceability, credible deforestation-free certification, water-footprint verification, community governance models, or geographic shifts to new producing regions.\n\n---\n\n### 6. Research Gaps and Suggestions\n\n- **Quantify complete flows across coupling types**: Current evidence is strong for export volume/value and deforestation, but a full metacoupling study should quantify matter, capital, virtual water, embodied carbon, energy, pesticides, and labor flows from specific municipalities to specific receiving markets.\n\n- **Improve subnational traceability**: More work is needed to connect individual orchards, municipalities, packing houses, export destinations, retailers, and environmental outcomes. TRACAST-style supply-chain reconstruction and remote sensing provide a strong model for this [T1:W6].\n\n- **Assess certification credibility and outcomes**: Michoacán’s deforestation-free certification claims should be independently audited to determine whether certified supply is actually free from post-2018 deforestation, fire-related conversion, illegal water use, and protected-area encroachment [T1:W23].\n\n- **Compare Michoacán, Jalisco, and emerging production frontiers**: Research should test whether sustainability pressures are being reduced or displaced as Jalisco and other regions increase production and export eligibility.\n\n- **Analyze water governance at watershed scale**: Water-footprint studies should be integrated with hydrological monitoring, water concessions, illegal intakes, streamflow data, and household water access to identify who gains and who loses from export-oriented irrigation [T1:W12, T1:W13, T1:W18].\n\n- **Include organized crime and violence in sustainability metrics**: Economic and environmental studies should incorporate extortion, illegal logging, threats to environmental defenders, and homicide risks where organized-crime presence intersects with avocado export value [T1:W19, T1:W20, T1:W21, T1:W22].\n\n- **Study spillover systems more rigorously**: Mexico’s avocado sustainability cannot be assessed alone. Comparative work with Chile and Peru is needed to determine whether market demand, certification, or environmental restrictions in one producing country shift water, land, and biodiversity pressures elsewhere [T1:1, T1:4].\n\n- **Move toward flow-based governance**: Governance should not focus only on orchards or municipalities; it should track flows of avocados, capital, water, carbon, pesticides, information, and responsibility across producers, packers, retailers, consumers, and regulators.\n\n---\n\n### 7. Evidence Coverage\n\nStrong evidence base: The trade structure is well supported by government sources showing Mexico’s production scale, export value, and destination concentration, especially the dominance of the U.S. market [T1:W1, T1:W2]. The environmental-impact claims are also well supported by peer-reviewed and NGO sources linking avocado expansion to deforestation, forest fragmentation, water scarcity, agrochemical risks, and carbon loss in Michoacán and Jalisco [T1:W5, T1:W6, T1:W7, T1:W12, T1:W14, T1:W16, T1:W17].\n\nModerate evidence: Social and governance effects are supported by a mix of peer-reviewed studies, NGO reports, government communications, and news reporting. The strongest systematic evidence concerns organized crime and homicide risk where export value increases under criminal presence [T1:W21, T1:W22]. Water conflict and environmental-defender risks are documented through AP reporting and should be paired with more local fieldwork and official records [T1:W18, T1:W19].\n\nTelecoupling evidence is conceptually strong. The retrieved literature explicitly frames avocado production and trade as telecoupled, with international demand shaping local production and sustainability burdens [T1:1, T1:3, T1:5]. Mexico-specific telecoupling to U.S. consumption is directly supported by the Journal of Environmental Management study and landscape-sustainability synthesis [T1:W6, T1:5].\n\nLimited evidence: Energy flows, full life-cycle greenhouse-gas emissions, detailed labor migration, and exact capital distribution among growers, packers, retailers, and criminal actors are inferred from the structure of the avocado supply chain rather than directly quantified in the provided sources. Spillover effects on Peru, Chile, and distant forest-supply regions are plausible and partly supported by Ortiz et al. and Dade et al., but require more direct comparative trade and land-use evidence [T1:1, T1:4, T1:5].",
  "research_gaps": [
    "**Quantify complete flows across coupling types**: Current evidence is strong for export volume/value and deforestation, but a full metacoupling study should quantify matter, capital, virtual water, embodied carbon, energy, pesticides, and labor flows from specific municipalities to specific receiving markets.",
    "**Improve subnational traceability**: More work is needed to connect individual orchards, municipalities, packing houses, export destinations, retailers, and environmental outcomes. TRACAST-style supply-chain reconstruction and remote sensing provide a strong model for this [T1:W6].",
    "**Assess certification credibility and outcomes**: Michoacán’s deforestation-free certification claims should be independently audited to determine whether certified supply is actually free from post-2018 deforestation, fire-related conversion, illegal water use, and protected-area encroachment [T1:W23].",
    "**Compare Michoacán, Jalisco, and emerging production frontiers**: Research should test whether sustainability pressures are being reduced or displaced as Jalisco and other regions increase production and export eligibility.",
    "**Analyze water governance at watershed scale**: Water-footprint studies should be integrated with hydrological monitoring, water concessions, illegal intakes, streamflow data, and household water access to identify who gains and who loses from export-oriented irrigation [T1:W12, T1:W13, T1:W18].",
    "**Include organized crime and violence in sustainability metrics**: Economic and environmental studies should incorporate extortion, illegal logging, threats to environmental defenders, and homicide risks where organized-crime presence intersects with avocado export value [T1:W19, T1:W20, T1:W21, T1:W22].",
    "**Study spillover systems more rigorously**: Mexico’s avocado sustainability cannot be assessed alone. Comparative work with Chile and Peru is needed to determine whether market demand, certification, or environmental restrictions in one producing country shift water, land, and biodiversity pressures elsewhere [T1:1, T1:4].",
    "**Move toward flow-based governance**: Governance should not focus only on orchards or municipalities; it should track flows of avocados, capital, water, carbon, pesticides, information, and responsibility across producers, packers, retailers, consumers, and regulators."
  ],
  "telecoupling": {
    "systems": [
      {
        "role": "sending",
        "system_scope": "distant",
        "name": "Mexico, especially Michoacán and Jalisco",
        "human_subsystem": "Avocado producers, laborers, packers, exporters, certification bodies, state and federal agencies, communities, and intermediaries.",
        "natural_subsystem": "Forests, orchards, water resources, soils, biodiversity areas, and carbon stocks.",
        "geographic_scope": "Mexican avocado-producing regions supplying distant national and urban markets."
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "Canada",
        "human_subsystem": "Consumers, importers, retailers, restaurants, food distributors, and trade agencies.",
        "natural_subsystem": "Canadian consumption landscapes and domestic food systems that receive imported avocados rather than producing them locally at scale.",
        "geographic_scope": "Canada; Data México reports US$257 million in 2024 Mexican avocado exports to Canada [T1:W2]."
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "Japan",
        "human_subsystem": "Consumers, retailers, food-service firms, importers, and trade regulators.",
        "natural_subsystem": "Japanese food-consumption landscapes and indirect environmental footprints of imported fruit.",
        "geographic_scope": "Japan; Data México reports US$108 million in 2024 Mexican avocado exports to Japan [T1:W2]."
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "El Salvador and Honduras",
        "human_subsystem": "Importers, retailers, consumers, food-service firms, and trade agencies.",
        "natural_subsystem": "Domestic food-consumption landscapes indirectly linked to Mexican production impacts.",
        "geographic_scope": "Central American receiving markets; Data México reports US$38 million in 2024 exports to El Salvador and US$31.6 million to Honduras [T1:W2]."
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "Distant U.S. urban consumer landscapes, treated at subnational scale",
        "human_subsystem": "Urban consumers, restaurants, supermarkets, marketers, importers, and retailers.",
        "natural_subsystem": "Urban food-consumption systems that draw on distant Mexican ecosystem services and externalize water, forest, biodiversity, and carbon burdens.",
        "geographic_scope": "Major U.S. cities and metropolitan markets far from Michoacán and Jalisco. A landscape-sustainability source explicitly identifies U.S. urban consumer landscapes and Michoacán avocado-producing landscapes as telecoupled systems [T1:5]."
      },
      {
        "role": "spillover",
        "name": "Other avocado-producing and exporting countries, especially Peru and Chile",
        "human_subsystem": "Avocado growers, exporters, irrigation users, local communities, water-rights holders, agribusiness firms, and national trade agencies in competitor countries.",
        "natural_subsystem": "Water-stressed agricultural valleys, biodiversity, forests or semi-arid ecosystems, soils, and watersheds affected by avocado expansion.",
        "geographic_scope": "Chilean production regions such as Valparaíso and Metropolitana, Peruvian avocado regions, and other emerging avocado producers. The literature identifies Mexico, Chile, and Peru as major avocado producers and exporters and calls for further investigation of trade dynamics among them [T1:3, T1:4]."
      },
      {
        "role": "spillover",
        "name": "Distant timber or forest supply regions",
        "human_subsystem": "Forestry producers, communities, timber markets, conservation agencies, and land managers affected when forest-protection policies in importing regions displace production burdens abroad.",
        "natural_subsystem": "Forest ecosystems, biodiversity, carbon stocks, and regulation services in regions indirectly affected by global commodity demand and forest policy displacement.",
        "geographic_scope": "Not precisely identified in the provided evidence; included because one landscape-sustainability synthesis names distant timber or forest supply regions as part of the telecoupled burden structure [T1:5]."
      }
    ],
    "flows": [
      {
        "category": "matter",
        "direction": "Mexico → Canada, Japan, El Salvador, Honduras, and distant U.S. urban markets",
        "description": "Fresh or dried avocados flow from Mexican producing regions to distant consumer systems. Data México documents 2024 export values to Canada, Japan, El Salvador, and Honduras, while USDA identifies Canada and Japan as leading destinations after the U.S. [T1:W1, T1:W2]."
      },
      {
        "category": "capital",
        "direction": "Canada, Japan, El Salvador, Honduras, and distant U.S. urban markets → Mexico",
        "description": "Payments for imported avocados flow back to Mexican exporters, packers, growers, logistics firms, and intermediaries. These capital flows reinforce orchard expansion and intensification incentives."
      },
      {
        "category": "information",
        "direction": "Distant consumer markets, retailers, marketers, NGOs, and regulators ↔ Mexican production regions",
        "description": "Consumer preferences, price signals, health narratives, sustainability concerns, certification requirements, NGO reports, and media attention move across distance and shape production decisions. Global demand for avocado as a “superfood” is identified as a key cause of production growth [T1:2, T1:3]."
      },
      {
        "category": "energy",
        "direction": "Mexican forests / soils / carbon stocks → distant consumer demand, embodied in avocado imports",
        "description": "Energy is embodied in harvesting, packing, cooling, long-distance transport, retail storage, and waste management. The provided evidence does not quantify this flow but supports the existence of export supply chains. **Matter Flow — Virtual Water** Water used in irrigation and crop evapotranspiration is embodied in exported avocados. Studies show high water footprints in Michoacán, especially for irrigated orchards [T1:W12, T1:W13]. **Matter Flow — Embodied Land and Carbon** Land conversion, forest-carbon loss, and soil impacts are embedded in avocado supply chains. Forests store more aboveground carbon than avocado orchards [T1:W16], and deforestation has been linked to avocado expansion [T1:W6]."
      },
      {
        "category": "information",
        "direction": "Mexico, Chile, Peru, and other producers ↔ global avocado markets",
        "description": "Competitive market signals, export standards, prices, and lessons about sustainability practices circulate among producer countries. Ortiz et al. argue that trade dynamics among Mexico, Chile, and Peru should be investigated further for safeguards for local producers [T1:4]."
      }
    ],
    "agents": [
      {
        "level": "individuals / households",
        "name": "**Consumers in Canada, Japan, El Salvador, Honduras, and distant U.S. cities**",
        "description": "Generate demand for imported avocados and respond to marketing, price, health narratives, and sustainability information."
      },
      {
        "level": "individuals / households",
        "name": "**Mexican growers, farmworkers, and rural households**",
        "description": "Bear many local production impacts while supplying distant demand."
      },
      {
        "level": "firms / traders / corporations",
        "name": "**International retailers, importers, wholesalers, restaurants, and marketers**",
        "description": "Shape product standards, volumes, prices, branding, and consumer access."
      },
      {
        "level": "firms / traders / corporations",
        "name": "**Mexican exporters and packing firms**",
        "description": "Connect orchards to distant markets and mediate compliance with export and quality standards."
      },
      {
        "level": "firms / traders / corporations",
        "name": "**Competitor-country agribusinesses in Peru and Chile**",
        "description": "Respond to global demand and market competition, potentially expanding production in their own water-stressed or ecologically sensitive regions."
      },
      {
        "level": "governments / policymakers",
        "name": "**Mexican federal and state agencies**",
        "description": "Regulate land use, water concessions, forest protection, phytosanitary access, and export certification."
      },
      {
        "level": "governments / policymakers",
        "name": "**Importing-country trade and food-safety agencies**",
        "description": "Set import rules and may increase scrutiny of deforestation, water, or human-rights risks."
      },
      {
        "level": "organizations / NGOs",
        "name": "**Academic researchers, NGOs, CEC, Global Forest Watch / WRI, Climate Rights International**",
        "description": "Produce knowledge and political pressure around telecoupled sustainability burdens [T1:W4, T1:W5, T1:W6, T1:W17]."
      },
      {
        "level": "non-human agents",
        "name": "**Avocado trees**",
        "description": "Mediate water, land, carbon, and agrochemical effects through perennial orchard systems."
      },
      {
        "level": "non-human agents",
        "name": "**Forest species, pollinators, pests, pathogens, and monarch butterflies**",
        "description": "Shape biodiversity stakes, pesticide decisions, and conservation concerns."
      }
    ],
    "causes": {
      "economic": [
        "Global demand for avocados has risen substantially over the past two decades, driven partly by “superfood” narratives and Western dietary trends [T1:3].",
        "Mexico’s export dominance and high prices create incentives for expansion, intensification, and market competition with Chile and Peru [T1:3, T1:4].",
        "Retail and food-service demand in distant markets links consumer purchasing to land-use decisions in Mexican producing regions [T1:5]."
      ],
      "political / institutional": [
        "Economic globalization and free trade agreements promoted the shift from domestic production toward international markets [T1:3].",
        "Export-oriented agribusiness policies and trade institutions create conditions for distant market demand to shape local land-use decisions [T1:2].",
        "Deforestation-free certification and traceability initiatives are emerging partly because distant markets, NGOs, and trade institutions are scrutinizing avocado supply chains [T1:W23]."
      ],
      "ecological / biological": [
        "Avocado’s high water demand makes production especially consequential in water-stressed regions and during drought. This concern is documented for Chile and Mexico [T1:3, T1:W12, T1:W13].",
        "Forest and biodiversity overlap with suitable avocado-growing zones increases ecological sensitivity [T1:W6, T1:W17].",
        "Competition among producer countries may shift expansion into new ecological frontiers."
      ],
      "technological / infrastructural": [
        "Global cold chains, shipping, highways, packing systems, phytosanitary inspection, and export certification enable long-distance fresh-fruit trade.",
        "Remote sensing, machine learning, and supply-chain reconstruction allow researchers to trace distant consumption to local deforestation risk [T1:W6].",
        "Irrigation technology enables production in drier regions but can intensify water stress."
      ],
      "cultural / social / demographic": [
        "Avocado consumption is promoted by health, lifestyle, and “superfood” narratives in distant consumer systems [T1:3].",
        "Consumer awareness campaigns and sustainability branding may reshape demand, but can also mask offstage environmental burdens [T1:5].",
        "Rural livelihood expectations in producing systems encourage participation in avocado markets despite social risks."
      ],
      "hydrological": [
        "Water scarcity and drought interact with export-oriented production to create conflicts, especially where irrigation demand exceeds legal water concessions [T1:W12, T1:W13].",
        "Chilean evidence shows how international avocado demand can worsen water injustice under neoliberal water policies and climate-induced water stress, offering a spillover comparison for Mexico [T1:1, T1:3]."
      ],
      "climatic / atmospheric": [
        "Climate change may alter where avocado cultivation is viable, potentially expanding suitability in some areas and increasing water stress in others [T1:6, T1:W10].",
        "Forest loss reduces carbon storage and may create climate feedbacks [T1:W16]."
      ],
      "geological / geomorphological": [
        "Soil, elevation, slope, and terrain suitability influence where avocado frontiers expand [T1:W8, T1:W9].",
        "As optimal soils become scarce, expansion pressure may move into less suitable or more fragile landscapes."
      ]
    },
    "effects": {
      "economic": [
        "**Sending system — Mexico**: Export revenues are large and globally significant; Mexico was the world’s leading fresh or dried avocado exporter in 2022, and exports totaled US$3.969 billion in 2024 [T1:W2].",
        "**Receiving systems — Canada, Japan, El Salvador, Honduras, distant U.S. cities**: Consumers and firms gain access to imported avocados.",
        "**Spillover systems — Chile and Peru**: Competitive global demand may encourage expansion and intensification in other producer countries; Ortiz et al. identify Mexico, Chile, and Peru as key producers whose trade dynamics require further study [T1:4].",
        "**Sending system — Mexico**: Local economic benefits may be undermined by inequality, precarious labor, and organized-crime rent extraction [T1:W7, T1:W21, T1:W22]."
      ],
      "political / institutional": [
        "**Sending system — Mexico**: Sustainability concerns create pressure for stronger enforcement of forest, water, and land-use laws [T1:W4].",
        "**Receiving systems**: Retailers, governments, and consumers may face pressure to adopt deforestation-free sourcing and supply-chain transparency.",
        "**Spillover systems — Chile and Peru**: Lessons from Mexico may inform safeguards, but competition may also intensify environmental pressure if markets reward volume without adequate regulation [T1:4]."
      ],
      "ecological / biological": [
        "**Sending system — Mexico**: Deforestation, fragmentation, biodiversity loss, fire-related forest conversion, and agrochemical contamination are central sustainability effects [T1:W6, T1:W14, T1:W15, T1:W17].",
        "**Receiving systems**: Ecological impacts are mostly embodied and displaced to producing landscapes rather than occurring directly in consumer regions [T1:5].",
        "**Spillover systems — Chile**: International demand and trade are linked to pressures on water resources, biodiversity, and local communities [T1:1].",
        "**Spillover systems — distant forest regions**: Sustainability gains in importing regions may mask burdens shifted to other production landscapes [T1:5]."
      ],
      "technological / infrastructural": [
        "**Sending system — Mexico**: Export demand increases packing, refrigeration, irrigation, traceability, and road infrastructure.",
        "**Receiving systems**: Import logistics, ripening facilities, supermarket distribution, and food-service supply chains expand.",
        "**Spillover systems**: Competitor countries may invest in similar export infrastructure to capture global avocado demand."
      ],
      "cultural / social / demographic": [
        "**Sending system — Mexico**: Communities experience mixed effects: employment and reduced out-migration in some places, but inequality, insecurity, and weakened cohesion in others [T1:W7, T1:W11].",
        "**Receiving systems**: Consumers benefit from dietary options but may remain disconnected from environmental and human-rights burdens in producing regions [T1:5].",
        "**Spillover systems — Chile**: Water injustice and socioecological conflict can emerge where export-oriented avocado expansion competes with local water needs [T1:1, T1:3]."
      ],
      "hydrological": [
        "**Sending system — Mexico**: Virtual water is exported through avocados while water scarcity remains local; studies show irrigated avocado production can exceed legal agricultural water concessions in some settings [T1:W12, T1:W13].",
        "**Receiving systems**: Distant consumers import water-intensive fruit without directly experiencing the water stress embedded in production.",
        "**Spillover systems — Chile**: International demand has been linked to severe water scarcity, community reliance on cistern trucks, and socioecological conflict in Petorca and Central Chile [T1:3]."
      ],
      "climatic / atmospheric": [
        "**Sending system — Mexico**: Forest conversion reduces aboveground carbon storage [T1:W16].",
        "**Receiving systems**: Embodied emissions and carbon losses are associated with consumption, though direct consumer-region impacts are indirect.",
        "**Spillover systems**: If production expands in Chile, Peru, or other countries, land-carbon and water-stress impacts may be redistributed across the global avocado frontier."
      ],
      "geological / geomorphological": [
        "**Sending system — Mexico**: Soil degradation, erosion, and nutrient imbalance may follow intensive orchard management [T1:W7, T1:W16].",
        "**Spillover systems**: Expansion into less suitable soils or steep terrain in other producer countries may create erosion and land-degradation risks."
      ]
    }
  }
}
```

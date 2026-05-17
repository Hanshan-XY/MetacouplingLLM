# 06 — Parsed analysis

`AnalysisResult.parsed` — the LLM response (file 05) re-parsed into structured fields. This is what downstream consumers (formatter, map extractor, pericoupling validators) read.

```json
{
  "coupling_classification": "- **Intracoupling — present.** Avocado production creates strong within-system interactions in Mexican production landscapes: land conversion, orchard management, water use, labor, local capital accumulation, and effects on rural/Indigenous communities and ecosystems. Michoacán is especially relevant because it is repeatedly identified as Mexico’s avocado production center and a focal landscape of avocado-boom impacts [T1:1], [T1:W1].\n\n- **Pericoupling — likely present, but should be specified spatially.** If the focal system is defined as a major avocado-producing region such as Michoacán or Jalisco, then adjacent municipalities, watersheds, forest frontiers, labor-sending communities, and neighboring production regions are pericoupled through land-use displacement, labor flows, water competition, pest/disease movement, and infrastructure expansion. This is a provisional classification because the research description does not yet specify the exact focal region.\n\n- **Telecoupling — clearly present.** Mexican avocado production is strongly linked to distant consumer markets, especially the United States, through long-distance commodity trade, capital flows, market signals, and sustainability burdens externalized from consuming regions to producing landscapes. Literature links Mexico’s avocado expansion to trade liberalization, U.S. demand, and globally organized supply chains [T1:2], [T1:3], [T1:5], [T1:W2], [T1:W4].\n\n---",
  "cross_coupling_interactions": [
    "**Amplification across scales**: Telecoupled U.S. and international demand amplifies intracoupling within Mexican production landscapes by increasing orchard profitability, labor demand, irrigation investment, land conversion, and local infrastructure. The same telecoupled demand can also amplify pericoupling by pushing production into neighboring municipalities and watersheds.",
    "**Spatial tradeoffs**: Receiving systems benefit from abundant avocados, retailer profits, and consumer choice, while producing regions may bear deforestation, water stress, labor precarity, and biodiversity loss. This matches the broader telecoupling insight that consumer landscapes can appear sustainable while environmental burdens are displaced to distant production landscapes [T1:5].",
    "**Economic-social tradeoffs within Mexico**: Avocado exports can generate income and employment, but benefits may be unevenly distributed. Evidence from Mexico indicates that profits often concentrate with agribusinesses, while many workers face seasonal employment and rural/Indigenous communities may become precarious laborers in their own territories [T1:4].",
    "**Pericoupling leakage**: If forest enforcement or water regulation becomes stricter in one producing municipality, expansion may move to adjacent jurisdictions with weaker enforcement. This would convert an intracoupling sustainability intervention into a pericoupled displacement problem.",
    "**Telecoupled competition and spillovers**: Mexico’s role in global avocado trade affects other producers such as Chile and Peru. Conversely, climate shocks, water crises, or policy changes in Chile or Peru could increase demand for Mexican avocados, raising pressure on Mexican landscapes. Literature explicitly calls for more research on trade dynamics among Mexico, Chile, and Peru [T1:4].",
    "**Coupling transformations**:",
    "**Coupling**: Trade liberalization and U.S. demand strengthened the Mexico–U.S. avocado telecoupling after the 1994 free trade agreement [T1:2].",
    "**Decoupling**: Border closures, phytosanitary bans, droughts, violence, labor disruptions, or consumer boycotts could weaken flows.",
    "**Recoupling**: New certification systems, traceability, deforestation-free sourcing, or alternative trade routes could restore flows under modified governance conditions.",
    "**Cascading interactions**: A drought, pest outbreak, regulatory change, or violence-related disruption in Michoacán could reduce Mexican supply, raise prices in U.S. markets, stimulate expansion in Jalisco or other Mexican regions, and increase production pressure in Chile, Peru, or Colombia. Conversely, shifts in U.S. consumer demand could rapidly cascade back into land-use decisions in Mexican production frontiers."
  ],
  "intracoupling": {
    "systems": [
      {
        "role": "focal",
        "name": "Mexican avocado-producing landscapes, especially Michoacán; potentially also Jalisco and other expanding production frontiers.",
        "human_subsystem": "Avocado growers, Indigenous and rural communities, seasonal workers, packing houses, local elites, agribusiness firms, local governments, water users, landowners, ejidos and communal land institutions, local transport and processing actors.",
        "natural_subsystem": "Pine-oak forests, tropical dry forests, agricultural soils, watersheds, aquifers, biodiversity, pollinators, pests/pathogens, microclimates suitable for avocado production.",
        "geographic_scope": "Primarily Michoacán, Mexico’s central avocado-producing region, with possible extension to Jalisco and other expanding avocado frontiers. Michoacán has been described as the center of Mexico’s avocado production and a key site of rural and Indigenous community impacts [T1:4], while Jalisco is increasingly relevant in studies of Mexican avocado production and trade with Europe [T1:W5]."
      }
    ],
    "flows": [
      {
        "category": "matter",
        "direction": "Forest/agricultural land → avocado orchards within producing regions.",
        "description": "Land, biomass, soil nutrients, agrochemicals, irrigation water, and harvested avocados move through the local production system. Remote-sensing and frontier studies examine land-use change associated with Mexican avocado production, especially in Michoacán [T1:W2], [T1:W3]."
      },
      {
        "category": "capital",
        "direction": "Avocado sales revenue → growers, landowners, packing houses, local intermediaries, labor markets.",
        "description": "Export-oriented avocado production brings income into producing regions, but evidence from Mexico suggests profits are unevenly distributed and concentrated among agribusinesses rather than workers [T1:4]."
      },
      {
        "category": "people",
        "direction": "Local and migrant labor → orchards, packing houses, transport and processing nodes.",
        "description": "Workers move seasonally or permanently into avocado production and processing. The avocado industry in Mexico has reportedly created large numbers of jobs, but much of this work is seasonal and precarious [T1:4]."
      },
      {
        "category": "information",
        "direction": "Agribusinesses, certification bodies, exporters, and extension actors → growers and packers.",
        "description": "Quality standards, phytosanitary requirements, market prices, sustainability claims, and export protocols circulate within production regions and shape orchard management."
      },
      {
        "category": "energy",
        "direction": "Fuel/electricity suppliers → irrigation systems, transport, packing, refrigeration.",
        "description": "Energy is embodied in irrigation pumping, agrochemical production, cold-chain logistics, packing operations, and truck transport from orchards to export facilities."
      },
      {
        "category": "organisms",
        "direction": "Nurseries, orchards, and surrounding habitats ↔ avocado production landscapes.",
        "description": "Avocado seedlings, pollinators, pests, pathogens, and possibly invasive organisms move within production landscapes, affecting yields, biodiversity, and management intensity."
      }
    ],
    "agents": [
      {
        "name": "**[Individuals / households] Smallholder and medium-scale avocado growers**",
        "description": "Decide whether to convert land, intensify production, invest in irrigation, or participate in export markets.",
        "level": "individuals / households"
      },
      {
        "name": "**[Individuals / households] Rural and Indigenous households**",
        "description": "Experience livelihood opportunities, labor precarity, water stress, land pressure, and cultural-territorial change.",
        "level": "individuals / households"
      },
      {
        "name": "**[Individuals / households] Seasonal agricultural workers**",
        "description": "Provide labor for orchard management, harvesting, packing, and transport; often bear occupational and livelihood risks.",
        "level": "individuals / households"
      },
      {
        "name": "**[Firms / traders / corporations] Agribusiness firms, packers, exporters, and logistics companies**",
        "description": "Coordinate production, quality control, cold-chain logistics, and export market access.",
        "level": "firms / traders / corporations"
      },
      {
        "name": "**[Firms / traders / corporations] Supermarkets and marketers linked to export chains**",
        "description": "Shape demand, quality standards, prices, and branding of avocados as a healthy or “superfood” product.",
        "level": "firms / traders / corporations"
      },
      {
        "name": "**[Governments / policymakers] Mexican federal, state, and municipal authorities**",
        "description": "Regulate land use, water extraction, forest protection, phytosanitary standards, labor conditions, and export certification.",
        "level": "governments / policymakers"
      },
      {
        "name": "**[Organizations / NGOs] Universities, environmental NGOs, certification bodies, and producer associations**",
        "description": "Generate sustainability information, monitor land-use change, promote standards, and mediate conflicts.",
        "level": "organizations / NGOs"
      },
      {
        "name": "**[Non-human agents] Avocado trees, pests, pathogens, and pollinators**",
        "description": "Actively mediate production success, ecological impacts, chemical use, and land-management decisions.",
        "level": "non-human agents"
      }
    ],
    "causes": {
      "economic": [
        "Rising international demand for avocados, including demand associated with “superfood” diets, has contributed to expansion and intensification of avocado production in Mexico, Chile, and Peru [T1:3].",
        "High export profitability incentivizes land conversion from subsistence or diversified agriculture toward avocado orchards.",
        "Price signals from international markets encourage producers to intensify production and meet export standards."
      ],
      "political / institutional": [
        "Trade liberalization and free trade agreements helped shift avocado production from domestic consumption toward international markets [T1:3].",
        "The U.S.–Mexico Free Trade Agreement established in 1994 amplified avocado demand and helped transform Mexican avocado production into a globalized production chain [T1:2].",
        "Weak or uneven enforcement of land-use, forest, and water regulations can facilitate illegal or informal expansion."
      ],
      "ecological / biological": [
        "Suitable climate and soils in Michoacán and other highland regions enable avocado cultivation.",
        "Avocado’s water demand increases pressure on local watersheds, especially where orchards expand or irrigation intensifies.",
        "Pests, pathogens, and disease risks shape chemical use, monitoring, and certification requirements."
      ],
      "technological / infrastructural": [
        "Expansion of packing houses, roads, irrigation systems, cold-chain logistics, and export inspection infrastructure enables avocado commercialization.",
        "Remote sensing is increasingly used to assess avocado-related land-use change in Mexico [T1:W2], [T1:W3].",
        "Improved transport and refrigeration technologies allow perishable avocados to reach distant markets."
      ],
      "cultural / social / demographic": [
        "Avocado’s popularity in alternative Western diets and its branding as a “superfood” have increased global demand [T1:3].",
        "Local aspirations for income, employment, and participation in export agriculture encourage household-level participation.",
        "Rural demographic change and labor needs reshape community relations and land-use decisions."
      ],
      "hydrological": [
        "Avocado production requires significant water inputs, making water availability and irrigation access key drivers of orchard expansion.",
        "Competition among domestic, agricultural, and ecological water uses can intensify in avocado-producing watersheds."
      ],
      "climatic / atmospheric": [
        "Climate variability and warming can shift suitable production zones and alter water demand.",
        "Drought risk increases the vulnerability of avocado systems that depend on irrigation or moisture-sensitive production."
      ],
      "geological / geomorphological": [
        "Soil depth, volcanic soils, slopes, elevation, and drainage influence avocado suitability.",
        "Production on steep terrain may increase erosion risk if forest or perennial vegetation is removed."
      ]
    },
    "effects": {
      "economic": [
        "Avocado production generates income, employment, and export revenue in producing regions.",
        "Benefits are unevenly distributed: evidence from Mexico indicates that profits are often concentrated among agribusinesses, while workers may receive mostly seasonal and precarious employment [T1:4].",
        "Rising land values may benefit landowners while excluding land-poor households or small producers unable to meet export standards."
      ],
      "political / institutional": [
        "Expansion of avocado production can generate pressure for stronger land-use planning, forest enforcement, water regulation, and labor oversight.",
        "Local governance may be strained where export profitability outpaces regulatory capacity.",
        "Certification and traceability programs may improve accountability, but can also exclude smaller producers if compliance costs are high."
      ],
      "ecological / biological": [
        "Forest conversion and land-use change can reduce habitat quality, biodiversity, carbon storage, and regulating ecosystem services.",
        "Studies of Mexican avocado supply chains and frontier dynamics explicitly examine environmental impacts and land-use change linked to avocado production [T1:W2], [T1:W3].",
        "Monoculture expansion can simplify landscapes, increase pest vulnerability, and reduce ecological resilience."
      ],
      "technological / infrastructural": [
        "Export growth stimulates investment in roads, packing houses, irrigation, refrigeration, monitoring, and phytosanitary systems.",
        "These improvements can increase market access but also intensify production pressure and accelerate land conversion."
      ],
      "cultural / social / demographic": [
        "Rural and Indigenous communities may experience livelihood shifts from diversified farming or communal land uses toward wage labor and export agriculture.",
        "Evidence from Michoacán suggests the avocado boom has made some Indigenous and rural communities precarious agricultural workers in their own lands [T1:4].",
        "Social conflict may emerge around land, water, labor conditions, and unequal benefit distribution."
      ],
      "hydrological": [
        "Irrigation and orchard expansion can increase water extraction and stress local watersheds.",
        "Water scarcity effects may fall disproportionately on households, small farmers, and ecosystems lacking political or economic power."
      ],
      "climatic / atmospheric": [
        "Land-use change can reduce carbon storage and increase emissions associated with deforestation, transport, refrigeration, and agrochemical use.",
        "Export-oriented supply chains create embodied emissions through cold-chain logistics and long-distance transport."
      ],
      "geological / geomorphological": [
        "Forest clearing and orchard establishment on slopes may increase erosion, soil compaction, and sediment movement.",
        "Intensive orchard management can alter soil structure, nutrient cycling, and long-term land productivity."
      ]
    }
  },
  "is_parsed": true,
  "map_data": {
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
  },
  "pericoupling": {
    "systems": [
      {
        "role": "sending",
        "system_scope": "adjacent",
        "name": "Neighboring rural communities, municipalities, or states supplying labor, land, water, or production inputs to avocado-producing zones.",
        "human_subsystem": "Rural households, seasonal workers, input suppliers, local governments, transport operators, landowners.",
        "natural_subsystem": "Adjacent forests, agricultural lands, watersheds, aquifers, and biodiversity corridors.",
        "geographic_scope": "Municipalities and states adjacent to major avocado-producing regions, especially around Michoacán and Jalisco."
      },
      {
        "role": "receiving",
        "system_scope": "adjacent",
        "name": "Avocado-producing municipalities or expanding avocado frontiers.",
        "human_subsystem": "Avocado growers, packers, local labor markets, municipal authorities, producer associations.",
        "natural_subsystem": "Avocado orchards, converted forest/agricultural land, water sources, soils, local biodiversity.",
        "geographic_scope": "Core and frontier avocado-producing landscapes in Mexico."
      },
      {
        "role": "sending",
        "system_scope": "adjacent",
        "name": "Core avocado-producing zones generating environmental or economic spillovers into neighboring areas.",
        "human_subsystem": "Export growers, agribusiness firms, local authorities, intermediaries.",
        "natural_subsystem": "Orchards, watersheds, agrochemical-use zones, altered landscapes.",
        "geographic_scope": "Established avocado municipalities in Michoacán and comparable producing regions."
      },
      {
        "role": "receiving",
        "system_scope": "adjacent",
        "name": "Neighboring ecosystems, communities, watersheds, and agricultural regions affected by expansion or displacement.",
        "human_subsystem": "Non-avocado farmers, Indigenous and rural communities, downstream water users, local governments.",
        "natural_subsystem": "Forest patches, river systems, aquifers, soils, biodiversity corridors.",
        "geographic_scope": "Adjacent municipalities, shared watersheds, and nearby forest frontiers."
      }
    ],
    "flows": [
      {
        "category": "people",
        "direction": "Neighboring communities → avocado-producing municipalities.",
        "description": "Seasonal and permanent labor moves into orchards, packing houses, transport, and related services."
      },
      {
        "category": "matter",
        "direction": "Adjacent forests/agricultural lands → avocado production frontier.",
        "description": "Land, timber biomass, soil nutrients, irrigation water, and agrochemical inputs are transformed into avocado production capacity."
      },
      {
        "category": "matter",
        "direction": "Shared watersheds and aquifers → orchards and downstream users.",
        "description": "Surface water and groundwater are redistributed among avocado producers, households, downstream agriculture, and ecosystems."
      },
      {
        "category": "capital",
        "direction": "Core production regions ↔ adjacent land markets and service economies.",
        "description": "Avocado profits can finance land purchases, orchard expansion, irrigation, transport, and service-sector growth in nearby regions."
      },
      {
        "category": "information",
        "direction": "Established avocado producers, packers, and exporters → adjacent frontier producers.",
        "description": "Knowledge about cultivation techniques, export certification, prices, and market access spreads from core regions to neighboring areas."
      },
      {
        "category": "organisms",
        "direction": "Orchards ↔ adjacent habitats and farms.",
        "description": "Pests, pathogens, pollinators, and cultivated avocado varieties move across local ecological boundaries."
      }
    ],
    "agents": [
      {
        "name": "**[Individuals / households] Neighboring rural workers**",
        "description": "Supply labor to avocado orchards and packing facilities.",
        "level": "individuals / households"
      },
      {
        "name": "**[Individuals / households] Adjacent smallholders and landowners**",
        "description": "Decide whether to convert land, lease land, sell land, or enter avocado production.",
        "level": "individuals / households"
      },
      {
        "name": "**[Firms / traders / corporations] Input suppliers and local transport firms**",
        "description": "Provide seedlings, fertilizers, agrochemicals, irrigation equipment, and logistics services across municipal boundaries.",
        "level": "firms / traders / corporations"
      },
      {
        "name": "**[Firms / traders / corporations] Packers and exporters**",
        "description": "Extend sourcing networks into neighboring production frontiers.",
        "level": "firms / traders / corporations"
      },
      {
        "name": "**[Governments / policymakers] Municipal and state governments**",
        "description": "Manage land-use permissions, water regulation, forest enforcement, and rural development.",
        "level": "governments / policymakers"
      },
      {
        "name": "**[Organizations / NGOs] Local NGOs, universities, and producer associations**",
        "description": "Monitor land-use change, advise producers, and mediate conflicts.",
        "level": "organizations / NGOs"
      },
      {
        "name": "**[Non-human agents] Pests, pathogens, pollinators, and avocado cultivars**",
        "description": "Shape cross-boundary ecological dynamics and production risks.",
        "level": "non-human agents"
      }
    ],
    "causes": {
      "economic": [
        "High avocado profitability encourages expansion into adjacent areas.",
        "Labor demand in core production zones draws workers from nearby communities.",
        "Rising land values and investment opportunities incentivize land conversion in neighboring municipalities."
      ],
      "political / institutional": [
        "Differing municipal enforcement of forest, water, and land-use regulations can shift expansion pressure across boundaries.",
        "State-level agricultural promotion policies may encourage regional production growth.",
        "Export certification and phytosanitary requirements can structure which adjacent producers enter formal markets."
      ],
      "ecological / biological": [
        "Suitable adjacent habitats and agricultural lands enable avocado frontier expansion.",
        "Pest and disease movement across orchards and natural vegetation can affect neighboring systems.",
        "Landscape connectivity allows ecological effects, including biodiversity loss or pollinator disruption, to spread beyond orchard boundaries."
      ],
      "technological / infrastructural": [
        "Roads, packing facilities, irrigation networks, and cold-chain infrastructure expand the radius of profitable production.",
        "Knowledge and technology from established growers can be transferred to adjacent producers."
      ],
      "cultural / social / demographic": [
        "Neighboring households may shift livelihood strategies from subsistence farming or diversified agriculture to wage labor or avocado production.",
        "Social tensions can arise between communities benefiting from avocado expansion and those bearing environmental costs."
      ],
      "hydrological": [
        "Shared aquifers and watersheds connect avocado orchards with adjacent communities and ecosystems.",
        "Water extraction in one municipality can reduce availability or quality in neighboring areas."
      ],
      "climatic / atmospheric": [
        "Climate variability can push production into adjacent zones perceived as more suitable or less water-stressed.",
        "Local microclimatic differences influence where avocado expansion is most profitable."
      ],
      "geological / geomorphological": [
        "Soil suitability, slope, elevation, and drainage patterns shape adjacent expansion.",
        "Conversion on steep or erosion-prone terrain can create downstream sediment effects."
      ]
    },
    "effects": {
      "economic": [
        "Avocado-producing focal system: Increased income, employment, and investment.",
        "Adjacent systems: New labor opportunities but also rising land prices and possible displacement of non-avocado livelihoods.",
        "Adjacent smallholders may be excluded if they lack capital, irrigation access, or certification capacity."
      ],
      "political / institutional": [
        "Focal and adjacent systems: Increased pressure for coordinated land-use planning, watershed governance, and forest enforcement.",
        "Adjacent systems: Regulatory leakage may occur if expansion moves to jurisdictions with weaker oversight."
      ],
      "ecological / biological": [
        "Focal system: Habitat simplification and biodiversity loss from orchard expansion.",
        "Adjacent systems: Land-use displacement, forest-edge degradation, and reduced habitat connectivity.",
        "Shared landscapes: Pest and disease pressures may increase with orchard concentration."
      ],
      "technological / infrastructural": [
        "Focal system: Growth of packing, irrigation, monitoring, and road infrastructure.",
        "Adjacent systems: Infrastructure expansion can improve market access but also accelerate land conversion."
      ],
      "cultural / social / demographic": [
        "Focal system: Labor demand and export earnings may increase local opportunity.",
        "Adjacent systems: Rural households may become more dependent on seasonal avocado labor.",
        "Indigenous and communal land systems may face pressure from market-driven land-use change, similar to broader findings that global value chains can reshape rural and Indigenous landscapes, economies, and cultures [T1:8]."
      ],
      "hydrological": [
        "Focal system: Increased irrigation demand and altered local water allocation.",
        "Adjacent systems: Potential downstream or cross-boundary water scarcity, reduced streamflow, or conflict among water users."
      ],
      "climatic / atmospheric": [
        "Focal and adjacent systems: Forest loss can reduce local carbon storage and alter microclimates.",
        "Adjacent systems: Expansion into new zones may shift climate-related risks rather than eliminate them."
      ],
      "geological / geomorphological": [
        "Focal system: Soil compaction and erosion risks under intensive orchard management.",
        "Adjacent systems: Downstream sedimentation or slope instability may increase where forest clearing occurs on steep terrain."
      ]
    }
  },
  "pericoupling_info": {
    "level": "adm1",
    "focal_region": "Jalisco (MEX014)",
    "focal_country": "Mexico (MEX)",
    "domestic_neighbors": "Aguascalientes (MEX001), Colima (MEX006), Guanajuato (MEX011), Michoacán de Ocampo (MEX016), Nayarit (MEX018), Zacatecas (MEX032)",
    "note": "LLM classification is consistent with the ADM1 pericoupling database."
  },
  "raw_text": "## 1. Coupling Classification\n\n- **Intracoupling — present.** Avocado production creates strong within-system interactions in Mexican production landscapes: land conversion, orchard management, water use, labor, local capital accumulation, and effects on rural/Indigenous communities and ecosystems. Michoacán is especially relevant because it is repeatedly identified as Mexico’s avocado production center and a focal landscape of avocado-boom impacts [T1:1], [T1:W1].\n\n- **Pericoupling — likely present, but should be specified spatially.** If the focal system is defined as a major avocado-producing region such as Michoacán or Jalisco, then adjacent municipalities, watersheds, forest frontiers, labor-sending communities, and neighboring production regions are pericoupled through land-use displacement, labor flows, water competition, pest/disease movement, and infrastructure expansion. This is a provisional classification because the research description does not yet specify the exact focal region.\n\n- **Telecoupling — clearly present.** Mexican avocado production is strongly linked to distant consumer markets, especially the United States, through long-distance commodity trade, capital flows, market signals, and sustainability burdens externalized from consuming regions to producing landscapes. Literature links Mexico’s avocado expansion to trade liberalization, U.S. demand, and globally organized supply chains [T1:2], [T1:3], [T1:5], [T1:W2], [T1:W4].\n\n---\n\n## 2. Intracoupling Analysis — within the focal system\n\n### 2.1 Systems Identification\n\n**Focal System**: Mexican avocado-producing landscapes, especially Michoacán; potentially also Jalisco and other expanding production frontiers.\n\n- **Human subsystem**: Avocado growers, Indigenous and rural communities, seasonal workers, packing houses, local elites, agribusiness firms, local governments, water users, landowners, ejidos and communal land institutions, local transport and processing actors.\n- **Natural subsystem**: Pine-oak forests, tropical dry forests, agricultural soils, watersheds, aquifers, biodiversity, pollinators, pests/pathogens, microclimates suitable for avocado production.\n- **Geographic scope**: Primarily Michoacán, Mexico’s central avocado-producing region, with possible extension to Jalisco and other expanding avocado frontiers. Michoacán has been described as the center of Mexico’s avocado production and a key site of rural and Indigenous community impacts [T1:4], while Jalisco is increasingly relevant in studies of Mexican avocado production and trade with Europe [T1:W5].\n\n### 2.2 Flows Analysis\n\n**Matter Flow**\n\n- **Direction**: Forest/agricultural land → avocado orchards within producing regions.\n- **Description**: Land, biomass, soil nutrients, agrochemicals, irrigation water, and harvested avocados move through the local production system. Remote-sensing and frontier studies examine land-use change associated with Mexican avocado production, especially in Michoacán [T1:W2], [T1:W3].\n\n**Capital Flow**\n\n- **Direction**: Avocado sales revenue → growers, landowners, packing houses, local intermediaries, labor markets.\n- **Description**: Export-oriented avocado production brings income into producing regions, but evidence from Mexico suggests profits are unevenly distributed and concentrated among agribusinesses rather than workers [T1:4].\n\n**People Flow**\n\n- **Direction**: Local and migrant labor → orchards, packing houses, transport and processing nodes.\n- **Description**: Workers move seasonally or permanently into avocado production and processing. The avocado industry in Mexico has reportedly created large numbers of jobs, but much of this work is seasonal and precarious [T1:4].\n\n**Information Flow**\n\n- **Direction**: Agribusinesses, certification bodies, exporters, and extension actors → growers and packers.\n- **Description**: Quality standards, phytosanitary requirements, market prices, sustainability claims, and export protocols circulate within production regions and shape orchard management.\n\n**Energy Flow**\n\n- **Direction**: Fuel/electricity suppliers → irrigation systems, transport, packing, refrigeration.\n- **Description**: Energy is embodied in irrigation pumping, agrochemical production, cold-chain logistics, packing operations, and truck transport from orchards to export facilities.\n\n**Organism Flow**\n\n- **Direction**: Nurseries, orchards, and surrounding habitats ↔ avocado production landscapes.\n- **Description**: Avocado seedlings, pollinators, pests, pathogens, and possibly invasive organisms move within production landscapes, affecting yields, biodiversity, and management intensity.\n\n### 2.3 Agents\n\n- **[Individuals / households] Smallholder and medium-scale avocado growers**: Decide whether to convert land, intensify production, invest in irrigation, or participate in export markets.\n- **[Individuals / households] Rural and Indigenous households**: Experience livelihood opportunities, labor precarity, water stress, land pressure, and cultural-territorial change.\n- **[Individuals / households] Seasonal agricultural workers**: Provide labor for orchard management, harvesting, packing, and transport; often bear occupational and livelihood risks.\n- **[Firms / traders / corporations] Agribusiness firms, packers, exporters, and logistics companies**: Coordinate production, quality control, cold-chain logistics, and export market access.\n- **[Firms / traders / corporations] Supermarkets and marketers linked to export chains**: Shape demand, quality standards, prices, and branding of avocados as a healthy or “superfood” product.\n- **[Governments / policymakers] Mexican federal, state, and municipal authorities**: Regulate land use, water extraction, forest protection, phytosanitary standards, labor conditions, and export certification.\n- **[Organizations / NGOs] Universities, environmental NGOs, certification bodies, and producer associations**: Generate sustainability information, monitor land-use change, promote standards, and mediate conflicts.\n- **[Non-human agents] Avocado trees, pests, pathogens, and pollinators**: Actively mediate production success, ecological impacts, chemical use, and land-management decisions.\n\n### 2.4 Causes\n\n**Economic**\n\n- Rising international demand for avocados, including demand associated with “superfood” diets, has contributed to expansion and intensification of avocado production in Mexico, Chile, and Peru [T1:3].\n- High export profitability incentivizes land conversion from subsistence or diversified agriculture toward avocado orchards.\n- Price signals from international markets encourage producers to intensify production and meet export standards.\n\n**Political / Institutional**\n\n- Trade liberalization and free trade agreements helped shift avocado production from domestic consumption toward international markets [T1:3].\n- The U.S.–Mexico Free Trade Agreement established in 1994 amplified avocado demand and helped transform Mexican avocado production into a globalized production chain [T1:2].\n- Weak or uneven enforcement of land-use, forest, and water regulations can facilitate illegal or informal expansion.\n\n**Ecological / Biological**\n\n- Suitable climate and soils in Michoacán and other highland regions enable avocado cultivation.\n- Avocado’s water demand increases pressure on local watersheds, especially where orchards expand or irrigation intensifies.\n- Pests, pathogens, and disease risks shape chemical use, monitoring, and certification requirements.\n\n**Technological / Infrastructural**\n\n- Expansion of packing houses, roads, irrigation systems, cold-chain logistics, and export inspection infrastructure enables avocado commercialization.\n- Remote sensing is increasingly used to assess avocado-related land-use change in Mexico [T1:W2], [T1:W3].\n- Improved transport and refrigeration technologies allow perishable avocados to reach distant markets.\n\n**Cultural / Social / Demographic**\n\n- Avocado’s popularity in alternative Western diets and its branding as a “superfood” have increased global demand [T1:3].\n- Local aspirations for income, employment, and participation in export agriculture encourage household-level participation.\n- Rural demographic change and labor needs reshape community relations and land-use decisions.\n\n**Hydrological**\n\n- Avocado production requires significant water inputs, making water availability and irrigation access key drivers of orchard expansion.\n- Competition among domestic, agricultural, and ecological water uses can intensify in avocado-producing watersheds.\n\n**Climatic / Atmospheric**\n\n- Climate variability and warming can shift suitable production zones and alter water demand.\n- Drought risk increases the vulnerability of avocado systems that depend on irrigation or moisture-sensitive production.\n\n**Geological / Geomorphological**\n\n- Soil depth, volcanic soils, slopes, elevation, and drainage influence avocado suitability.\n- Production on steep terrain may increase erosion risk if forest or perennial vegetation is removed.\n\n### 2.5 Effects\n\n**Economic**\n\n- Avocado production generates income, employment, and export revenue in producing regions.\n- Benefits are unevenly distributed: evidence from Mexico indicates that profits are often concentrated among agribusinesses, while workers may receive mostly seasonal and precarious employment [T1:4].\n- Rising land values may benefit landowners while excluding land-poor households or small producers unable to meet export standards.\n\n**Political / Institutional**\n\n- Expansion of avocado production can generate pressure for stronger land-use planning, forest enforcement, water regulation, and labor oversight.\n- Local governance may be strained where export profitability outpaces regulatory capacity.\n- Certification and traceability programs may improve accountability, but can also exclude smaller producers if compliance costs are high.\n\n**Ecological / Biological**\n\n- Forest conversion and land-use change can reduce habitat quality, biodiversity, carbon storage, and regulating ecosystem services.\n- Studies of Mexican avocado supply chains and frontier dynamics explicitly examine environmental impacts and land-use change linked to avocado production [T1:W2], [T1:W3].\n- Monoculture expansion can simplify landscapes, increase pest vulnerability, and reduce ecological resilience.\n\n**Technological / Infrastructural**\n\n- Export growth stimulates investment in roads, packing houses, irrigation, refrigeration, monitoring, and phytosanitary systems.\n- These improvements can increase market access but also intensify production pressure and accelerate land conversion.\n\n**Cultural / Social / Demographic**\n\n- Rural and Indigenous communities may experience livelihood shifts from diversified farming or communal land uses toward wage labor and export agriculture.\n- Evidence from Michoacán suggests the avocado boom has made some Indigenous and rural communities precarious agricultural workers in their own lands [T1:4].\n- Social conflict may emerge around land, water, labor conditions, and unequal benefit distribution.\n\n**Hydrological**\n\n- Irrigation and orchard expansion can increase water extraction and stress local watersheds.\n- Water scarcity effects may fall disproportionately on households, small farmers, and ecosystems lacking political or economic power.\n\n**Climatic / Atmospheric**\n\n- Land-use change can reduce carbon storage and increase emissions associated with deforestation, transport, refrigeration, and agrochemical use.\n- Export-oriented supply chains create embodied emissions through cold-chain logistics and long-distance transport.\n\n**Geological / Geomorphological**\n\n- Forest clearing and orchard establishment on slopes may increase erosion, soil compaction, and sediment movement.\n- Intensive orchard management can alter soil structure, nutrient cycling, and long-term land productivity.\n\n---\n\n## 3. Pericoupling Analysis — adjacent systems\n\n*This section is provisional because the research description does not define a precise focal region. It applies if the focal system is specified as Michoacán, Jalisco, or another avocado-producing state or watershed.*\n\n### 3.1 Systems Identification\n\n**Sending System (adjacent)**: Neighboring rural communities, municipalities, or states supplying labor, land, water, or production inputs to avocado-producing zones.\n\n- **Human subsystem**: Rural households, seasonal workers, input suppliers, local governments, transport operators, landowners.\n- **Natural subsystem**: Adjacent forests, agricultural lands, watersheds, aquifers, and biodiversity corridors.\n- **Geographic scope**: Municipalities and states adjacent to major avocado-producing regions, especially around Michoacán and Jalisco.\n\n**Receiving System (adjacent)**: Avocado-producing municipalities or expanding avocado frontiers.\n\n- **Human subsystem**: Avocado growers, packers, local labor markets, municipal authorities, producer associations.\n- **Natural subsystem**: Avocado orchards, converted forest/agricultural land, water sources, soils, local biodiversity.\n- **Geographic scope**: Core and frontier avocado-producing landscapes in Mexico.\n\n**Sending System (adjacent)**: Core avocado-producing zones generating environmental or economic spillovers into neighboring areas.\n\n- **Human subsystem**: Export growers, agribusiness firms, local authorities, intermediaries.\n- **Natural subsystem**: Orchards, watersheds, agrochemical-use zones, altered landscapes.\n- **Geographic scope**: Established avocado municipalities in Michoacán and comparable producing regions.\n\n**Receiving System (adjacent)**: Neighboring ecosystems, communities, watersheds, and agricultural regions affected by expansion or displacement.\n\n- **Human subsystem**: Non-avocado farmers, Indigenous and rural communities, downstream water users, local governments.\n- **Natural subsystem**: Forest patches, river systems, aquifers, soils, biodiversity corridors.\n- **Geographic scope**: Adjacent municipalities, shared watersheds, and nearby forest frontiers.\n\n### 3.2 Flows Analysis\n\n**People Flow**\n\n- **Direction**: Neighboring communities → avocado-producing municipalities.\n- **Description**: Seasonal and permanent labor moves into orchards, packing houses, transport, and related services.\n\n**Matter Flow**\n\n- **Direction**: Adjacent forests/agricultural lands → avocado production frontier.\n- **Description**: Land, timber biomass, soil nutrients, irrigation water, and agrochemical inputs are transformed into avocado production capacity.\n\n**Water / Matter Flow**\n\n- **Direction**: Shared watersheds and aquifers → orchards and downstream users.\n- **Description**: Surface water and groundwater are redistributed among avocado producers, households, downstream agriculture, and ecosystems.\n\n**Capital Flow**\n\n- **Direction**: Core production regions ↔ adjacent land markets and service economies.\n- **Description**: Avocado profits can finance land purchases, orchard expansion, irrigation, transport, and service-sector growth in nearby regions.\n\n**Information Flow**\n\n- **Direction**: Established avocado producers, packers, and exporters → adjacent frontier producers.\n- **Description**: Knowledge about cultivation techniques, export certification, prices, and market access spreads from core regions to neighboring areas.\n\n**Organism Flow**\n\n- **Direction**: Orchards ↔ adjacent habitats and farms.\n- **Description**: Pests, pathogens, pollinators, and cultivated avocado varieties move across local ecological boundaries.\n\n### 3.3 Agents\n\n- **[Individuals / households] Neighboring rural workers**: Supply labor to avocado orchards and packing facilities.\n- **[Individuals / households] Adjacent smallholders and landowners**: Decide whether to convert land, lease land, sell land, or enter avocado production.\n- **[Firms / traders / corporations] Input suppliers and local transport firms**: Provide seedlings, fertilizers, agrochemicals, irrigation equipment, and logistics services across municipal boundaries.\n- **[Firms / traders / corporations] Packers and exporters**: Extend sourcing networks into neighboring production frontiers.\n- **[Governments / policymakers] Municipal and state governments**: Manage land-use permissions, water regulation, forest enforcement, and rural development.\n- **[Organizations / NGOs] Local NGOs, universities, and producer associations**: Monitor land-use change, advise producers, and mediate conflicts.\n- **[Non-human agents] Pests, pathogens, pollinators, and avocado cultivars**: Shape cross-boundary ecological dynamics and production risks.\n\n### 3.4 Causes\n\n**Economic**\n\n- High avocado profitability encourages expansion into adjacent areas.\n- Labor demand in core production zones draws workers from nearby communities.\n- Rising land values and investment opportunities incentivize land conversion in neighboring municipalities.\n\n**Political / Institutional**\n\n- Differing municipal enforcement of forest, water, and land-use regulations can shift expansion pressure across boundaries.\n- State-level agricultural promotion policies may encourage regional production growth.\n- Export certification and phytosanitary requirements can structure which adjacent producers enter formal markets.\n\n**Ecological / Biological**\n\n- Suitable adjacent habitats and agricultural lands enable avocado frontier expansion.\n- Pest and disease movement across orchards and natural vegetation can affect neighboring systems.\n- Landscape connectivity allows ecological effects, including biodiversity loss or pollinator disruption, to spread beyond orchard boundaries.\n\n**Technological / Infrastructural**\n\n- Roads, packing facilities, irrigation networks, and cold-chain infrastructure expand the radius of profitable production.\n- Knowledge and technology from established growers can be transferred to adjacent producers.\n\n**Cultural / Social / Demographic**\n\n- Neighboring households may shift livelihood strategies from subsistence farming or diversified agriculture to wage labor or avocado production.\n- Social tensions can arise between communities benefiting from avocado expansion and those bearing environmental costs.\n\n**Hydrological**\n\n- Shared aquifers and watersheds connect avocado orchards with adjacent communities and ecosystems.\n- Water extraction in one municipality can reduce availability or quality in neighboring areas.\n\n**Climatic / Atmospheric**\n\n- Climate variability can push production into adjacent zones perceived as more suitable or less water-stressed.\n- Local microclimatic differences influence where avocado expansion is most profitable.\n\n**Geological / Geomorphological**\n\n- Soil suitability, slope, elevation, and drainage patterns shape adjacent expansion.\n- Conversion on steep or erosion-prone terrain can create downstream sediment effects.\n\n### 3.5 Effects\n\n**Economic**\n\n- Avocado-producing focal system: Increased income, employment, and investment.\n- Adjacent systems: New labor opportunities but also rising land prices and possible displacement of non-avocado livelihoods.\n- Adjacent smallholders may be excluded if they lack capital, irrigation access, or certification capacity.\n\n**Political / Institutional**\n\n- Focal and adjacent systems: Increased pressure for coordinated land-use planning, watershed governance, and forest enforcement.\n- Adjacent systems: Regulatory leakage may occur if expansion moves to jurisdictions with weaker oversight.\n\n**Ecological / Biological**\n\n- Focal system: Habitat simplification and biodiversity loss from orchard expansion.\n- Adjacent systems: Land-use displacement, forest-edge degradation, and reduced habitat connectivity.\n- Shared landscapes: Pest and disease pressures may increase with orchard concentration.\n\n**Technological / Infrastructural**\n\n- Focal system: Growth of packing, irrigation, monitoring, and road infrastructure.\n- Adjacent systems: Infrastructure expansion can improve market access but also accelerate land conversion.\n\n**Cultural / Social / Demographic**\n\n- Focal system: Labor demand and export earnings may increase local opportunity.\n- Adjacent systems: Rural households may become more dependent on seasonal avocado labor.\n- Indigenous and communal land systems may face pressure from market-driven land-use change, similar to broader findings that global value chains can reshape rural and Indigenous landscapes, economies, and cultures [T1:8].\n\n**Hydrological**\n\n- Focal system: Increased irrigation demand and altered local water allocation.\n- Adjacent systems: Potential downstream or cross-boundary water scarcity, reduced streamflow, or conflict among water users.\n\n**Climatic / Atmospheric**\n\n- Focal and adjacent systems: Forest loss can reduce local carbon storage and alter microclimates.\n- Adjacent systems: Expansion into new zones may shift climate-related risks rather than eliminate them.\n\n**Geological / Geomorphological**\n\n- Focal system: Soil compaction and erosion risks under intensive orchard management.\n- Adjacent systems: Downstream sedimentation or slope instability may increase where forest clearing occurs on steep terrain.\n\n---\n\n## 4. Telecoupling Analysis — distant systems\n\n### 4.1 Systems Identification\n\n**Sending System (distant)**: Mexican avocado-producing regions, especially Michoacán; potentially Jalisco and other export-oriented regions.\n\n- **Human subsystem**: Growers, workers, agribusinesses, packers, exporters, logistics firms, landowners, Indigenous and rural communities, Mexican government agencies.\n- **Natural subsystem**: Forests, agricultural land, watersheds, aquifers, soils, biodiversity, carbon stocks, avocado orchards.\n- **Geographic scope**: Major avocado-producing and exporting regions in Mexico, with Michoacán central and Jalisco increasingly relevant.\n\n**Receiving System (distant)**: United States consumer and retail system.\n\n- **Human subsystem**: Consumers, supermarkets, restaurants, importers, distributors, marketing firms, food-service companies, U.S. regulators.\n- **Natural subsystem**: Urban and suburban food-consumption landscapes, domestic agricultural land indirectly spared or substituted, waste streams associated with food consumption.\n- **Geographic scope**: U.S. avocado-importing and consuming regions. The U.S.–Mexico avocado supply chain has been the subject of environmental impact research [T1:W2], [T1:W4].\n\n**Receiving System (distant)**: European consumer markets, if included in the study.\n\n- **Human subsystem**: European consumers, retailers, importers, certification bodies, sustainability initiatives.\n- **Natural subsystem**: Consumption-related waste streams, domestic agricultural systems indirectly affected by imports.\n- **Geographic scope**: European avocado-importing countries. Available context includes a report on Mexican avocado production and trade with Europe, focusing on Jalisco and deforestation-free value-chain concerns [T1:W5].\n\n**Spillover System**: Competing avocado-producing countries, especially Chile and Peru.\n\n- **Human subsystem**: Avocado growers, exporters, workers, governments, rural communities, and agribusinesses in competing producer countries.\n- **Natural subsystem**: Water-stressed avocado landscapes, biodiversity, soils, watersheds, and ecosystems affected by export-oriented production.\n- **Geographic scope**: Chile and Peru as major Latin American avocado exporters. Literature notes that Mexico, Chile, and Peru are among the world’s biggest avocado producers/exporters and that trade dynamics among them need further investigation [T1:3], [T1:4].\n\n**Spillover System**: Non-avocado agricultural producers and food systems affected by market displacement.\n\n- **Human subsystem**: Farmers producing subsistence crops or alternative commodities, domestic consumers, local traders, food-security institutions.\n- **Natural subsystem**: Agricultural lands, forests, soils, and water resources that may be converted or spared depending on market shifts.\n- **Geographic scope**: Mexican domestic food-producing regions and other countries affected by changing avocado prices, supply competition, or substitution effects.\n\n**Spillover System**: Distant “apparent sustainability” landscapes in importing countries.\n\n- **Human subsystem**: Consumers, retailers, policymakers, urban sustainability advocates.\n- **Natural subsystem**: Importing-country landscapes where environmental burdens may appear lower because production impacts are displaced abroad.\n- **Geographic scope**: U.S. urban consumer landscapes and other affluent consumption regions. Dade et al. describe U.S. urban consumer landscapes and Michoacán avocado-producing landscapes as telecoupled through trade and externalized ecosystem-service burdens [T1:5].\n\n### 4.2 Flows Analysis\n\n**Matter Flow**\n\n- **Direction**: Mexico → United States.\n- **Description**: Fresh avocados and avocado products move from Mexican production landscapes to U.S. consumers. The U.S.–Mexico avocado supply chain is documented as a major focus of environmental impact research [T1:W2], [T1:W4].\n\n**Matter Flow**\n\n- **Direction**: Mexico → Europe and other international markets.\n- **Description**: Avocados exported from Mexico to European and other markets create additional long-distance commodity links; Jalisco has been examined in relation to Mexican avocado production and trade with Europe [T1:W5].\n\n**Capital Flow**\n\n- **Direction**: United States / international consumers and retailers → Mexican exporters, packers, growers, and intermediaries.\n- **Description**: Payments for avocados, retail contracts, export revenues, and investment signals flow back to Mexico, financing orchard expansion, packing, logistics, and land markets.\n\n**Information Flow**\n\n- **Direction**: United States / European retailers and consumers → Mexican producers and exporters.\n- **Description**: Demand signals, quality standards, sustainability expectations, price information, phytosanitary standards, and branding narratives influence Mexican production decisions. Distant consumption and marketing are identified as drivers of land-use change and offstage environmental burdens [T1:5].\n\n**Information Flow**\n\n- **Direction**: Mexican producers, researchers, NGOs, and media → international consumers and policymakers.\n- **Description**: Information about deforestation, water use, labor conditions, and sustainability risks circulates back to importing markets, potentially affecting purchasing behavior, certification, and policy.\n\n**Energy Flow**\n\n- **Direction**: Mexico → United States / Europe as embodied energy in avocado exports.\n- **Description**: Energy embedded in irrigation, fertilizer, agrochemicals, machinery, refrigeration, packing, and transport is transferred virtually through avocado consumption.\n\n**Hydrological / Matter Flow**\n\n- **Direction**: Mexican watersheds → distant consumers as virtual water embedded in avocados.\n- **Description**: Water used to produce exported avocados is effectively embodied in the commodity and consumed abroad, shifting water burdens to Mexican production regions.\n\n**People Flow**\n\n- **Direction**: Limited direct Mexico ↔ importing-country movement, but labor is indirectly mobilized by export demand.\n- **Description**: Export demand stimulates local and regional labor flows within Mexico; international migration may also be indirectly affected through rural livelihood change, though this requires additional data.\n\n**Organism Flow**\n\n- **Direction**: Mexico → importing countries, potentially through fruit shipments.\n- **Description**: Avocado fruits and packaging can carry phytosanitary risks, although regulated inspection systems are intended to prevent pest or pathogen movement.\n\n### 4.3 Agents\n\n- **[Individuals / households] U.S. and European consumers**: Drive demand through dietary preferences, including avocado’s popularity as a healthy or “superfood” product [T1:3].\n- **[Individuals / households] Mexican growers and workers**: Produce and harvest avocados, experiencing both livelihood opportunities and environmental/social risks.\n- **[Firms / traders / corporations] Mexican exporters, packers, and agribusiness firms**: Organize production, packing, certification, and export logistics.\n- **[Firms / traders / corporations] U.S. importers, supermarkets, restaurants, and distributors**: Set quality standards, manage procurement, market avocados, and transmit price signals.\n- **[Firms / traders / corporations] International logistics and cold-chain firms**: Move avocados across long distances while maintaining quality.\n- **[Governments / policymakers] Mexican federal and state governments**: Regulate production, land use, water, labor, phytosanitary compliance, and export authorization.\n- **[Governments / policymakers] U.S. and European regulators**: Set import rules, food-safety standards, phytosanitary requirements, and sustainability-related market conditions.\n- **[Organizations / NGOs] Certification bodies, environmental NGOs, universities, and sustainability initiatives**: Assess deforestation, water use, labor risks, and supply-chain transparency.\n- **[Non-human agents] Avocado trees, pests, pathogens, and pollinators**: Influence yield, pest risk, agrochemical use, and phytosanitary regulation.\n\n### 4.4 Causes\n\n**Economic**\n\n- International demand for avocados has increased significantly over the past two decades [T1:3].\n- Avocado’s status as a globally traded commodity encourages export-oriented production and intensification.\n- High prices and reliable access to U.S. markets create incentives for orchard expansion, land acquisition, and infrastructure investment.\n- Supermarkets, landowner elites, local farmers, and global companies participate in complex production chains shaped by economic liberalization [T1:2].\n\n**Political / Institutional**\n\n- Trade liberalization and free trade agreements helped reorient production from domestic consumption toward international markets [T1:3].\n- The U.S.–Mexico Free Trade Agreement established in 1994 amplified U.S. demand and helped transform Mexican avocado production into a globalized chain [T1:2].\n- Import rules, phytosanitary agreements, and certification systems determine which Mexican regions and producers can access distant markets.\n\n**Ecological / Biological**\n\n- Mexico’s suitable avocado-growing environments provide the ecological basis for export expansion.\n- Avocado’s water demand links consumer markets to watershed stress in production regions.\n- Pest and disease risks create strong phytosanitary governance connections between exporting and importing systems.\n\n**Technological / Infrastructural**\n\n- Cold-chain logistics, roads, packing houses, and border inspection infrastructure make large-scale avocado trade possible.\n- Remote sensing and supply-chain monitoring are increasingly used to assess environmental impacts and land-use change in Mexico [T1:W2], [T1:W3].\n- Export certification systems transmit distant market requirements into local production practices.\n\n**Cultural / Social / Demographic**\n\n- Avocado consumption is shaped by dietary trends, health branding, and “superfood” narratives in Western consumer markets [T1:3].\n- Marketing campaigns and retailer promotion create demand spikes and normalize year-round avocado consumption.\n- Consumer concern about sustainability may generate pressure for deforestation-free or water-responsible supply chains.\n\n**Hydrological**\n\n- Virtual water demand from importing countries increases pressure on Mexican watersheds.\n- Water availability and irrigation access condition which producers can participate in export markets.\n\n**Climatic / Atmospheric**\n\n- Climate variability affects yields, irrigation needs, pest risk, and future production suitability.\n- Long-distance transport and cold chains add embodied greenhouse-gas emissions to avocado trade.\n\n**Geological / Geomorphological**\n\n- Soil suitability, slope, elevation, and volcanic landscapes shape where export-oriented avocado orchards expand.\n- Terrain constraints influence erosion and infrastructure costs.\n\n### 4.5 Effects\n\n**Economic**\n\n- Sending system — Mexico: Export revenue, employment, agribusiness growth, land-value increases, and infrastructure investment.\n- Sending system — Mexico: Unequal benefit distribution, with evidence that Mexican avocado profits are concentrated among agribusinesses while many workers receive seasonal employment [T1:4].\n- Receiving system — United States / Europe: Reliable avocado supply, consumer welfare, retailer profits, and food-service revenues.\n- Spillover systems — Chile and Peru: Competition with Mexico can affect prices, market access, production decisions, and incentives for expansion. The trade dynamics among Mexico, Chile, and Peru are identified as an important research need [T1:4].\n\n**Political / Institutional**\n\n- Sending system — Mexico: Increased need for land-use enforcement, water governance, labor oversight, anti-deforestation rules, and traceability.\n- Receiving system — United States / Europe: Sustainability concerns may lead to retailer standards, certification, or deforestation-free procurement requirements.\n- Spillover systems — competing producers: Countries such as Chile and Peru may adjust export strategies, regulations, or sustainability claims in response to Mexican market dominance.\n\n**Ecological / Biological**\n\n- Sending system — Mexico: Deforestation, habitat fragmentation, biodiversity loss, and reduced regulating ecosystem services may occur where orchard expansion replaces forest or diversified agriculture. Dade et al. identify deforestation, water scarcity, and lost regulating services as telecoupled burdens associated with U.S. avocado consumption and Michoacán production [T1:5].\n- Receiving system — United States / Europe: Environmental burdens are partly displaced abroad, creating apparent sustainability gains in consumer landscapes while impacts occur in production landscapes [T1:5].\n- Spillover systems — Chile and Peru: Competitive pressures may encourage expansion or intensification in other avocado-producing regions, with associated water and biodiversity impacts. Chilean avocado production has been linked to water stress, biodiversity pressure, and local community impacts under international demand [T1:1].\n\n**Technological / Infrastructural**\n\n- Sending system — Mexico: Expansion of packing houses, irrigation systems, roads, cold-chain infrastructure, and monitoring systems.\n- Receiving system — United States / Europe: Improved distribution and retail systems support year-round avocado availability.\n- Spillover systems: Competing producer countries may invest in similar export infrastructure to maintain market share.\n\n**Cultural / Social / Demographic**\n\n- Sending system — Mexico: Rural communities may shift from subsistence or diversified livelihoods toward export-oriented production and wage labor.\n- Sending system — Mexico: Indigenous and rural communities may become precarious workers in landscapes where they previously held stronger livelihood or territorial roles [T1:4].\n- Receiving system — United States / Europe: Avocado consumption becomes normalized as part of health-oriented diets, brunch culture, and “superfood” consumption.\n- Spillover systems: Global avocado demand may reshape rural labor relations in other producing countries.\n\n**Hydrological**\n\n- Sending system — Mexico: Virtual water exports can intensify local water stress, particularly where irrigation expands or groundwater governance is weak.\n- Receiving system — United States / Europe: Consumers receive avocado benefits without directly experiencing production-region water scarcity.\n- Spillover systems — Chile and Peru: Competitive expansion may intensify water stress in other already vulnerable production regions; Chilean avocado production has been linked to climate-induced water stress and pressure on local communities [T1:1].\n\n**Climatic / Atmospheric**\n\n- Sending system — Mexico: Deforestation and land-use change reduce carbon storage and may alter local microclimates.\n- Receiving system — United States / Europe: Long-distance refrigerated transport and supply-chain logistics create embodied emissions.\n- Spillover systems: Market competition may shift emissions and land-use pressure among avocado-producing countries.\n\n**Geological / Geomorphological**\n\n- Sending system — Mexico: Orchard expansion on slopes can increase soil erosion, sedimentation, and long-term soil degradation.\n- Spillover systems: Similar geomorphological risks may emerge in competing production frontiers if global demand shifts expansion pressure elsewhere.\n\n---\n\n## 5. Cross-coupling Interactions\n\n- **Amplification across scales**: Telecoupled U.S. and international demand amplifies intracoupling within Mexican production landscapes by increasing orchard profitability, labor demand, irrigation investment, land conversion, and local infrastructure. The same telecoupled demand can also amplify pericoupling by pushing production into neighboring municipalities and watersheds.\n\n- **Spatial tradeoffs**: Receiving systems benefit from abundant avocados, retailer profits, and consumer choice, while producing regions may bear deforestation, water stress, labor precarity, and biodiversity loss. This matches the broader telecoupling insight that consumer landscapes can appear sustainable while environmental burdens are displaced to distant production landscapes [T1:5].\n\n- **Economic-social tradeoffs within Mexico**: Avocado exports can generate income and employment, but benefits may be unevenly distributed. Evidence from Mexico indicates that profits often concentrate with agribusinesses, while many workers face seasonal employment and rural/Indigenous communities may become precarious laborers in their own territories [T1:4].\n\n- **Pericoupling leakage**: If forest enforcement or water regulation becomes stricter in one producing municipality, expansion may move to adjacent jurisdictions with weaker enforcement. This would convert an intracoupling sustainability intervention into a pericoupled displacement problem.\n\n- **Telecoupled competition and spillovers**: Mexico’s role in global avocado trade affects other producers such as Chile and Peru. Conversely, climate shocks, water crises, or policy changes in Chile or Peru could increase demand for Mexican avocados, raising pressure on Mexican landscapes. Literature explicitly calls for more research on trade dynamics among Mexico, Chile, and Peru [T1:4].\n\n- **Coupling transformations**:\n - **Coupling**: Trade liberalization and U.S. demand strengthened the Mexico–U.S. avocado telecoupling after the 1994 free trade agreement [T1:2].\n - **Decoupling**: Border closures, phytosanitary bans, droughts, violence, labor disruptions, or consumer boycotts could weaken flows.\n - **Recoupling**: New certification systems, traceability, deforestation-free sourcing, or alternative trade routes could restore flows under modified governance conditions.\n\n- **Cascading interactions**: A drought, pest outbreak, regulatory change, or violence-related disruption in Michoacán could reduce Mexican supply, raise prices in U.S. markets, stimulate expansion in Jalisco or other Mexican regions, and increase production pressure in Chile, Peru, or Colombia. Conversely, shifts in U.S. consumer demand could rapidly cascade back into land-use decisions in Mexican production frontiers.\n\n---\n\n## 6. Research Gaps and Suggestions\n\n1. **Define the focal system more precisely.** The analysis should specify whether the focal system is all of Mexico, Michoacán, Jalisco, a watershed, a municipality, or a set of production frontiers. This choice determines which interactions are intracouplings, pericouplings, or telecouplings.\n\n2. **Quantify land-use change and deforestation linked specifically to avocado expansion.** Remote sensing can help distinguish avocado orchards from other crops and identify whether expansion occurs through forest clearing, agricultural substitution, or intensification [T1:W2], [T1:W3].\n\n3. **Measure virtual water and local water stress.** A stronger metacoupling analysis should estimate water use per exported avocado, identify affected watersheds and aquifers, and compare water burdens across domestic consumption, U.S. exports, and other export markets.\n\n4. **Disaggregate economic benefits and costs by agent.** Research should distinguish profits captured by agribusinesses, packers, landowners, smallholders, workers, Indigenous communities, and local governments. This is essential because existing evidence suggests strong inequalities in benefit distribution [T1:4].\n\n5. **Identify spillover systems more rigorously.** Competing producers such as Chile and Peru, adjacent Mexican production frontiers, and domestic food systems may experience significant indirect effects. These should be analyzed rather than treated as background context.\n\n6. **Trace governance and certification effects.** Future work should evaluate whether sustainability standards, deforestation-free sourcing, phytosanitary rules, or retailer codes actually reduce impacts or merely shift production pressure to less monitored regions.\n\n7. **Integrate flow-based governance.** Instead of governing only individual production sites, researchers should examine flows of avocados, capital, water, labor, information, and embodied emissions across the whole supply chain. This would better reveal who benefits, who bears costs, and where interventions could improve sustainability.",
  "research_gaps": [],
  "telecoupling": {
    "systems": [
      {
        "role": "sending",
        "system_scope": "distant",
        "name": "Mexican avocado-producing regions, especially Michoacán; potentially Jalisco and other export-oriented regions.",
        "human_subsystem": "Growers, workers, agribusinesses, packers, exporters, logistics firms, landowners, Indigenous and rural communities, Mexican government agencies.",
        "natural_subsystem": "Forests, agricultural land, watersheds, aquifers, soils, biodiversity, carbon stocks, avocado orchards.",
        "geographic_scope": "Major avocado-producing and exporting regions in Mexico, with Michoacán central and Jalisco increasingly relevant."
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "United States consumer and retail system.",
        "human_subsystem": "Consumers, supermarkets, restaurants, importers, distributors, marketing firms, food-service companies, U.S. regulators.",
        "natural_subsystem": "Urban and suburban food-consumption landscapes, domestic agricultural land indirectly spared or substituted, waste streams associated with food consumption.",
        "geographic_scope": "U.S. avocado-importing and consuming regions. The U.S.–Mexico avocado supply chain has been the subject of environmental impact research [T1:W2], [T1:W4]."
      },
      {
        "role": "receiving",
        "system_scope": "distant",
        "name": "European consumer markets, if included in the study.",
        "human_subsystem": "European consumers, retailers, importers, certification bodies, sustainability initiatives.",
        "natural_subsystem": "Consumption-related waste streams, domestic agricultural systems indirectly affected by imports.",
        "geographic_scope": "European avocado-importing countries. Available context includes a report on Mexican avocado production and trade with Europe, focusing on Jalisco and deforestation-free value-chain concerns [T1:W5]."
      },
      {
        "role": "spillover",
        "name": "Competing avocado-producing countries, especially Chile and Peru.",
        "human_subsystem": "Avocado growers, exporters, workers, governments, rural communities, and agribusinesses in competing producer countries.",
        "natural_subsystem": "Water-stressed avocado landscapes, biodiversity, soils, watersheds, and ecosystems affected by export-oriented production.",
        "geographic_scope": "Chile and Peru as major Latin American avocado exporters. Literature notes that Mexico, Chile, and Peru are among the world’s biggest avocado producers/exporters and that trade dynamics among them need further investigation [T1:3], [T1:4]."
      },
      {
        "role": "spillover",
        "name": "Non-avocado agricultural producers and food systems affected by market displacement.",
        "human_subsystem": "Farmers producing subsistence crops or alternative commodities, domestic consumers, local traders, food-security institutions.",
        "natural_subsystem": "Agricultural lands, forests, soils, and water resources that may be converted or spared depending on market shifts.",
        "geographic_scope": "Mexican domestic food-producing regions and other countries affected by changing avocado prices, supply competition, or substitution effects."
      },
      {
        "role": "spillover",
        "name": "Distant “apparent sustainability” landscapes in importing countries.",
        "human_subsystem": "Consumers, retailers, policymakers, urban sustainability advocates.",
        "natural_subsystem": "Importing-country landscapes where environmental burdens may appear lower because production impacts are displaced abroad.",
        "geographic_scope": "U.S. urban consumer landscapes and other affluent consumption regions. Dade et al. describe U.S. urban consumer landscapes and Michoacán avocado-producing landscapes as telecoupled through trade and externalized ecosystem-service burdens [T1:5]."
      }
    ],
    "flows": [
      {
        "category": "matter",
        "direction": "Mexico → United States.",
        "description": "Fresh avocados and avocado products move from Mexican production landscapes to U.S. consumers. The U.S.–Mexico avocado supply chain is documented as a major focus of environmental impact research [T1:W2], [T1:W4]."
      },
      {
        "category": "matter",
        "direction": "Mexico → Europe and other international markets.",
        "description": "Avocados exported from Mexico to European and other markets create additional long-distance commodity links; Jalisco has been examined in relation to Mexican avocado production and trade with Europe [T1:W5]."
      },
      {
        "category": "capital",
        "direction": "United States / international consumers and retailers → Mexican exporters, packers, growers, and intermediaries.",
        "description": "Payments for avocados, retail contracts, export revenues, and investment signals flow back to Mexico, financing orchard expansion, packing, logistics, and land markets."
      },
      {
        "category": "information",
        "direction": "United States / European retailers and consumers → Mexican producers and exporters.",
        "description": "Demand signals, quality standards, sustainability expectations, price information, phytosanitary standards, and branding narratives influence Mexican production decisions. Distant consumption and marketing are identified as drivers of land-use change and offstage environmental burdens [T1:5]."
      },
      {
        "category": "information",
        "direction": "Mexican producers, researchers, NGOs, and media → international consumers and policymakers.",
        "description": "Information about deforestation, water use, labor conditions, and sustainability risks circulates back to importing markets, potentially affecting purchasing behavior, certification, and policy."
      },
      {
        "category": "energy",
        "direction": "Mexico → United States / Europe as embodied energy in avocado exports.",
        "description": "Energy embedded in irrigation, fertilizer, agrochemicals, machinery, refrigeration, packing, and transport is transferred virtually through avocado consumption."
      },
      {
        "category": "matter",
        "direction": "Mexican watersheds → distant consumers as virtual water embedded in avocados.",
        "description": "Water used to produce exported avocados is effectively embodied in the commodity and consumed abroad, shifting water burdens to Mexican production regions."
      },
      {
        "category": "people",
        "direction": "Limited direct Mexico ↔ importing-country movement, but labor is indirectly mobilized by export demand.",
        "description": "Export demand stimulates local and regional labor flows within Mexico; international migration may also be indirectly affected through rural livelihood change, though this requires additional data."
      },
      {
        "category": "organisms",
        "direction": "Mexico → importing countries, potentially through fruit shipments.",
        "description": "Avocado fruits and packaging can carry phytosanitary risks, although regulated inspection systems are intended to prevent pest or pathogen movement."
      }
    ],
    "agents": [
      {
        "name": "**[Individuals / households] U.S. and European consumers**",
        "description": "Drive demand through dietary preferences, including avocado’s popularity as a healthy or “superfood” product [T1:3].",
        "level": "individuals / households"
      },
      {
        "name": "**[Individuals / households] Mexican growers and workers**",
        "description": "Produce and harvest avocados, experiencing both livelihood opportunities and environmental/social risks.",
        "level": "individuals / households"
      },
      {
        "name": "**[Firms / traders / corporations] Mexican exporters, packers, and agribusiness firms**",
        "description": "Organize production, packing, certification, and export logistics.",
        "level": "firms / traders / corporations"
      },
      {
        "name": "**[Firms / traders / corporations] U.S. importers, supermarkets, restaurants, and distributors**",
        "description": "Set quality standards, manage procurement, market avocados, and transmit price signals.",
        "level": "firms / traders / corporations"
      },
      {
        "name": "**[Firms / traders / corporations] International logistics and cold-chain firms**",
        "description": "Move avocados across long distances while maintaining quality.",
        "level": "firms / traders / corporations"
      },
      {
        "name": "**[Governments / policymakers] Mexican federal and state governments**",
        "description": "Regulate production, land use, water, labor, phytosanitary compliance, and export authorization.",
        "level": "governments / policymakers"
      },
      {
        "name": "**[Governments / policymakers] U.S. and European regulators**",
        "description": "Set import rules, food-safety standards, phytosanitary requirements, and sustainability-related market conditions.",
        "level": "governments / policymakers"
      },
      {
        "name": "**[Organizations / NGOs] Certification bodies, environmental NGOs, universities, and sustainability initiatives**",
        "description": "Assess deforestation, water use, labor risks, and supply-chain transparency.",
        "level": "organizations / NGOs"
      },
      {
        "name": "**[Non-human agents] Avocado trees, pests, pathogens, and pollinators**",
        "description": "Influence yield, pest risk, agrochemical use, and phytosanitary regulation.",
        "level": "non-human agents"
      }
    ],
    "causes": {
      "economic": [
        "International demand for avocados has increased significantly over the past two decades [T1:3].",
        "Avocado’s status as a globally traded commodity encourages export-oriented production and intensification.",
        "High prices and reliable access to U.S. markets create incentives for orchard expansion, land acquisition, and infrastructure investment.",
        "Supermarkets, landowner elites, local farmers, and global companies participate in complex production chains shaped by economic liberalization [T1:2]."
      ],
      "political / institutional": [
        "Trade liberalization and free trade agreements helped reorient production from domestic consumption toward international markets [T1:3].",
        "The U.S.–Mexico Free Trade Agreement established in 1994 amplified U.S. demand and helped transform Mexican avocado production into a globalized chain [T1:2].",
        "Import rules, phytosanitary agreements, and certification systems determine which Mexican regions and producers can access distant markets."
      ],
      "ecological / biological": [
        "Mexico’s suitable avocado-growing environments provide the ecological basis for export expansion.",
        "Avocado’s water demand links consumer markets to watershed stress in production regions.",
        "Pest and disease risks create strong phytosanitary governance connections between exporting and importing systems."
      ],
      "technological / infrastructural": [
        "Cold-chain logistics, roads, packing houses, and border inspection infrastructure make large-scale avocado trade possible.",
        "Remote sensing and supply-chain monitoring are increasingly used to assess environmental impacts and land-use change in Mexico [T1:W2], [T1:W3].",
        "Export certification systems transmit distant market requirements into local production practices."
      ],
      "cultural / social / demographic": [
        "Avocado consumption is shaped by dietary trends, health branding, and “superfood” narratives in Western consumer markets [T1:3].",
        "Marketing campaigns and retailer promotion create demand spikes and normalize year-round avocado consumption.",
        "Consumer concern about sustainability may generate pressure for deforestation-free or water-responsible supply chains."
      ],
      "hydrological": [
        "Virtual water demand from importing countries increases pressure on Mexican watersheds.",
        "Water availability and irrigation access condition which producers can participate in export markets."
      ],
      "climatic / atmospheric": [
        "Climate variability affects yields, irrigation needs, pest risk, and future production suitability.",
        "Long-distance transport and cold chains add embodied greenhouse-gas emissions to avocado trade."
      ],
      "geological / geomorphological": [
        "Soil suitability, slope, elevation, and volcanic landscapes shape where export-oriented avocado orchards expand.",
        "Terrain constraints influence erosion and infrastructure costs."
      ]
    },
    "effects": {
      "economic": [
        "Sending system — Mexico: Export revenue, employment, agribusiness growth, land-value increases, and infrastructure investment.",
        "Sending system — Mexico: Unequal benefit distribution, with evidence that Mexican avocado profits are concentrated among agribusinesses while many workers receive seasonal employment [T1:4].",
        "Receiving system — United States / Europe: Reliable avocado supply, consumer welfare, retailer profits, and food-service revenues.",
        "Spillover systems — Chile and Peru: Competition with Mexico can affect prices, market access, production decisions, and incentives for expansion. The trade dynamics among Mexico, Chile, and Peru are identified as an important research need [T1:4]."
      ],
      "political / institutional": [
        "Sending system — Mexico: Increased need for land-use enforcement, water governance, labor oversight, anti-deforestation rules, and traceability.",
        "Receiving system — United States / Europe: Sustainability concerns may lead to retailer standards, certification, or deforestation-free procurement requirements.",
        "Spillover systems — competing producers: Countries such as Chile and Peru may adjust export strategies, regulations, or sustainability claims in response to Mexican market dominance."
      ],
      "ecological / biological": [
        "Sending system — Mexico: Deforestation, habitat fragmentation, biodiversity loss, and reduced regulating ecosystem services may occur where orchard expansion replaces forest or diversified agriculture. Dade et al. identify deforestation, water scarcity, and lost regulating services as telecoupled burdens associated with U.S. avocado consumption and Michoacán production [T1:5].",
        "Receiving system — United States / Europe: Environmental burdens are partly displaced abroad, creating apparent sustainability gains in consumer landscapes while impacts occur in production landscapes [T1:5].",
        "Spillover systems — Chile and Peru: Competitive pressures may encourage expansion or intensification in other avocado-producing regions, with associated water and biodiversity impacts. Chilean avocado production has been linked to water stress, biodiversity pressure, and local community impacts under international demand [T1:1]."
      ],
      "technological / infrastructural": [
        "Sending system — Mexico: Expansion of packing houses, irrigation systems, roads, cold-chain infrastructure, and monitoring systems.",
        "Receiving system — United States / Europe: Improved distribution and retail systems support year-round avocado availability.",
        "Spillover systems: Competing producer countries may invest in similar export infrastructure to maintain market share."
      ],
      "cultural / social / demographic": [
        "Sending system — Mexico: Rural communities may shift from subsistence or diversified livelihoods toward export-oriented production and wage labor.",
        "Sending system — Mexico: Indigenous and rural communities may become precarious workers in landscapes where they previously held stronger livelihood or territorial roles [T1:4].",
        "Receiving system — United States / Europe: Avocado consumption becomes normalized as part of health-oriented diets, brunch culture, and “superfood” consumption.",
        "Spillover systems: Global avocado demand may reshape rural labor relations in other producing countries."
      ],
      "hydrological": [
        "Sending system — Mexico: Virtual water exports can intensify local water stress, particularly where irrigation expands or groundwater governance is weak.",
        "Receiving system — United States / Europe: Consumers receive avocado benefits without directly experiencing production-region water scarcity.",
        "Spillover systems — Chile and Peru: Competitive expansion may intensify water stress in other already vulnerable production regions; Chilean avocado production has been linked to climate-induced water stress and pressure on local communities [T1:1]."
      ],
      "climatic / atmospheric": [
        "Sending system — Mexico: Deforestation and land-use change reduce carbon storage and may alter local microclimates.",
        "Receiving system — United States / Europe: Long-distance refrigerated transport and supply-chain logistics create embodied emissions.",
        "Spillover systems: Market competition may shift emissions and land-use pressure among avocado-producing countries."
      ],
      "geological / geomorphological": [
        "Sending system — Mexico: Orchard expansion on slopes can increase soil erosion, sedimentation, and long-term soil degradation.",
        "Spillover systems: Similar geomorphological risks may emerge in competing production frontiers if global demand shifts expansion pressure elsewhere."
      ]
    }
  }
}
```

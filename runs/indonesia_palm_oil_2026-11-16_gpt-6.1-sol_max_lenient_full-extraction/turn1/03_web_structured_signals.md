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
        "W2"
      ],
      "reason": "Identified as Indonesia's largest palm oil export destination."
    },
    {
      "country": "CHE",
      "kind": "direct",
      "confidence": 0.99,
      "evidence": [
        "W21"
      ],
      "reason": "Documented historical importer: received 21 tonnes of qualifying Indonesian palm oil in 2022, although no such preferential imports were recorded in 2023–2024."
    }
  ],
  "spillover_systems": [
    {
      "country": "MYS",
      "kind": "proxy",
      "confidence": 0.95,
      "evidence": [
        "W1"
      ],
      "reason": "Competing-exporter context: Malaysia and Indonesia jointly supplied around 90% of global refined palm oil exports during 2019–2023; no bilateral spillover is quantified."
    }
  ],
  "flows": [
    {
      "category": "matter",
      "source_country": "IDN",
      "target_country": "IND",
      "direction": "Indonesia → India",
      "description": "Palm oil exports from Indonesia to India, identified as Indonesia's largest export destination.",
      "kind": "direct",
      "confidence": 0.99,
      "evidence": [
        "W2"
      ]
    },
    {
      "category": "matter",
      "source_country": "IDN",
      "target_country": "CHE",
      "direction": "Indonesia → Switzerland",
      "description": "Switzerland imported 21 tonnes of qualifying Indonesian palm oil under sustainability-linked preferences in 2022; no such preferential imports were recorded in 2023–2024.",
      "kind": "direct",
      "confidence": 0.99,
      "evidence": [
        "W21"
      ]
    }
  ],
  "evidence_cards": [
    {
      "source_id": "W1",
      "claims_supported": [
        "Indonesia and Malaysia supplied around 90% of global refined palm oil exports during 2019–2023.",
        "Palm oil expansion is linked to deforestation, biodiversity loss, emissions and labour concerns.",
        "Smallholders supply an estimated 40% of global palm oil but face financial and traceability barriers."
      ],
      "relevance_score": 0.94,
      "source_type": "international_organization"
    },
    {
      "source_id": "W2",
      "claims_supported": [
        "India was Indonesia's largest palm oil export destination.",
        "Domestic demand accounted for 67% of attributed deforestation exposure, which is not equivalent to annual clearing.",
        "Peat subsidence released an estimated 148 million tonnes CO2e in 2024.",
        "Almost 17% of exports lacked identifiable shipment records."
      ],
      "relevance_score": 1.0,
      "source_type": "other"
    },
    {
      "source_id": "W3",
      "claims_supported": [
        "A ten-percentage-point increase in district oil palm land share corresponded to six percentage points faster poverty reduction.",
        "Higher agricultural labour returns explained most estimated household expenditure gains.",
        "Expansion also increased deforestation, fires, certain health problems and conflict."
      ],
      "relevance_score": 0.97,
      "source_type": "academic"
    },
    {
      "source_id": "W4",
      "claims_supported": [
        "Socioeconomic gains were concentrated in market-oriented Kalimantan villages with low or moderate forest cover.",
        "Remote, heavily forested subsistence villages experienced worse socioeconomic outcomes after plantation development.",
        "Oil-palm villages generally showed slower social and environmental improvement than comparison villages."
      ],
      "relevance_score": 0.94,
      "source_type": "academic"
    },
    {
      "source_id": "W5",
      "claims_supported": [
        "Across ten investigated partnerships, estimated annual receipts averaged about US$338 per two-hectare participant.",
        "Some investigated communities retained debts of US$2.5–18.5 million a decade after plantation establishment.",
        "Contracts frequently gave companies extensive control over land, loans, costs and profit calculations."
      ],
      "relevance_score": 0.93,
      "source_type": "news"
    },
    {
      "source_id": "W6",
      "claims_supported": [
        "The West Kalimantan investigation documented plantation expansion without genuine consultation or adequate compensation.",
        "Residents reported farmland losses, altered water conditions and livelihood hardship.",
        "The report connects peat drainage and subsidence with carbon releases."
      ],
      "relevance_score": 0.93,
      "source_type": "ngo"
    },
    {
      "source_id": "W7",
      "claims_supported": [
        "The 2016 investigation interviewed 120 workers on Indonesian plantations supplying Wilmar.",
        "Documented abuses included hazardous child labour, forced labour and discrimination against women.",
        "Supply-chain research linked plantation output to nine global firms."
      ],
      "relevance_score": 0.93,
      "source_type": "ngo"
    },
    {
      "source_id": "W8",
      "claims_supported": [
        "Indonesia's mapped oil palm plantation area reached 16.24 million hectares in 2019.",
        "About 3.09 million hectares of primary forest lost during 2001–2019 had become oil palm by 2019.",
        "Lower palm oil prices were associated with reduced industrial expansion and forest loss."
      ],
      "relevance_score": 0.97,
      "source_type": "academic"
    },
    {
      "source_id": "W9",
      "claims_supported": [
        "By 2020, about 7.8 million hectares of Indonesia's cleared primary forest had become oil palm.",
        "Approximately 8.8 million hectares of cleared forest remained unused in 2020.",
        "Delayed planting complicates commodity attribution and deforestation-free supply-chain verification."
      ],
      "relevance_score": 0.9,
      "source_type": "academic"
    },
    {
      "source_id": "W10",
      "claims_supported": [
        "The 2018 brief reported that palm oil production affected at least 193 threatened species globally.",
        "At least half of Borneo's 2005–2015 deforestation was linked to oil palm development.",
        "Replacing palm oil with lower-yielding oils could displace land pressure and biodiversity damage elsewhere."
      ],
      "relevance_score": 0.86,
      "source_type": "international_organization"
    },
    {
      "source_id": "W11",
      "claims_supported": [
        "Field greenhouse gas measurements came from North Selangor, Malaysia, not Indonesia.",
        "Estimated peat-conversion emission factors ranged from 70 to 117 tonnes CO2e per hectare annually.",
        "Regional extrapolation suggested 16.6–27.9% of combined Indonesian and Malaysian national emissions under the study's assumptions."
      ],
      "relevance_score": 0.78,
      "source_type": "academic"
    },
    {
      "source_id": "W12",
      "claims_supported": [
        "First-rotation biodiesel after forest conversion on mineral soils had estimated emissions of about 216 g CO2e per megajoule.",
        "That first-rotation scenario provided no greenhouse gas savings relative to fossil diesel.",
        "The modelled baseline second-rotation scenario achieved approximately 65% emissions savings."
      ],
      "relevance_score": 0.97,
      "source_type": "academic"
    },
    {
      "source_id": "W13",
      "claims_supported": [
        "Indonesia's 2015 fires caused estimated losses and damages of US$16.1 billion, equivalent to 1.9% of GDP.",
        "The report links peat drainage and conversion, driven largely by palm oil production, with more intense haze.",
        "Not all fires were set for oil palm, and the estimate excludes long-term health consequences and many regional costs."
      ],
      "relevance_score": 0.92,
      "source_type": "international_organization"
    },
    {
      "source_id": "W14",
      "claims_supported": [
        "Low stream nutrient concentrations indicated no eutrophication under sampled conditions in the studied mature-plantation landscape.",
        "Loamy-sand uplands exported more dissolved inorganic nitrogen and total phosphorus than loamy lowlands.",
        "Organic fertilization reduced nutrient fluxes relative to mineral fertilization."
      ],
      "relevance_score": 0.9,
      "source_type": "academic"
    },
    {
      "source_id": "W15",
      "claims_supported": [
        "Industrial estates averaged roughly twice smallholder yields in the Sumatran sample.",
        "Lower management intensity improved ecological outcomes without reducing yields in the studied systems.",
        "Surrounding forest cover generally supported plantation biodiversity and ecosystem functioning."
      ],
      "relevance_score": 0.96,
      "source_type": "academic"
    },
    {
      "source_id": "W16",
      "claims_supported": [
        "Estimated actual yields were 62% of attainable yields on large plantations and 53% on smallholder plantations.",
        "Modelled intensification with constrained expansion could spare 2.6 million hectares of forest and peatland.",
        "The scenario could avoid 732 million tonnes CO2e relative to continuation of historical trends."
      ],
      "relevance_score": 0.96,
      "source_type": "academic"
    },
    {
      "source_id": "W17",
      "claims_supported": [
        "Indonesia doubled its replanting subsidy to 60 million rupiah per hectare in 2024.",
        "Scenario analysis indicated a seven-to-eight-year income recovery period after replanting.",
        "Land documentation, forest-zone status and farmer-group requirements create access barriers."
      ],
      "relevance_score": 0.95,
      "source_type": "academic"
    },
    {
      "source_id": "W18",
      "claims_supported": [
        "The historical evaluation estimated that RSPO certification reduced deforestation by 33%.",
        "No causal effect was detected on peatland forest loss or active fire rates.",
        "Certified areas contained less than 1% of forest remaining inside Indonesian oil palm plantations by 2015."
      ],
      "relevance_score": 0.98,
      "source_type": "academic"
    },
    {
      "source_id": "W19",
      "claims_supported": [
        "Commitment-linked concessions had annual deforestation below 1% during the evaluated period.",
        "No additional deforestation reduction from commitments was detected during 2018–2020.",
        "Comparable concessions without commitments experienced similar reductions."
      ],
      "relevance_score": 0.98,
      "source_type": "academic"
    },
    {
      "source_id": "W20",
      "claims_supported": [
        "A permit review assessed 55 oil palm companies in Papua.",
        "Authorities recommended revoking 33 expired licences and corrective action for 22 other companies.",
        "Licence revocation does not automatically restore Indigenous land rights."
      ],
      "relevance_score": 0.94,
      "source_type": "ngo"
    },
    {
      "source_id": "W21",
      "claims_supported": [
        "Switzerland imported 21 tonnes of sustainability-qualifying Indonesian palm oil in 2022.",
        "No such preferential imports were recorded in Switzerland in 2023–2024.",
        "Norway and Iceland recorded no Indonesian palm oil imports through 2024.",
        "Sustainability-linked tariff preferences address forest protection, peat drainage and pollution."
      ],
      "relevance_score": 0.98,
      "source_type": "international_organization"
    },
    {
      "source_id": "W22",
      "claims_supported": [
        "EU rules require covered oil palm products to be deforestation-free, legally produced and supported by due diligence.",
        "The deforestation cutoff is December 31, 2020, and traceability includes supplying-plot information.",
        "The consolidated text gives December 30, 2026 as the main application date, with later dates for specified smaller operators."
      ],
      "relevance_score": 0.95,
      "source_type": "government"
    },
    {
      "source_id": "W23",
      "claims_supported": [
        "Independent smallholders produced about one-third of regional output but supplied only 7% of certified mill inputs.",
        "Only 36% of certified mills had ever purchased from independent farmers.",
        "The study found no evidence that independent farmers exited markets after mills became certified."
      ],
      "relevance_score": 0.95,
      "source_type": "academic"
    },
    {
      "source_id": "W24",
      "claims_supported": [
        "Modelled higher biodiesel blends lost more palm oil export revenue than they saved on diesel imports.",
        "Higher-blend scenarios widened the current-account deficit and reduced real GDP and household consumption.",
        "Additional forest and peat conversion could undermine biodiesel emissions benefits."
      ],
      "relevance_score": 0.96,
      "source_type": "government"
    },
    {
      "source_id": "W25",
      "claims_supported": [
        "The report records implementation of B50 in July 2026.",
        "It forecasts 17.5 billion litres of biodiesel consumption and about 16.19 million tonnes of crude palm oil feedstock use in 2026.",
        "Reported biodiesel exports effectively disappeared as domestic consumption absorbed output.",
        "Reported palm oil mill effluent exports fell from about 1.5 million tonnes in 2024 to 22,000 tonnes in 2025."
      ],
      "relevance_score": 0.95,
      "source_type": "government"
    }
  ],
  "suggested_followup_queries": [
    "Indonesia India palm oil import volumes traceability 2026",
    "Indonesian palm oil European Union import volumes traceability",
    "Indonesian palm oil transboundary haze affected countries attribution",
    "Indonesia palm oil replanting finance household income outcomes",
    "Indonesian palm oil plantation labour rights monitoring 2026"
  ]
}
```

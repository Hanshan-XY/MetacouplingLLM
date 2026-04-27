"""
Curated case-study examples from the telecoupling/metacoupling literature.

Each example demonstrates how the framework components map to a real-world
study, providing concrete illustrations the LLM can reference when advising
researchers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FrameworkExample:
    """A curated example of the telecoupling/metacoupling framework applied."""

    title: str
    source: str
    domain: str
    coupling_type: str  # "telecoupling" | "metacoupling"
    description: str
    systems: dict[str, str | dict[str, str]]  # role -> description or nested dict
    flows: list[dict[str, str]]  # each: {category, direction, description}
    agents: list[dict[str, str]]  # each: {level, name, description}
    causes: list[dict[str, str]]  # each: {category, description}
    effects: list[dict[str, str]]  # each: {system, type, description}
    keywords: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Example 1: Soybean Trade (Liu et al., 2013)
# ---------------------------------------------------------------------------

SOYBEAN_TRADE = FrameworkExample(
    title="International Soybean Trade: Brazil to China",
    source="Liu et al. (2013) – Framing Sustainability in a Telecoupled World",
    domain="agriculture, trade, land use",
    coupling_type="telecoupling",
    description=(
        "Large-scale soybean trade between Brazil (sending) and China "
        "(receiving), with spillover effects on other soy-producing and "
        "consuming nations. This telecoupling drives land-use change in "
        "Brazil and food supply dynamics in China."
    ),
    systems={
        "sending": {
            "name": "Brazil",
            "human_subsystem": (
                "Soybean farmers, agribusiness companies, government "
                "land-use policy makers, export agencies."
            ),
            "natural_subsystem": (
                "Tropical forests, Cerrado grasslands, biodiversity, "
                "soil and water resources in the Amazon and Cerrado biomes."
            ),
            "geographic_scope": "Amazon and Cerrado biomes, Brazil.",
        },
        "receiving": {
            "name": "China",
            "human_subsystem": (
                "Food processing industry, livestock sector, consumers, "
                "trade policy makers."
            ),
            "natural_subsystem": (
                "Agricultural land spared from soybean cultivation, "
                "local ecosystems."
            ),
            "geographic_scope": "Soybean-importing and consuming regions of China.",
        },
        "spillover": {
            "name": "Other soy-producing and importing nations",
            "human_subsystem": (
                "Soybean farmers in Argentina and USA affected by market "
                "competition; consumers in other importing nations facing "
                "price changes."
            ),
            "natural_subsystem": (
                "Agricultural and forest ecosystems in competing regions "
                "experiencing land-use pressure; ecosystems in countries "
                "neighboring Brazil affected by cross-border deforestation."
            ),
            "geographic_scope": "Argentina, USA, and other major soy-producing regions.",
        },
    },
    flows=[
        {
            "category": "matter",
            "direction": "Brazil → China",
            "description": "Soybeans and soybean products exported from Brazil to China",
        },
        {
            "category": "capital",
            "direction": "China → Brazil",
            "description": "Payment for soybeans; Chinese investment in Brazilian agriculture",
        },
        {
            "category": "information",
            "direction": "bidirectional",
            "description": "Market information, price signals, trade regulations",
        },
    ],
    agents=[
        {"level": "individuals / households", "name": "Brazilian soybean farmers", "description": "Smallholder and commercial farmers in sending system"},
        {"level": "firms / traders / corporations", "name": "Brazilian agribusiness companies", "description": "Large-scale soy producers and exporters in sending system"},
        {"level": "firms / traders / corporations", "name": "Chinese food processing and livestock companies", "description": "Import and process soybeans in receiving system"},
        {"level": "governments / policymakers", "name": "Brazilian and Chinese government trade agencies", "description": "Regulate and facilitate bilateral trade"},
        {"level": "firms / traders / corporations", "name": "International commodity traders and brokers", "description": "Intermediaries facilitating global soybean trade"},
        {"level": "individuals / households", "name": "Consumers in China", "description": "Drive demand for soybean products in receiving system"},
    ],
    causes=[
        {"category": "economic", "description": "Growing demand for soybeans in China for animal feed and cooking oil"},
        {"category": "economic", "description": "Competitive pricing of Brazilian soybeans on the global market"},
        {"category": "economic", "description": "Rapid economic growth and rising meat consumption in China"},
        {"category": "ecological / biological", "description": "Availability of arable land in Brazil for soybean expansion in Cerrado and Amazon biomes"},
        {"category": "political / institutional", "description": "International trade liberalization policies facilitating bilateral commodity trade"},
    ],
    effects=[
        {"system": "sending", "type": "ecological / biological", "description": "Deforestation and habitat loss in the Amazon and Cerrado (sending system)"},
        {"system": "sending", "type": "climatic / atmospheric", "description": "Carbon emissions from land-use change and soil degradation (sending system)"},
        {"system": "sending", "type": "ecological / biological", "description": "Loss of biodiversity due to conversion of native habitats to monoculture (sending system)"},
        {"system": "sending", "type": "hydrological", "description": "Water pollution from agrochemical runoff and altered watershed hydrology (sending system)"},
        {"system": "sending", "type": "economic", "description": "Economic growth in agricultural sector; land conflicts (sending system)"},
        {"system": "receiving", "type": "economic", "description": "Affordable animal feed supporting livestock industry growth (receiving system)"},
        {"system": "receiving", "type": "ecological / biological", "description": "Reduced pressure on domestic land for soybean cultivation (receiving system)"},
        {"system": "spillover", "type": "economic", "description": "Price competition affecting soybean farmers in Argentina, USA (spillover system)"},
        {"system": "spillover", "type": "ecological / biological", "description": "Potential shifts in deforestation to other regions (spillover system)"},
        {"system": "spillover", "type": "political / institutional", "description": "Trade policy adjustments in competing nations responding to market shifts (spillover system)"},
    ],
    keywords=["soybean", "trade", "agriculture", "deforestation", "land use",
              "commodity", "food", "Brazil", "China"],
)


# ---------------------------------------------------------------------------
# Example 2: Red Imported Fire Ant Invasion (Liu et al., 2013)
# ---------------------------------------------------------------------------

RIFA_INVASION = FrameworkExample(
    title="Red Imported Fire Ant (RIFA) Invasion: USA to China",
    source="Liu et al. (2013) – Framing Sustainability in a Telecoupled World",
    domain="species invasion, trade, biosecurity",
    coupling_type="telecoupling",
    description=(
        "Invasion of red imported fire ants from the United States to China "
        "via international trade and shipping. The sending system (USA) "
        "harbors established RIFA populations that are inadvertently "
        "transported to the receiving system (China) through cargo."
    ),
    systems={
        "sending": {
            "name": "United States",
            "human_subsystem": (
                "Port operators, exporters, biosecurity agencies in "
                "the southeastern US."
            ),
            "natural_subsystem": (
                "Ecosystems where RIFA is established, including "
                "disturbed habitats and agricultural lands."
            ),
            "geographic_scope": "Southeastern United States port regions.",
        },
        "receiving": {
            "name": "China",
            "human_subsystem": (
                "Importers, quarantine agencies, affected communities "
                "near ports receiving international cargo."
            ),
            "natural_subsystem": (
                "Native ecosystems vulnerable to RIFA invasion, "
                "agricultural lands, and urban green spaces."
            ),
            "geographic_scope": "Chinese port cities and surrounding regions.",
        },
        "spillover": {
            "name": "Other countries along shipping routes",
            "human_subsystem": (
                "Port operators and biosecurity agencies in countries "
                "along shared shipping routes, communities affected by "
                "secondary RIFA spread."
            ),
            "natural_subsystem": (
                "Native ecosystems in transit and neighboring countries "
                "potentially exposed to RIFA through similar trade pathways."
            ),
            "geographic_scope": (
                "Southeast Asian nations, Pacific island countries, "
                "and other trading partners along shared shipping routes."
            ),
        },
    },
    flows=[
        {
            "category": "organisms",
            "direction": "USA → China",
            "description": "Unintentional transport of RIFA colonies in shipping containers and cargo",
        },
        {
            "category": "matter",
            "direction": "USA → China",
            "description": "Trade goods (legitimate cargo) that serve as vectors for ant transport",
        },
        {
            "category": "information",
            "direction": "bidirectional",
            "description": "Biosecurity protocols, invasion alerts, scientific knowledge on RIFA management",
        },
    ],
    agents=[
        {"level": "firms / traders / corporations", "name": "Shipping companies and port operators", "description": "Intermediaries inadvertently transporting RIFA in cargo"},
        {"level": "firms / traders / corporations", "name": "Exporters and importers of goods", "description": "Commercial entities in sending and receiving systems"},
        {"level": "governments / policymakers", "name": "National biosecurity and quarantine agencies", "description": "Government agencies managing invasive species risk in both systems"},
        {"level": "organizations / NGOs", "name": "Scientists studying invasive species", "description": "Researchers providing management knowledge in both systems"},
        {"level": "non-human agents", "name": "Red imported fire ants", "description": "Invasive organisms whose survival and spread actively shape receiving-system impacts"},
    ],
    causes=[
        {"category": "ecological / biological", "description": "RIFA colonies established in disturbed habitats near ports, facilitating inadvertent loading onto cargo containers"},
        {"category": "political / institutional", "description": "Inadequate biosecurity inspection protocols for shipped goods at ports"},
        {"category": "economic", "description": "Growth of international trade volume between the US and China"},
        {"category": "economic", "description": "Globalization of supply chains increasing shipping frequency and vector pathways"},
    ],
    effects=[
        {"system": "receiving", "type": "ecological / biological", "description": "Displacement of native ant species and reduction in arthropod biodiversity (receiving system)"},
        {"system": "receiving", "type": "ecological / biological", "description": "Ecological damage to native ecosystems and disruption of food webs (receiving system)"},
        {"system": "receiving", "type": "economic", "description": "Agricultural damage, public health risks (stings), control costs (receiving system)"},
        {"system": "sending", "type": "political / institutional", "description": "Potential trade restrictions or increased inspection requirements (sending system)"},
        {"system": "sending", "type": "economic", "description": "Increased biosecurity costs for exporters (sending system)"},
        {"system": "spillover", "type": "ecological / biological", "description": "Risk of secondary spread to neighboring countries from China (spillover system)"},
    ],
    keywords=["invasive", "species", "fire ant", "RIFA", "biosecurity", "trade",
              "shipping", "pest", "biological invasion"],
)


# ---------------------------------------------------------------------------
# Example 3: Beijing Urban Water Supply (Deines et al., 2016)
# ---------------------------------------------------------------------------

BEIJING_WATER = FrameworkExample(
    title="Beijing Urban Water Supply Telecouplings",
    source="Deines et al. (2016) – Telecoupling in Urban Water Systems",
    domain="water resources, urban, infrastructure",
    coupling_type="telecoupling",
    description=(
        "Beijing's water supply involves three distinct telecouplings: "
        "(1) the South-to-North Water Transfer Project (SNWTP) physically "
        "moving water from the Yangtze to Beijing; (2) virtual water imports "
        "through agricultural and industrial goods; and (3) payments for "
        "ecosystem services (PES) to upstream communities for water quality "
        "protection."
    ),
    systems={
        "sending": {
            "name": "Source water regions",
            "human_subsystem": (
                "Farmers, local governments, dam operators in the "
                "Yangtze River basin and upstream watersheds."
            ),
            "natural_subsystem": (
                "River ecosystems, watersheds, agricultural landscapes "
                "in the Yangtze basin and upstream areas."
            ),
            "geographic_scope": (
                "Yangtze River basin (SNWTP source), agricultural regions "
                "providing virtual water, upstream watersheds."
            ),
        },
        "receiving": {
            "name": "Beijing",
            "human_subsystem": (
                "Urban residents, municipal government, water utilities, "
                "industries in Beijing metropolitan area."
            ),
            "natural_subsystem": (
                "Local water resources, urban ecosystems, and aquifers "
                "under the Beijing metropolitan area."
            ),
            "geographic_scope": "Beijing metropolitan area.",
        },
        "spillover": {
            "name": "Communities along the SNWTP route and downstream areas",
            "human_subsystem": (
                "Displaced communities along SNWTP construction route, "
                "farmers and residents in regions competing for the same "
                "water resources."
            ),
            "natural_subsystem": (
                "River ecosystems downstream of source diversions with "
                "reduced flows, wetlands and aquatic habitats along the "
                "SNWTP corridor."
            ),
            "geographic_scope": (
                "Communities along the SNWTP route, downstream areas of "
                "source rivers, regions competing for shared water resources."
            ),
        },
    },
    flows=[
        {
            "category": "matter",
            "direction": "Yangtze basin → Beijing",
            "description": "Physical water transferred via the South-to-North Water Transfer Project",
        },
        {
            "category": "matter",
            "direction": "Agricultural regions → Beijing",
            "description": "Virtual water embedded in imported agricultural and industrial goods",
        },
        {
            "category": "capital",
            "direction": "Beijing → upstream communities",
            "description": "Payments for ecosystem services to protect water quality upstream",
        },
        {
            "category": "capital",
            "direction": "Beijing → source regions",
            "description": "Government investment in SNWTP infrastructure",
        },
        {
            "category": "information",
            "direction": "bidirectional",
            "description": "Water quality monitoring data, management policies, scientific assessments",
        },
    ],
    agents=[
        {"level": "governments / policymakers", "name": "Beijing municipal government and water utilities", "description": "Managing water demand in receiving system"},
        {"level": "governments / policymakers", "name": "Central government agencies managing SNWTP", "description": "Intermediary planning and operating water transfer infrastructure"},
        {"level": "individuals / households", "name": "Upstream farming communities participating in PES", "description": "Households in sending system receiving payments for ecosystem services"},
        {"level": "individuals / households", "name": "Agricultural producers in virtual water source regions", "description": "Farmers in sending system producing water-intensive goods"},
        {"level": "individuals / households", "name": "Displaced communities along SNWTP route", "description": "Households in spillover system affected by infrastructure construction"},
    ],
    causes=[
        {"category": "hydrological", "description": "Chronic water scarcity in Beijing relative to demand due to limited local water resources"},
        {"category": "political / institutional", "description": "Government decision to construct SNWTP infrastructure as a national priority project"},
        {"category": "cultural / social / demographic", "description": "Rapid urbanization and population growth in Beijing increasing water demand"},
        {"category": "hydrological", "description": "Uneven spatial distribution of water resources across China (south water-rich, north water-poor)"},
        {"category": "economic", "description": "Increasing per-capita water consumption driven by economic growth and rising living standards"},
    ],
    effects=[
        {"system": "receiving", "type": "hydrological", "description": "Improved water supply security for Beijing residents and industries (receiving system)"},
        {"system": "receiving", "type": "economic", "description": "Economic benefits from reliable water supply for urban industries (receiving system)"},
        {"system": "sending", "type": "hydrological", "description": "Reduced river flows and altered hydrology in source basins (sending system)"},
        {"system": "sending", "type": "ecological / biological", "description": "Altered riparian and aquatic ecosystems downstream of diversions (sending system)"},
        {"system": "sending", "type": "economic", "description": "Income from PES payments; restrictions on land use for water protection (sending system)"},
        {"system": "spillover", "type": "cultural / social / demographic", "description": "Displacement of communities for SNWTP construction; compensation disputes (spillover system)"},
        {"system": "spillover", "type": "hydrological", "description": "Reduced water availability for downstream users of source rivers (spillover system)"},
        {"system": "spillover", "type": "political / institutional", "description": "Inter-provincial water allocation conflicts and governance challenges (spillover system)"},
    ],
    keywords=["water", "urban", "transfer", "infrastructure", "SNWTP", "virtual water",
              "PES", "ecosystem services", "scarcity", "Beijing"],
)


# ---------------------------------------------------------------------------
# Example 4: Panda Conservation and SDGs (Zhao et al., 2021)
# ---------------------------------------------------------------------------

PANDA_SDGS = FrameworkExample(
    title="Panda Conservation Metacoupling and SDGs",
    source="Zhao et al. (2021) – Metacoupling of SDGs across Panda Reserves",
    domain="conservation, biodiversity, sustainable development goals",
    coupling_type="metacoupling",
    description=(
        "Metacoupling analysis across 67 giant panda nature reserves in China, "
        "examining how tourism flows and panda loan programs create "
        "socioeconomic-environmental interactions that generate synergies and "
        "tradeoffs among the UN Sustainable Development Goals (SDGs). "
        "This demonstrates metacoupling encompassing intracoupling (within "
        "reserves), pericoupling (between adjacent reserves), and telecoupling "
        "(between distant reserves and international partners)."
    ),
    systems={
        "sending": {
            "name": "Panda nature reserves (e.g., Wolong)",
            "human_subsystem": (
                "Reserve management authorities, local communities, "
                "tourism operators."
            ),
            "natural_subsystem": (
                "Giant panda habitat, bamboo forests, montane "
                "biodiversity."
            ),
            "geographic_scope": "67 giant panda nature reserves across China.",
        },
        "receiving": {
            "name": "Zoos worldwide and tourist source cities",
            "human_subsystem": (
                "Zoo management, urban populations, conservation donors, "
                "tourists."
            ),
            "natural_subsystem": (
                "Urban environments, captive breeding facilities."
            ),
            "geographic_scope": (
                "International and domestic zoos; tourist source cities "
                "sending visitors to reserves."
            ),
        },
        "spillover": {
            "name": "Adjacent communities and conservation community",
            "human_subsystem": (
                "Communities adjacent to reserves affected by tourism "
                "overflow, regions experiencing changes in conservation "
                "funding allocation, global conservation community "
                "influenced by panda diplomacy."
            ),
            "natural_subsystem": (
                "Non-protected ecosystems adjacent to reserves facing "
                "tourism development pressures, habitats in areas where "
                "conservation funding was redirected."
            ),
            "geographic_scope": (
                "Areas surrounding panda reserves, regions with "
                "competing conservation priorities."
            ),
        },
    },
    flows=[
        {
            "category": "organisms",
            "direction": "Reserves → zoos worldwide",
            "description": "Giant pandas loaned to international and domestic zoos",
        },
        {
            "category": "people",
            "direction": "Urban areas → reserves",
            "description": "Tourist flows from cities to panda nature reserves",
        },
        {
            "category": "capital",
            "direction": "Zoos/tourists → reserves",
            "description": "Panda loan fees and tourism revenue supporting conservation",
        },
        {
            "category": "information",
            "direction": "bidirectional",
            "description": "Conservation knowledge exchange, scientific research collaboration",
        },
    ],
    agents=[
        {"level": "governments / policymakers", "name": "Reserve management authorities", "description": "Manage panda reserves in sending system"},
        {"level": "organizations / NGOs", "name": "International and domestic zoos", "description": "Receive loaned pandas in receiving system"},
        {"level": "individuals / households", "name": "Local communities near reserves", "description": "Households in sending/spillover systems affected by tourism"},
        {"level": "individuals / households", "name": "Tourists", "description": "Visitors to panda reserves in receiving/intermediary role"},
        {"level": "firms / traders / corporations", "name": "Tourism operators", "description": "Commercial tourism services facilitating visitor flows"},
        {"level": "governments / policymakers", "name": "Chinese government conservation agencies", "description": "Intermediary agencies designing panda loan and conservation policies"},
        {"level": "organizations / NGOs", "name": "International conservation organizations", "description": "Intermediary NGOs supporting global conservation efforts"},
    ],
    causes=[
        {"category": "political / institutional", "description": "Panda loan agreements between Chinese government and foreign zoos as diplomatic instruments"},
        {"category": "cultural / social / demographic", "description": "Growing domestic and international ecotourism demand driven by cultural fascination with pandas"},
        {"category": "cultural / social / demographic", "description": "Global interest in giant panda conservation as a flagship and culturally iconic species"},
        {"category": "political / institutional", "description": "Chinese policy to use panda diplomacy for international relations"},
        {"category": "economic", "description": "Need for sustainable funding mechanisms for nature reserve management"},
    ],
    effects=[
        {"system": "sending", "type": "economic", "description": "Tourism revenue and panda loan fees supporting local livelihoods (SDG 1, 8) (sending system)"},
        {"system": "sending", "type": "ecological / biological", "description": "Habitat protection through conservation programs (SDG 15); but tourism pressure on ecosystems (sending system)"},
        {"system": "sending", "type": "ecological / biological", "description": "Panda population recovery from breeding and habitat protection programs (sending system)"},
        {"system": "receiving", "type": "cultural / social / demographic", "description": "Educational and recreational value of panda exhibits; zoo revenue (SDG 4) (receiving system)"},
        {"system": "receiving", "type": "ecological / biological", "description": "Contribution to captive breeding and genetic diversity programs (SDG 15) (receiving system)"},
        {"system": "spillover", "type": "economic", "description": "Unequal distribution of tourism benefits among nearby communities (SDG 10) (spillover system)"},
        {"system": "spillover", "type": "ecological / biological", "description": "Tourism development pressures on adjacent non-protected areas (spillover system)"},
        {"system": "spillover", "type": "political / institutional", "description": "Reallocation of conservation funding priorities influenced by panda diplomacy (spillover system)"},
    ],
    keywords=["panda", "conservation", "SDG", "sustainable development", "tourism",
              "ecotourism", "biodiversity", "nature reserve", "loan", "wildlife"],
)


# ---------------------------------------------------------------------------
# Example 5: Wolong Nature Reserve Metacoupling (Liu, 2017)
# ---------------------------------------------------------------------------

WOLONG_METACOUPLING = FrameworkExample(
    title="Wolong Nature Reserve: Full Metacoupling Analysis",
    source="Liu (2017) – Integration across a metacoupled world",
    domain="conservation, wildlife, tourism, migration, ecosystem services",
    coupling_type="metacoupling",
    description=(
        "Comprehensive metacoupling analysis of Wolong Nature Reserve in China, "
        "demonstrating how intracoupling (local human-nature interactions such as "
        "fuelwood collection and agriculture), pericoupling (immigration through "
        "marriage from adjacent counties, panda movement across reserve boundaries), "
        "and telecoupling (panda loans to distant zoos, international tourism) "
        "interact in complex ways. This is the foundational worked example of "
        "the metacoupling framework showing all three coupling types."
    ),
    systems={
        "sending": {
            "name": "Wolong Nature Reserve",
            "human_subsystem": (
                "Reserve management, local residents (~5,000 people in "
                "~1,200 households), breeding center staff."
            ),
            "natural_subsystem": (
                "Giant panda habitat, bamboo forests, montane ecosystems, "
                "~150 wild pandas and 200+ captive pandas."
            ),
            "geographic_scope": "Wolong Nature Reserve, Sichuan Province, China.",
        },
        "receiving": {
            "name": "Zoos and tourist source cities",
            "human_subsystem": (
                "Zoo managers, tourists, families in adjacent counties "
                "(Xiaojin, Li, Wenchuan for pericoupling)."
            ),
            "natural_subsystem": (
                "Urban environments, captive breeding facilities at "
                "receiving zoos."
            ),
            "geographic_scope": (
                "Beijing Zoo, National Zoo (Washington DC), other "
                "domestic and international zoos; adjacent counties."
            ),
        },
        "spillover": {
            "name": "Neighboring regions and wider conservation community",
            "human_subsystem": (
                "Communities in areas from which people travel to see "
                "pandas at receiving-system zoos; residents in neighboring "
                "regions affected by tourism infrastructure development."
            ),
            "natural_subsystem": (
                "Panda habitat corridors crossing reserve boundaries into "
                "adjacent areas; ecosystems in neighboring regions affected "
                "by development for tourism infrastructure."
            ),
            "geographic_scope": (
                "Regions neighboring Wolong reserve, areas along panda "
                "habitat corridors, broader regions affected by conservation "
                "funding reallocation."
            ),
        },
    },
    flows=[
        {
            "category": "organisms",
            "direction": "Wolong -> zoos worldwide",
            "description": "Captive pandas loaned to domestic and international zoos (telecoupling); total loans grew from <20 in 1998 to 85 in 2010",
        },
        {
            "category": "people",
            "direction": "Tourist cities -> Wolong",
            "description": "Tourist flows from domestic and international cities to Wolong (telecoupling)",
        },
        {
            "category": "people",
            "direction": "Adjacent counties -> Wolong",
            "description": "Immigration through marriage from neighboring Xiaojin, Li, Wenchuan counties (pericoupling)",
        },
        {
            "category": "organisms",
            "direction": "Across reserve boundaries",
            "description": "Wild panda movement across Wolong reserve boundaries to adjacent areas (pericoupling)",
        },
        {
            "category": "capital",
            "direction": "Zoos/tourists -> Wolong",
            "description": "Panda loan fees (up to $1M per panda per year) and tourism revenue (telecoupling)",
        },
        {
            "category": "matter",
            "direction": "Within Wolong",
            "description": "Fuelwood collection, agriculture, and resource use by local households (intracoupling)",
        },
    ],
    agents=[
        {"level": "governments / policymakers", "name": "Wolong Nature Reserve Administration", "description": "Reserve management in sending system"},
        {"level": "governments / policymakers", "name": "China's State Forestry Administration", "description": "Designs panda loan policies as intermediary"},
        {"level": "non-human agents", "name": "Giant pandas", "description": "Flagship organisms whose movement, reproduction, and conservation status shape tourism and panda-loan couplings"},
        {"level": "individuals / households", "name": "Local residents in Wolong (~1,200 households)", "description": "Households in sending system / intracoupling"},
        {"level": "organizations / NGOs", "name": "International and domestic zoos", "description": "Receive loaned pandas in receiving system"},
        {"level": "individuals / households", "name": "Tourists from China and worldwide", "description": "Visitors in receiving system"},
        {"level": "individuals / households", "name": "Families in adjacent counties", "description": "Involved in cross-boundary marriage (pericoupling)"},
    ],
    causes=[
        {"category": "cultural / social / demographic", "description": "Strong global interest in giant pandas as an iconic and culturally significant species"},
        {"category": "political / institutional", "description": "Panda loan agreements between Chinese government and foreign zoos as diplomatic instruments"},
        {"category": "cultural / social / demographic", "description": "Marriage customs enabling cross-boundary immigration from adjacent counties (pericoupling)"},
        {"category": "economic", "description": "Need for sustainable funding for nature reserve management"},
        {"category": "political / institutional", "description": "China's panda diplomacy strategy for international relations"},
        {"category": "cultural / social / demographic", "description": "Local population dynamics (aging, young people leaving) creating labor needs filled by marriage immigration"},
    ],
    effects=[
        {"system": "sending", "type": "economic", "description": "Tourism revenue and panda loan fees supporting conservation and local livelihoods (sending system)"},
        {"system": "sending", "type": "ecological / biological", "description": "Reduced local disturbance from fuelwood programs; but tourism pressure on habitat (sending system)"},
        {"system": "sending", "type": "cultural / social / demographic", "description": "Population growth from cross-boundary marriage immigration (pericoupling) (sending system)"},
        {"system": "receiving", "type": "cultural / social / demographic", "description": "Educational value and visitor revenue at zoos with loaned pandas (receiving system)"},
        {"system": "receiving", "type": "climatic / atmospheric", "description": "CO2 emissions from transporting pandas between systems (receiving system)"},
        {"system": "spillover", "type": "economic", "description": "Increased visitation to zoos in spillover areas to see pandas (spillover system)"},
        {"system": "spillover", "type": "ecological / biological", "description": "Panda habitat connectivity affected by development in adjacent areas (spillover system)"},
        {"system": "spillover", "type": "ecological / biological", "description": "Reduced genetic diversity risk from panda movement across reserve boundaries (spillover system)"},
    ],
    keywords=["panda", "Wolong", "conservation", "nature reserve", "tourism",
              "panda loan", "marriage", "migration", "wildlife", "ecosystem",
              "intracoupling", "pericoupling", "fuelwood"],
)


# ---------------------------------------------------------------------------
# All Examples
# ---------------------------------------------------------------------------

ALL_EXAMPLES: list[FrameworkExample] = [
    SOYBEAN_TRADE,
    RIFA_INVASION,
    BEIJING_WATER,
    PANDA_SDGS,
    WOLONG_METACOUPLING,
]


def get_relevant_examples(
    research_context: str,
    max_examples: int = 2,
) -> list[FrameworkExample]:
    """Select the most relevant examples based on keyword overlap.

    Parameters
    ----------
    research_context:
        The user's research description to match against.
    max_examples:
        Maximum number of examples to return.

    Returns
    -------
    A list of the most relevant ``FrameworkExample`` instances, ordered by
    relevance (highest first).
    """
    if not research_context.strip():
        # Return the first examples as defaults when no context given.
        return ALL_EXAMPLES[:max_examples]

    context_lower = research_context.lower()
    context_words = set(context_lower.split())

    scored: list[tuple[float, FrameworkExample]] = []
    for example in ALL_EXAMPLES:
        score = 0.0
        # Keyword matching (each keyword found in context scores a point)
        for kw in example.keywords:
            kw_lower = kw.lower()
            if kw_lower in context_lower:
                score += 2.0  # substring match (stronger)
            elif kw_lower in context_words:
                score += 1.0  # exact word match
        # Domain match
        for domain_word in example.domain.split(", "):
            if domain_word.lower() in context_lower:
                score += 1.5
        scored.append((score, example))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [example for _, example in scored[:max_examples]]


def format_example(example: FrameworkExample) -> str:
    """Format a single example as readable text for prompt injection."""
    lines: list[str] = []
    lines.append(f"### {example.title}")
    lines.append(f"Source: {example.source}")
    lines.append(f"Type: {example.coupling_type.title()}")
    lines.append(f"\n{example.description}\n")

    lines.append("**Systems:**")
    for role, value in example.systems.items():
        if isinstance(value, dict):
            name = value.get("name", "")
            lines.append(f"  **{role.title()} System**: {name}")
            if value.get("human_subsystem"):
                lines.append(f"    - **Human subsystem**: {value['human_subsystem']}")
            if value.get("natural_subsystem"):
                lines.append(f"    - **Natural subsystem**: {value['natural_subsystem']}")
            if value.get("geographic_scope"):
                lines.append(f"    - **Geographic scope**: {value['geographic_scope']}")
        else:
            lines.append(f"  - {role.title()}: {value}")

    lines.append("\n**Flows:**")
    for flow in example.flows:
        lines.append(
            f"  - [{flow['category'].title()}] {flow['direction']}: "
            f"{flow['description']}"
        )

    lines.append("\n**Agents:**")
    for agent in example.agents:
        level = agent.get("level", "").title()
        name = agent.get("name", "")
        desc = agent.get("description", "")
        lines.append(f"  - [{level}] {name}: {desc}")

    lines.append("\n**Causes:**")
    for cause in example.causes:
        cat = cause.get("category", cause.get("type", "")).title()
        lines.append(f"  - [{cat}] {cause['description']}")

    lines.append("\n**Effects:**")
    # Group effects by category (type), not by system
    effects_by_type: dict[str, list[str]] = {}
    for effect in example.effects:
        cat = effect.get("type", "general").title()
        effects_by_type.setdefault(cat, []).append(effect["description"])
    for cat, items in effects_by_type.items():
        lines.append(f"  **{cat}**")
        for item in items:
            lines.append(f"  - {item}")

    return "\n".join(lines)

"""
Curated case-study examples from the metacoupling literature.

Each example summarises one published study from the package's literature
corpus and is laid out the way the analysis itself must be written: coupling
type first (intracoupling, pericoupling, telecoupling), and within each coupling
type the systems in the role they play for that type's flows, then the flows,
agents, causes and effects, followed by the cross-coupling interactions. Facts
come from the cited paper; where an example departs from it, for instance by
leaving out a flow between countries that share no border, ``adaptation_note``
says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

COUPLING_TYPES: tuple[str, ...] = ("intracoupling", "pericoupling", "telecoupling")

SYSTEM_ROLES: tuple[str, ...] = ("focal", "sending", "receiving", "spillover")

FLOW_CATEGORIES: tuple[str, ...] = (
    "capital", "energy", "information", "matter", "organisms", "people",
)

# Agent levels and cause/effect categories, with their display labels exactly
# as the output template writes them.
AGENT_LEVELS: dict[str, str] = {
    "individuals / households": "Individuals / households",
    "firms / traders / corporations": "Firms / traders / corporations",
    "governments / policymakers": "Governments / policymakers",
    "organizations / NGOs": "Organizations / NGOs",
    "non-human agents": "Non-human agents",
}
CATEGORIES: dict[str, str] = {
    "economic": "Economic",
    "political / institutional": "Political / Institutional",
    "ecological / biological": "Ecological / Biological",
    "technological / infrastructural": "Technological / Infrastructural",
    "cultural / social / demographic": "Cultural / Social / Demographic",
    "hydrological": "Hydrological",
    "climatic / atmospheric": "Climatic / Atmospheric",
    "geological / geomorphological": "Geological / Geomorphological",
}

# The template's approved vocabulary for cross-coupling interactions; each
# cross-coupling label starts with one of these terms.
CROSS_COUPLING_TERMS: tuple[str, ...] = (
    "Amplification", "Offset", "Spatial tradeoffs", "Temporal tradeoffs",
    "Synergies", "Cascading effects", "Feedback loops", "Displacement",
    "Coupling transformations", "Spillover effects",
)

_SECTION_HEADINGS: dict[str, str] = {
    "intracoupling": "Intracoupling (within the focal system)",
    "pericoupling": "Pericoupling (adjacent systems)",
    "telecoupling": "Telecoupling (distant systems)",
}


@dataclass(frozen=True)
class FrameworkExample:
    """A published application of the metacoupling framework, coupling-first.

    Every system, flow, agent, cause and effect names the ``coupling`` type it
    belongs to. Systems also name their ``role`` for the flows of that coupling
    type (``focal``, ``sending``, ``receiving`` or ``spillover``), so a place
    that sends one flow and receives another appears once in each role.
    """

    title: str
    source: str  # full reference of the paper the example summarises
    corpus_file: str  # the paper's file name, without ".md", in the bundled corpus
    domain: str
    description: str
    classification: dict[str, str]  # coupling type -> why it is present
    systems: list[dict[str, str]]  # coupling, role, name, human_subsystem, natural_subsystem, geographic_scope
    flows: list[dict[str, str]]  # coupling, category, direction, description
    agents: list[dict[str, str]]  # coupling, level, name, description
    causes: list[dict[str, str]]  # coupling, category, description
    effects: list[dict[str, str]]  # coupling, category, system, description
    cross_coupling: list[tuple[str, str]] = field(default_factory=list)  # (label, text)
    adaptation_note: str = ""
    keywords: list[str] = field(default_factory=list)

    @property
    def coupling_types(self) -> list[str]:
        """The coupling types the example demonstrates, in canonical order."""
        return [c for c in COUPLING_TYPES if c in self.classification]


# ---------------------------------------------------------------------------
# Example 1: Wolong Nature Reserve (Liu, 2017)
# ---------------------------------------------------------------------------

WOLONG_METACOUPLING = FrameworkExample(
    title="Wolong Nature Reserve, China: Panda Loans, Tourism and Marriage Migration",
    source="Liu, J. (2017). Integration across a metacoupled world. Ecology and Society, 22(4), 29.",
    corpus_file="Liu - 2017 - Integration across a metacoupled world",
    domain="conservation, wildlife, tourism, migration",
    description=(
        "Liu (2017) illustrates the metacoupling framework with Wolong Nature "
        "Reserve, a 2,000 km² reserve in Sichuan, China, home to giant pandas and "
        "about 5,000 residents in more than 1,100 households. Two processes stand "
        "for each coupling type: fuelwood and timber collection within the "
        "reserve; wild-panda movement and marriage migration with the bordering "
        "counties; and panda loans and tourism with distant places. Wolong sends "
        "some flows and receives others, so its role is given per flow."
    ),
    classification={
        "intracoupling": (
            "Residents collect fuelwood and timber from the reserve's forests and "
            "panda habitat, and forest-protection programs pay households to "
            "restore and guard the forest."
        ),
        "pericoupling": (
            "Wild pandas cross the reserve boundary into abutting areas, and women "
            "from the bordering counties marry into Wolong households; the paper "
            "treats systems that share a boundary as adjacent."
        ),
        "telecoupling": (
            "Captive pandas are loaned to zoos around the world, and tourists "
            "arrive from across China and other countries, places that share no "
            "boundary with Wolong."
        ),
    },
    systems=[
        {"coupling": "intracoupling", "role": "focal",
         "name": "Wolong Nature Reserve, Sichuan Province, China",
         "human_subsystem": "About 5,000 residents, mostly farmers, in more than 1,100 households; reserve managers and policy makers",
         "natural_subsystem": "Forests, giant pandas and their habitat, and over 6,000 other animal and plant species",
         "geographic_scope": "The 2,000 km² reserve in southwestern China"},
        {"coupling": "pericoupling", "role": "sending",
         "name": "Xiaojin, Li and Wenchuan counties (for marriage migration)",
         "human_subsystem": "Families whose daughters marry into Wolong; matchmakers and hometown networks",
         "natural_subsystem": "Rural mountain landscapes outside the reserve (not analysed in the paper)",
         "geographic_scope": "Counties that share a boundary with the reserve"},
        {"coupling": "pericoupling", "role": "receiving",
         "name": "Wolong Nature Reserve (for marriage migration)",
         "human_subsystem": "Households gaining brides, and with them more people and households",
         "natural_subsystem": "Forests facing higher demand for natural resources",
         "geographic_scope": "The reserve"},
        {"coupling": "pericoupling", "role": "sending",
         "name": "Wolong Nature Reserve (for wild-panda movement)",
         "human_subsystem": "Wolong Administration Bureau, which protects panda habitat and movement corridors",
         "natural_subsystem": "Wild giant pandas and their habitat inside the reserve",
         "geographic_scope": "The reserve"},
        {"coupling": "pericoupling", "role": "receiving",
         "name": "Areas abutting the reserve (for wild-panda movement)",
         "human_subsystem": "Land users and agencies next to the boundary (not detailed in the paper)",
         "natural_subsystem": "Panda habitat beyond an unfenced boundary",
         "geographic_scope": "Land next to the reserve boundary"},
        {"coupling": "pericoupling", "role": "spillover",
         "name": "Other neighbouring counties and areas affected by immigrants or moving pandas",
         "human_subsystem": "Communities whose residents leave or arrive through marriage",
         "natural_subsystem": "Habitat affected by panda movement beyond the abutting areas",
         "geographic_scope": "The wider region around Wolong"},
        {"coupling": "telecoupling", "role": "sending",
         "name": "Wolong Nature Reserve (for panda loans)",
         "human_subsystem": "Breeding-base staff and the reserve administration",
         "natural_subsystem": "More than 200 captive pandas at the breeding base",
         "geographic_scope": "The reserve"},
        {"coupling": "telecoupling", "role": "receiving",
         "name": "Zoos worldwide, such as the Beijing Zoo and the National Zoo in Washington, D.C.",
         "human_subsystem": "Zoo managers and zoo visitors",
         "natural_subsystem": "Captive-breeding facilities in urban zoos",
         "geographic_scope": "Domestic and international zoos"},
        {"coupling": "telecoupling", "role": "sending",
         "name": "Tourists' home regions (for tourism)",
         "human_subsystem": "Tourists from 30 provinces and cities of mainland China, Taiwan, Hong Kong and 26 countries",
         "natural_subsystem": "Not analysed in the paper",
         "geographic_scope": "Where the 1,063 tourists surveyed in 2006-2007 came from"},
        {"coupling": "telecoupling", "role": "receiving",
         "name": "Wolong Nature Reserve (for tourism)",
         "human_subsystem": "Households earning tourism income; the Sichuan Tourism Bureau",
         "natural_subsystem": "Pandas, natural forests and wildlife, clean air and water, and the trails visitors use",
         "geographic_scope": "The reserve"},
        {"coupling": "telecoupling", "role": "spillover",
         "name": "Stopover cities such as Chengdu, areas that make outdoor gear, and the global atmosphere",
         "human_subsystem": "Businesses supporting tourism; the home areas of people who visit loaned pandas",
         "natural_subsystem": "The global atmosphere receiving CO2 from transporting pandas and from tourists' travel",
         "geographic_scope": "Places along and beyond the travel routes"},
    ],
    flows=[
        {"coupling": "intracoupling", "category": "people",
         "direction": "Households → forests and panda habitat",
         "description": "Residents going into the forest to collect wood and carry out other activities"},
        {"coupling": "intracoupling", "category": "matter",
         "direction": "Forests and panda habitat → households",
         "description": "Fuelwood, timber and non-timber products for cooking, heating and building"},
        {"coupling": "intracoupling", "category": "information",
         "direction": "Forests and pandas → policy makers",
         "description": "Information on the condition of forests, pandas and their habitat"},
        {"coupling": "pericoupling", "category": "people",
         "direction": "Xiaojin, Li and Wenchuan counties → Wolong",
         "description": "Women marrying into Wolong; in 2010, 49 of 370 married women in 287 interviewed households were born in neighbouring counties"},
        {"coupling": "pericoupling", "category": "organisms",
         "direction": "Wolong → areas abutting the reserve",
         "description": "Wild pandas moving across the unfenced reserve boundary"},
        {"coupling": "telecoupling", "category": "organisms",
         "direction": "Wolong → zoos worldwide",
         "description": "Captive pandas on loans of several years; total loans rose from fewer than 20 in 1998 to 85 in 2010"},
        {"coupling": "telecoupling", "category": "capital",
         "direction": "Borrowing zoos → the panda-loan program",
         "description": "Loan fees of as much as US$1 million per panda per year"},
        {"coupling": "telecoupling", "category": "people",
         "direction": "Tourists' home regions → Wolong",
         "description": "Tourists; visits peaked at 220,000 in 2006, fell during road construction in 2007 and almost stopped after the 2008 Wenchuan earthquake"},
        {"coupling": "telecoupling", "category": "capital",
         "direction": "Tourists and donors → Wolong",
         "description": "Tourism spending, and donations to breeding and research through the Wolong Panda Club"},
    ],
    agents=[
        {"coupling": "intracoupling", "level": "individuals / households",
         "name": "Wolong households", "description": "Collect fuelwood and timber and farm inside the reserve"},
        {"coupling": "intracoupling", "level": "governments / policymakers",
         "name": "Reserve managers and policy makers", "description": "Run the forest-protection programs and use information on forest and panda conditions"},
        {"coupling": "pericoupling", "level": "non-human agents",
         "name": "Wild giant pandas", "description": "Move across the boundary to find food and mates and avoid risks"},
        {"coupling": "pericoupling", "level": "governments / policymakers",
         "name": "Wolong Administration Bureau", "description": "Protects panda habitat and movement corridors"},
        {"coupling": "pericoupling", "level": "individuals / households",
         "name": "Immigrant brides and matchmakers", "description": "Introduce women from their hometowns to men in Wolong"},
        {"coupling": "telecoupling", "level": "governments / policymakers",
         "name": "China's State Forestry Administration", "description": "Designs panda-loan policies and agreements"},
        {"coupling": "telecoupling", "level": "organizations / NGOs",
         "name": "Zoos worldwide", "description": "Borrow pandas and pay loan fees"},
        {"coupling": "telecoupling", "level": "governments / policymakers",
         "name": "Sichuan Tourism Bureau", "description": "Develops and implements tourism policy"},
        {"coupling": "telecoupling", "level": "individuals / households",
         "name": "Tourists from China and abroad", "description": "Visit the reserve and spend money there"},
    ],
    causes=[
        {"coupling": "intracoupling", "category": "economic",
         "description": "Households need fuelwood for cooking and heating"},
        {"coupling": "intracoupling", "category": "political / institutional",
         "description": "The Grain-to-Green Program (2000) pays farmers to return steep cropland to forest, and the Natural Forest Conservation Program (2001) bans logging and pays households to monitor forests"},
        {"coupling": "pericoupling", "category": "ecological / biological",
         "description": "Pandas move to find food and mates and avoid risks, and no fence stops them at the boundary"},
        {"coupling": "pericoupling", "category": "economic",
         "description": "Living standards are higher in Wolong than in the bordering counties"},
        {"coupling": "pericoupling", "category": "cultural / social / demographic",
         "description": "Brides' hometown networks bring more women to Wolong"},
        {"coupling": "telecoupling", "category": "cultural / social / demographic",
         "description": "Strong public interest in pandas around the world"},
        {"coupling": "telecoupling", "category": "ecological / biological",
         "description": "Pandas, natural forests and wildlife, and clean air and water draw tourists"},
        {"coupling": "telecoupling", "category": "technological / infrastructural",
         "description": "Road construction cut tourist numbers in 2007"},
        {"coupling": "telecoupling", "category": "geological / geomorphological",
         "description": "The 2008 Wenchuan earthquake destroyed the road into the reserve and stopped tourism"},
    ],
    effects=[
        {"coupling": "intracoupling", "category": "ecological / biological", "system": "Wolong, focal system",
         "description": "Wood collection degraded and fragmented panda habitat; the forest programs work to restore it"},
        {"coupling": "intracoupling", "category": "climatic / atmospheric", "system": "the rest of the world, spillover system",
         "description": "CO2 from burning fuelwood leaves the reserve; a small hydropower station reduces fuelwood use"},
        {"coupling": "pericoupling", "category": "ecological / biological", "system": "wild pandas in the abutting areas, receiving system",
         "description": "Moving pandas obtain food and mates and avoid risks"},
        {"coupling": "pericoupling", "category": "cultural / social / demographic", "system": "Wolong, receiving system",
         "description": "Marriage migration adds people and households and raises demand for natural resources in the reserve"},
        {"coupling": "telecoupling", "category": "economic", "system": "Wolong, receiving system for tourists",
         "description": "In 2006, 76.5% of Wolong households earned income from tourism, directly or indirectly"},
        {"coupling": "telecoupling", "category": "economic", "system": "zoos, the receiving system, and spillover areas",
         "description": "Zoos bear facility and operating costs, visitors pay entrance fees and travel costs, and areas making outdoor clothes and hiking shoes gain business"},
        {"coupling": "telecoupling", "category": "ecological / biological", "system": "Wolong, receiving system for tourists",
         "description": "Tourists affect vegetation along trails"},
        {"coupling": "telecoupling", "category": "climatic / atmospheric", "system": "the global atmosphere, spillover system",
         "description": "CO2 from transporting pandas between Wolong and zoos and from tourists' travel"},
    ],
    cross_coupling=[
        ("Cascading effects across all three coupling types",
         "Reducing residents' pressure on panda habitat (intracoupling) can increase the number of wild "
         "pandas, which may increase panda movement across the boundary (pericoupling) and, as wild "
         "pandas join the breeding base, panda loans (telecoupling)."),
        ("Cascading effects with a time lag",
         "Tourism income (telecoupling) let some households pay for more education; children who reach "
         "college usually settle in cities, which reduces pressure on panda habitat (intracoupling), but "
         "only after about 12 years of schooling."),
        ("Coupling transformations",
         "Panda loans and tourism emerged only after the reserve was established, and tourism was "
         "decoupled when the 2008 earthquake destroyed the road into the reserve."),
    ],
    keywords=["panda", "Wolong", "wildlife", "conservation", "nature reserve",
              "protected area", "tourism", "ecotourism", "tourist", "migration",
              "marriage", "fuelwood", "zoo", "biodiversity", "China"],
)


# ---------------------------------------------------------------------------
# Example 2: Soybean, corn and wheat trade in South America (Herzberger et al., 2019)
# ---------------------------------------------------------------------------

SOUTH_AMERICA_FOOD_TRADE = FrameworkExample(
    title="Soybean, Corn and Wheat Trade: Brazil, China and the Mercosur Neighbours",
    source=(
        "Herzberger, A., Chung, M. G., Kapsar, K., Frank, K. A., & Liu, J. (2019). "
        "Telecoupled food trade affects pericoupled trade and intracoupled production. "
        "Sustainability, 11(10), 2908."
    ),
    corpus_file="Herzberger et al. - 2019 - Telecoupled Food Trade Affects Pericoupled Trade and Intracoupled Production",
    domain="agriculture, food trade, commodities",
    description=(
        "Herzberger et al. (2019) ask how a distant flow, Brazil's soybean exports "
        "to China, changed trade between adjacent countries and production within "
        "them. Using FAOSTAT production and UN Comtrade trade data for soybeans, "
        "corn and wheat in 1991-2016, they compare the years before and after China "
        "cut its soybean import tariff from 130% to 3% in 1995. Argentina, Paraguay "
        "and Uruguay were studied because each shares a border and Mercosur "
        "membership with Brazil."
    ),
    classification={
        "intracoupling": (
            "Soybean, corn and wheat production within each country, such as "
            "Brazil's expansion as farmers gained access to imported fertilizers, "
            "pesticides and seeds."
        ),
        "pericoupling": (
            "Crop trade and investment among Brazil, Argentina, Paraguay and "
            "Uruguay, countries that share land borders."
        ),
        "telecoupling": (
            "Soybean exports from Brazil, Argentina and Uruguay to China, and from "
            "Paraguay to Russia and Turkey, none of which share a border with the "
            "exporter."
        ),
    },
    systems=[
        {"coupling": "intracoupling", "role": "focal",
         "name": "Brazil",
         "human_subsystem": "Soybean, corn and wheat farmers; a government that liberalised trade and farmers' access to inputs",
         "natural_subsystem": "Cropland for soybean, corn and wheat, including land in the Amazon and Cerrado biomes",
         "geographic_scope": "Brazil"},
        {"coupling": "pericoupling", "role": "sending",
         "name": "Argentina, Paraguay and Uruguay (for crop exports to Brazil)",
         "human_subsystem": "Farmers and exporters; Argentina's government, which taxes soybean exports outside Mercosur",
         "natural_subsystem": "Cropland for soybean, corn and wheat",
         "geographic_scope": "The three Mercosur members that border Brazil"},
        {"coupling": "pericoupling", "role": "receiving",
         "name": "Brazil (for crops from its neighbours)",
         "human_subsystem": "Importers and processors of neighbours' soybean, corn and wheat",
         "natural_subsystem": "Brazilian cropland, whose output these imports supplement",
         "geographic_scope": "Brazil"},
        {"coupling": "pericoupling", "role": "receiving",
         "name": "Uruguay (for Argentine investment)",
         "human_subsystem": "A soybean sector expanding with investment from Argentine producers",
         "natural_subsystem": "Farmland about 50% cheaper than soybean land in Argentina",
         "geographic_scope": "Uruguay"},
        {"coupling": "telecoupling", "role": "sending",
         "name": "Brazil, Argentina and Uruguay (for soybean exports to China)",
         "human_subsystem": "Soybean exporters and the governments that set export rules",
         "natural_subsystem": "Soybean cropland",
         "geographic_scope": "The three exporters"},
        {"coupling": "telecoupling", "role": "receiving",
         "name": "China",
         "human_subsystem": "A growing middle class eating more meat; importers after the 1995 tariff cut",
         "natural_subsystem": "Not analysed in the paper",
         "geographic_scope": "China"},
        {"coupling": "telecoupling", "role": "receiving",
         "name": "Russia and Turkey (for Paraguayan soybeans)",
         "human_subsystem": "Processors that export soybean oil to China",
         "natural_subsystem": "Not analysed in the paper",
         "geographic_scope": "Russia and Turkey"},
        {"coupling": "telecoupling", "role": "spillover",
         "name": "United States",
         "human_subsystem": "Soybean exporters whose share of China's imports fell from 68% in 1995 to 40% in 2016",
         "natural_subsystem": "US soybean cropland (not analysed in the paper)",
         "geographic_scope": "The United States"},
    ],
    flows=[
        {"coupling": "intracoupling", "category": "matter",
         "direction": "Brazilian farms → domestic markets and export ports",
         "description": "Soybeans, corn and wheat produced within Brazil"},
        {"coupling": "pericoupling", "category": "matter",
         "direction": "Paraguay → Brazil",
         "description": "Soybeans, corn and wheat; compared with 1991-1995, these exports rose by 114%, 1,712% and 6,474%"},
        {"coupling": "pericoupling", "category": "matter",
         "direction": "Argentina → Brazil",
         "description": "Soybeans, corn and wheat; soybean exports stopped in 1995 and resumed consistently only around 2005"},
        {"coupling": "pericoupling", "category": "matter",
         "direction": "Uruguay → Brazil",
         "description": "Soybeans and corn; soybean exports fell to zero in 1997 and did not resume consistently"},
        {"coupling": "pericoupling", "category": "capital",
         "direction": "Argentina → Uruguay",
         "description": "Investment by Argentine producers, the main driver of Uruguay's soybean expansion"},
        {"coupling": "pericoupling", "category": "matter",
         "direction": "Paraguay → Argentina",
         "description": "Soybeans shipped by barge to Argentine river ports"},
        {"coupling": "telecoupling", "category": "matter",
         "direction": "Brazil → China",
         "description": "Soybeans; Brazil's share of China's soybean imports rose from 0% in 1991-1995 to 46% in 2016"},
        {"coupling": "telecoupling", "category": "matter",
         "direction": "Argentina and Uruguay → China",
         "description": "Soybeans; Uruguay's exports to China rose 1,250% from 2005 to 2016, while Argentina's grew only 23% after its 2006 export tax"},
        {"coupling": "telecoupling", "category": "matter",
         "direction": "Paraguay → Russia and Turkey",
         "description": "Soybeans, which those countries process and export to China as soybean oil"},
        {"coupling": "telecoupling", "category": "information",
         "direction": "China → the world soybean market",
         "description": "The demand signal of China's 1995 soybean tariff cut from 130% to 3%"},
    ],
    agents=[
        {"coupling": "intracoupling", "level": "individuals / households",
         "name": "Brazilian farmers", "description": "Expanded soybean production with imported fertilizers, pesticides and seeds"},
        {"coupling": "intracoupling", "level": "governments / policymakers",
         "name": "Brazilian government", "description": "Liberalised trade and farmers' access to inputs"},
        {"coupling": "pericoupling", "level": "governments / policymakers",
         "name": "Argentine government", "description": "Imposed a 23.5% soybean export tax in 2006 that exempts Mercosur partners"},
        {"coupling": "pericoupling", "level": "firms / traders / corporations",
         "name": "Argentine producers", "description": "Invested in Uruguayan soybean farming"},
        {"coupling": "pericoupling", "level": "governments / policymakers",
         "name": "Paraguayan government", "description": "Keeps diplomatic ties with Taiwan, which closes the Chinese market to Paraguay"},
        {"coupling": "telecoupling", "level": "governments / policymakers",
         "name": "Chinese government", "description": "Cut the soybean import tariff in 1995"},
        {"coupling": "telecoupling", "level": "individuals / households",
         "name": "China's middle class", "description": "Eats more meat, the main driver of China's soybean demand"},
        {"coupling": "telecoupling", "level": "firms / traders / corporations",
         "name": "Multinational companies", "description": "Invested in Uruguay's soybean expansion"},
    ],
    causes=[
        {"coupling": "intracoupling", "category": "political / institutional",
         "description": "Brazil liberalised trade and farmers' access to fertilizers, pesticides and seeds"},
        {"coupling": "pericoupling", "category": "political / institutional",
         "description": "Shared borders and joint Mercosur membership make neighbour trade feasible, and Argentina's export tax does not apply to Mercosur buyers"},
        {"coupling": "pericoupling", "category": "economic",
         "description": "Uruguayan land costs about half as much as Argentine soybean land, and an early-2000s financial crisis encouraged investment"},
        {"coupling": "pericoupling", "category": "technological / infrastructural",
         "description": "Paraguay ships its soybeans by barge to river ports in neighbouring countries"},
        {"coupling": "telecoupling", "category": "political / institutional",
         "description": "China's 1995 tariff cut; European and US biofuel mandates, which drew Argentine land into corn"},
        {"coupling": "telecoupling", "category": "cultural / social / demographic",
         "description": "Rising meat consumption by China's middle class"},
    ],
    effects=[
        {"coupling": "intracoupling", "category": "economic", "system": "Brazil, focal system",
         "description": "Brazil emerged as the most competitive soybean exporter and passed the United States in production in 2013"},
        {"coupling": "intracoupling", "category": "ecological / biological", "system": "Brazil, focal system",
         "description": "Citing Dou et al., the paper notes that conservation agreements such as the Soy Moratorium shifted deforestation from the Amazon to the Cerrado"},
        {"coupling": "pericoupling", "category": "economic", "system": "Argentina, sending system",
         "description": "After the 2006 export tax, soybean area grew 45% (2006-2016, against 215% in 1995-2005), corn area grew faster, and soybean exports shifted toward Mercosur neighbours"},
        {"coupling": "pericoupling", "category": "economic", "system": "Paraguay, sending system",
         "description": "Shut out of China, Paraguay increased exports of all three crops to Brazil"},
        {"coupling": "telecoupling", "category": "economic", "system": "United States, spillover system",
         "description": "Brazil passed the United States in China's soybean market in 2006"},
    ],
    cross_coupling=[
        ("Amplification and offset of pericoupled trade",
         "Brazil's soybean exports to China were significantly related to its soybean imports from its "
         "neighbours. Pericoupled trade was enhanced for some partners and crops, such as Paraguay's "
         "exports of all three crops to Brazil, and offset for others, such as Uruguay's soybean and corn "
         "exports to Brazil, which fell to zero or became intermittent after China's 1995 tariff cut."),
        ("Displacement of trade toward neighbours",
         "Argentina's 2006 export tax slowed its soybean exports to China, which rose 23% afterwards, "
         "while exports to Mercosur neighbours, exempt from the tax, grew faster."),
    ],
    adaptation_note=(
        "The paper names no spillover system; the United States, whose share of China's soybean imports "
        "Brazil overtook, is shown as one, as in Liu et al. (2013). The paper also counts Paraguay-Uruguay "
        "trade as pericoupled; the two countries share no border, so that flow is left out."
    ),
    keywords=["trade", "food", "soybean", "soy", "corn", "maize", "wheat", "crop",
              "agriculture", "commodity", "export", "import", "tariff", "Brazil",
              "Argentina", "Paraguay", "Uruguay", "Mercosur", "China"],
)


# ---------------------------------------------------------------------------
# Example 3: Global marine fishing (Carlson et al., 2020)
# ---------------------------------------------------------------------------

GLOBAL_MARINE_FISHING = FrameworkExample(
    title="Global Marine Fishing in Own, Adjacent and Distant Waters, 1950-2014",
    source=(
        "Carlson, A. K., Taylor, W. W., Rubenstein, D. I., Levin, S. A., & Liu, J. (2020). "
        "Global marine fishing across space and time. Sustainability, 12(11), 4714."
    ),
    corpus_file="Carlson et al. - 2020 - Global Marine Fishing Across Space and Time",
    domain="fisheries, marine, oceans, food",
    description=(
        "Carlson et al. (2020) classify every marine catch in the world's 280 "
        "exclusive economic zones (EEZs) and the high seas from 1950 to 2014 by "
        "coupling type, using Sea Around Us data. Fishing in a nation's own EEZ is "
        "intracoupling; industrial fishing in the EEZ of a nation sharing a land or "
        "maritime border is pericoupling; industrial fishing in the EEZs of "
        "non-bordering nations or on the high seas is telecoupling. Total catches "
        "rose from 26.3 million tonnes in 1950 to 104.6 million tonnes in 2014."
    ),
    classification={
        "intracoupling": (
            "Industrial, artisanal, subsistence and recreational catches by "
            "nations in their own EEZs: 73% of all catches in 1950-2014."
        ),
        "pericoupling": (
            "Industrial catches by nations in the EEZs of nations that share a "
            "land or maritime border with them: 748.9 million tonnes, 13%."
        ),
        "telecoupling": (
            "Industrial catches in the EEZs of non-bordering nations and on the "
            "high seas: 683.7 and 108.0 million tonnes, 12% and 2%."
        ),
    },
    systems=[
        {"coupling": "intracoupling", "role": "focal",
         "name": "A fishing nation and its own EEZ",
         "human_subsystem": "Industrial, artisanal, subsistence and recreational fishers of that nation",
         "natural_subsystem": "Fish stocks of the EEZ, supported by its primary production",
         "geographic_scope": "Each of the 280 EEZs; the largest own-EEZ catches were Peru's, Japan's (main islands) and China's"},
        {"coupling": "pericoupling", "role": "sending",
         "name": "EEZs of bordering nations",
         "human_subsystem": "Host nations and their fishers",
         "natural_subsystem": "Fish stocks of the adjacent EEZ",
         "geographic_scope": "EEZs that share a land or maritime border with the fishing nation, such as Cambodia's"},
        {"coupling": "pericoupling", "role": "receiving",
         "name": "Fishing nations whose industrial fleets fish next door",
         "human_subsystem": "Industrial fleets and the markets that land their catch",
         "natural_subsystem": "Home-EEZ stocks, which adjacent catches supplement",
         "geographic_scope": "Nations with neighbours' EEZs in reach, as in Northern and Western Europe"},
        {"coupling": "telecoupling", "role": "sending",
         "name": "EEZs of non-bordering nations and the high seas",
         "human_subsystem": "Host nations, many with limited fishing infrastructure, governance and enforcement",
         "natural_subsystem": "Highly productive fish stocks, as off Western Africa and in Oceania",
         "geographic_scope": "The largest distant catches came from the EEZs of Mauritania, eastern Canada and southern Morocco"},
        {"coupling": "telecoupling", "role": "receiving",
         "name": "Distant-water fishing nations",
         "human_subsystem": "Industrial fleets that fish far from home",
         "natural_subsystem": "Home-EEZ stocks, which distant catches supplement",
         "geographic_scope": "Nations fishing beyond their own and their neighbours' waters"},
    ],
    flows=[
        {"coupling": "intracoupling", "category": "matter",
         "direction": "Own EEZ → the same nation",
         "description": "Catches by all sectors; their value rose from US$25.9 billion in 1950 to US$127.1 billion in 2014; largest totals: Peru 454.4, Japan (main islands) 360.8 and China 315.3 million tonnes"},
        {"coupling": "pericoupling", "category": "matter",
         "direction": "A bordering nation's EEZ → the fishing nation",
         "description": "Industrial catch: 748.9 million tonnes in 1950-2014, 13% of all catches"},
        {"coupling": "telecoupling", "category": "matter",
         "direction": "A non-bordering nation's EEZ → the distant-water fishing nation",
         "description": "Industrial catch: 683.7 million tonnes, 12% of all catches; largest in the EEZs of Mauritania (79.2), eastern Canada (77.1) and southern Morocco (71.7 million tonnes)"},
        {"coupling": "telecoupling", "category": "matter",
         "direction": "High seas → fishing nations",
         "description": "Industrial catch: 108.0 million tonnes, 2% of all catches"},
    ],
    agents=[
        {"coupling": "intracoupling", "level": "individuals / households",
         "name": "Artisanal, subsistence and recreational fishers", "description": "Fish only in their own nation's EEZ"},
        {"coupling": "pericoupling", "level": "firms / traders / corporations",
         "name": "Industrial fishing fleets", "description": "The only sector that fishes in adjacent and distant EEZs"},
        {"coupling": "telecoupling", "level": "governments / policymakers",
         "name": "Fisheries agencies of host nations", "description": "Govern foreign access; weak enforcement allowed large foreign catches off Western Africa"},
        {"coupling": "telecoupling", "level": "firms / traders / corporations",
         "name": "Distant-water industrial fleets", "description": "Catch fish, legally and illegally, where infrastructure and enforcement are limited"},
    ],
    causes=[
        {"coupling": "intracoupling", "category": "economic",
         "description": "Own-EEZ catches rose with national GDP (r = 0.65)"},
        {"coupling": "pericoupling", "category": "economic",
         "description": "In Northern and Western Europe, wealthy, high-capacity fishing nations lie close together, and adjacent-EEZ catches rose with GDP (r = 0.81 and 0.82)"},
        {"coupling": "telecoupling", "category": "political / institutional",
         "description": "Limited fisheries governance and enforcement in host EEZs, notably in Western Africa"},
        {"coupling": "telecoupling", "category": "technological / infrastructural",
         "description": "Limited fishing infrastructure in Oceania and Western Africa"},
        {"coupling": "telecoupling", "category": "ecological / biological",
         "description": "High fish productivity in the same regions"},
    ],
    effects=[
        {"coupling": "intracoupling", "category": "ecological / biological", "system": "the fished EEZ, focal system",
         "description": "Counting all fishing types, catch relative to primary production, the paper's measure of fishing's effect on fish populations, was highest in Cambodia's EEZ (48.1 tonnes of fish per kilotonne of carbon per year)"},
        {"coupling": "pericoupling", "category": "ecological / biological", "system": "adjacent EEZs, sending systems",
         "description": "Adjacent catches relative to primary production were highest in Cambodia's EEZ (41.1 tonnes of fish per kilotonne of carbon per year), marking a hotspot for management"},
        {"coupling": "telecoupling", "category": "ecological / biological", "system": "distant EEZs, sending systems",
         "description": "Distant catches relative to primary production were highest in Western African EEZs, led by Guinea's (9.0 tonnes of fish per kilotonne of carbon per year)"},
        {"coupling": "telecoupling", "category": "economic", "system": "small-scale fishers in host EEZs, sending systems",
         "description": "Distant-water fishing was predicted to cut artisanal and subsistence catches and alter species composition"},
        {"coupling": "telecoupling", "category": "cultural / social / demographic", "system": "small-scale fishers in host EEZs, sending systems",
         "description": "Distant-water fishing was predicted to force small-scale fishers to move to new fishing grounds"},
    ],
    cross_coupling=[
        ("Spatial tradeoffs between distant and own-EEZ fishing",
         "Across 1950-2014, the numbers of EEZs dominated by distant fleets and by own-nation fishing "
         "moved in opposite directions (r = -0.94), consistent with the predicted conflict between "
         "distant-water fleets and local small-scale fishers."),
        ("Offset by stronger governance",
         "In Western African EEZs, adjacent and distant catches fell as fisheries policy effectiveness "
         "rose (r = -0.52), while own-EEZ catches grew (r = 0.52)."),
    ],
    adaptation_note=(
        "The paper assigns no sending or receiving role to named systems; roles here follow its flow "
        "direction, from the fished EEZ to the fishing nation. It names no spillover system, and its "
        "predicted effects on small-scale fishers are tested only through correlations."
    ),
    keywords=["fish", "fishing", "fisheries", "fishery", "marine", "ocean", "sea",
              "EEZ", "catch", "seafood", "maritime", "high seas", "coastal"],
)


# ---------------------------------------------------------------------------
# Example 4: Groundwater bottling and trout streams in Michigan (Carlson et al., 2020)
# ---------------------------------------------------------------------------

MICHIGAN_GROUNDWATER_STREAMS = FrameworkExample(
    title="Groundwater Bottling and Trout Streams in Michigan, USA",
    source=(
        "Carlson, A. K., Taylor, W. W., & Hughes, S. M. (2020). The metacoupling framework "
        "informs stream salmonid management and governance. Frontiers in Environmental "
        "Science, 8, 27."
    ),
    corpus_file="Carlson et al. - 2020 - The Metacoupling Framework Informs Stream Salmonid Management and Governance",
    domain="water, groundwater, freshwater fisheries, recreation",
    description=(
        "Carlson et al. (2020) apply the metacoupling framework to the brook charr "
        "and brown trout fisheries of Twin and Chippewa creeks in Osceola County, "
        "Michigan, where Nestlé pumps groundwater for bottling. The two creeks' "
        "watersheds share a common boundary, so interactions between them are "
        "pericoupling; interactions with non-adjacent streams, distant anglers and "
        "the company's headquarters in Connecticut are telecoupling. The study "
        "draws on groundwater models, stream gauging, interviews and a 2015 survey "
        "of 4,161 anglers."
    ),
    classification={
        "intracoupling": (
            "Within the Twin Creek system, groundwater is pumped to a bottling "
            "plant, and local anglers fish the creek."
        ),
        "pericoupling": (
            "Twin and Chippewa creeks, whose watersheds share a boundary, exchange "
            "surface, hyporheic and groundwater flows and fish, and both lose flow "
            "to the same withdrawals."
        ),
        "telecoupling": (
            "A decision at Nestlé's headquarters in Connecticut started the "
            "withdrawals, bottled water reaches markets across Michigan and the "
            "United States, and anglers come from non-adjacent watersheds."
        ),
    },
    systems=[
        {"coupling": "intracoupling", "role": "focal",
         "name": "Twin Creek watershed, Osceola County, Michigan",
         "human_subsystem": "Nestlé's groundwater-bottling operation, landowners and local anglers",
         "natural_subsystem": "A groundwater-fed coldwater stream with brook charr and brown trout",
         "geographic_scope": "The Twin Creek watershed in Michigan's Lower Peninsula"},
        {"coupling": "pericoupling", "role": "sending",
         "name": "Twin Creek watershed (for water)",
         "human_subsystem": "The bottling withdrawals and local water users",
         "natural_subsystem": "Groundwater and surface water shared across the watershed boundary",
         "geographic_scope": "Twin Creek"},
        {"coupling": "pericoupling", "role": "receiving",
         "name": "Chippewa Creek watershed (for water)",
         "human_subsystem": "Landowners and anglers along Chippewa Creek",
         "natural_subsystem": "A stream whose flow depends partly on the shared groundwater",
         "geographic_scope": "The adjacent Chippewa Creek watershed"},
        {"coupling": "pericoupling", "role": "sending",
         "name": "Chippewa Creek (for fish)",
         "human_subsystem": "Anglers fishing both creeks",
         "natural_subsystem": "Salmonids that move downstream into Twin Creek",
         "geographic_scope": "Chippewa Creek"},
        {"coupling": "pericoupling", "role": "receiving",
         "name": "Twin Creek (for fish)",
         "human_subsystem": "Anglers fishing both creeks",
         "natural_subsystem": "Habitat receiving fish from Chippewa Creek",
         "geographic_scope": "Twin Creek"},
        {"coupling": "telecoupling", "role": "sending",
         "name": "Nestlé headquarters, Stamford, Connecticut (for the bottling decision)",
         "human_subsystem": "Corporate managers who chose to bottle groundwater in Michigan for revenue",
         "natural_subsystem": "Not analysed in the paper",
         "geographic_scope": "Stamford, Connecticut"},
        {"coupling": "telecoupling", "role": "receiving",
         "name": "Bottled-water markets across Michigan and the United States",
         "human_subsystem": "Consumers of the bottled groundwater",
         "natural_subsystem": "Not analysed in the paper",
         "geographic_scope": "Michigan and the wider United States"},
        {"coupling": "telecoupling", "role": "spillover",
         "name": "Hoffmeyer Creek, the Hersey River, and stakeholder anglers and landowners",
         "human_subsystem": "Anglers and landowners in Osceola County who oppose the withdrawals",
         "natural_subsystem": "Non-adjacent streams in the same county",
         "geographic_scope": "Osceola County beyond the two creeks"},
    ],
    flows=[
        {"coupling": "intracoupling", "category": "matter",
         "direction": "Aquifer → Nestlé bottling plant",
         "description": "Groundwater withdrawal of 0.016 m³/s (250 gallons per minute); an increase to 0.025 m³/s was approved in April 2018"},
        {"coupling": "intracoupling", "category": "people",
         "direction": "Osceola County residents → local streams",
         "description": "Local anglers: an estimated 415 of the county's 1,079 stream salmonid anglers"},
        {"coupling": "pericoupling", "category": "matter",
         "direction": "Twin Creek → Chippewa Creek",
         "description": "Surface, hyporheic and groundwater flows across the shared watershed boundary, reduced by the withdrawals"},
        {"coupling": "pericoupling", "category": "organisms",
         "direction": "Chippewa Creek → Twin Creek",
         "description": "Salmonids moving downstream; movement from Twin Creek into Chippewa Creek is blocked by an impoundment near Chippewa's confluence with the Muskegon River"},
        {"coupling": "telecoupling", "category": "information",
         "direction": "Nestlé headquarters, Connecticut → Twin and Chippewa watersheds",
         "description": "The corporate decision to bottle groundwater in a groundwater-rich part of Michigan, approved by the state"},
        {"coupling": "telecoupling", "category": "matter",
         "direction": "Twin and Chippewa watersheds → markets in Michigan and the United States",
         "description": "Bottled groundwater"},
        {"coupling": "telecoupling", "category": "people",
         "direction": "Anglers' home areas elsewhere → Osceola County",
         "description": "Nonlocal anglers: an estimated 664 of the county's 1,079 stream salmonid anglers, about 1.6 times as many as local anglers"},
    ],
    agents=[
        {"coupling": "intracoupling", "level": "firms / traders / corporations",
         "name": "Nestlé Waters North America", "description": "Pumps groundwater for bottling"},
        {"coupling": "intracoupling", "level": "individuals / households",
         "name": "Local landowners and anglers", "description": "Use the streams and many oppose the withdrawals"},
        {"coupling": "pericoupling", "level": "non-human agents",
         "name": "Brook charr and brown trout", "description": "Their movement links the two creeks where no barrier blocks it"},
        {"coupling": "telecoupling", "level": "firms / traders / corporations",
         "name": "Nestlé headquarters in Stamford, Connecticut", "description": "Decided to bottle Michigan groundwater"},
        {"coupling": "telecoupling", "level": "governments / policymakers",
         "name": "Michigan Department of Environment, Great Lakes, and Energy (DEGLE)", "description": "Permits the withdrawals and approved the 2018 increase"},
        {"coupling": "telecoupling", "level": "organizations / NGOs",
         "name": "Michigan Citizens for Water Conservation, Michigan Trout Unlimited and the Muskegon River Watershed Council",
         "description": "Advocate for aquatic resource conservation"},
    ],
    causes=[
        {"coupling": "intracoupling", "category": "hydrological",
         "description": "Abundant groundwater in the Twin Creek watershed"},
        {"coupling": "pericoupling", "category": "hydrological",
         "description": "The creeks share groundwater and surface flow across a common watershed boundary"},
        {"coupling": "pericoupling", "category": "technological / infrastructural",
         "description": "An impoundment blocks fish moving from Twin Creek into Chippewa Creek"},
        {"coupling": "telecoupling", "category": "economic",
         "description": "Nestlé sought revenue from bottling groundwater"},
        {"coupling": "telecoupling", "category": "political / institutional",
         "description": "State permitting and groundwater policy"},
        {"coupling": "telecoupling", "category": "climatic / atmospheric",
         "description": "Pollution, climate change and land-use change act at scales beyond the two creeks"},
    ],
    effects=[
        {"coupling": "intracoupling", "category": "hydrological", "system": "Twin Creek, focal system",
         "description": "Baseflow fell by 0.0449 m³/s and the surface water level by 9.8 cm between 2006 and 2015"},
        {"coupling": "intracoupling", "category": "ecological / biological", "system": "Twin Creek, focal system",
         "description": "Studies found minimal effects on salmonid populations so far"},
        {"coupling": "intracoupling", "category": "economic", "system": "the local economy of Osceola and Mecosta counties, focal system",
         "description": "The bottling operation contributed 284 jobs and US$24.2 million in economic activity in 2017"},
        {"coupling": "pericoupling", "category": "hydrological", "system": "Chippewa Creek, receiving system",
         "description": "Baseflow fell by 0.0110 m³/s and the surface water level by 3.7 cm"},
        {"coupling": "telecoupling", "category": "political / institutional", "system": "Michigan's groundwater governance",
         "description": "During a 205-day comment period the state received 340,000 petition signatures against increased extraction and 50,000 statements of concern"},
    ],
    cross_coupling=[
        ("Cascading effects from a distant decision",
         "A telecoupled decision at Nestlé's Connecticut headquarters led to groundwater withdrawal in "
         "Twin Creek (intracoupling), which lowered flows in adjacent Chippewa Creek (pericoupling) and "
         "drew petitions from across the state."),
    ],
    adaptation_note=(
        "In the paper, telecoupling means not adjacent rather than far away: Hoffmeyer Creek and the "
        "Hersey River lie in the same county. Its count of nonlocal anglers combines pericoupled and "
        "telecoupled visitors."
    ),
    keywords=["water", "groundwater", "stream", "river", "watershed", "freshwater",
              "trout", "salmon", "salmonid", "angling", "angler", "recreation",
              "bottled water", "extraction", "dam", "Michigan", "United States"],
)


# ---------------------------------------------------------------------------
# Example 5: Transatlantic wood pellet trade (Parish et al., 2018)
# ---------------------------------------------------------------------------

TRANSATLANTIC_WOOD_PELLETS = FrameworkExample(
    title="Wood Pellets from the Southeastern United States to European Power Plants",
    source=(
        "Parish, E. S., Herzberger, A. J., Phifer, C. C., & Dale, V. H. (2018). "
        "Transatlantic wood pellet trade demonstrates telecoupled benefits. Ecology "
        "and Society, 23(1), 28."
    ),
    corpus_file="Parish et al. - 2018 - Transatlantic wood pellet trade demonstrates telecoupled benefits",
    domain="energy, bioenergy, forestry, climate",
    description=(
        "Parish et al. (2018) use the telecoupling framework to test claims that "
        "Europe's gains from burning wood pellets instead of coal come at the cost "
        "of forests in the southeastern United States (SE US). Drawing on trade "
        "statistics and the literature, they describe the sending system (forests, "
        "16 export pellet mills and six ports in nine SE US states), the receiving "
        "system (mainly the United Kingdom, Belgium and the Netherlands) and three "
        "spillover systems. No partner shares a border, so the analysis has no "
        "pericoupling section."
    ),
    classification={
        "intracoupling": (
            "Within the SE US, pellet mills draw wood from forests and sawmills "
            "within about 120 km, and forest owners earn income from bioenergy "
            "sales."
        ),
        "telecoupling": (
            "Wood pellets cross the Atlantic from SE US ports to power plants in "
            "the United Kingdom, Belgium and the Netherlands, driven by European "
            "renewable-energy targets."
        ),
    },
    systems=[
        {"coupling": "intracoupling", "role": "focal",
         "name": "Southeastern United States",
         "human_subsystem": "Family and institutional forest owners, loggers, truck drivers, pellet mills and port operators",
         "natural_subsystem": "Public and private timberland, including mixed hardwood stands and pine plantations",
         "geographic_scope": "Alabama, Florida, Georgia, Louisiana, Mississippi, North Carolina, South Carolina, Virginia and eastern Texas"},
        {"coupling": "telecoupling", "role": "sending",
         "name": "Southeastern United States (for wood pellets)",
         "human_subsystem": "Sixteen export pellet mills and six shipping ports",
         "natural_subsystem": "Timberland supplying pellet feedstock",
         "geographic_scope": "Ports on the Atlantic Ocean and the Gulf of Mexico"},
        {"coupling": "telecoupling", "role": "receiving",
         "name": "European power sector, mainly the United Kingdom, Belgium and the Netherlands",
         "human_subsystem": "Power producers such as Drax; EU and member-state governments; power consumers",
         "natural_subsystem": "Europe's limited forests, which the imports spare",
         "geographic_scope": "Coal plants converted to burn biomass"},
        {"coupling": "telecoupling", "role": "spillover",
         "name": "The coal industry, the Atlantic Ocean and the global atmosphere",
         "human_subsystem": "US coal producers, including Appalachian mines that export to Europe",
         "natural_subsystem": "The Atlantic Ocean, crossed by pellet shipping, and the global atmosphere, which receives carbon emissions",
         "geographic_scope": "Appalachia, the transatlantic shipping lanes and the global atmosphere"},
    ],
    flows=[
        {"coupling": "intracoupling", "category": "matter",
         "direction": "SE US forests and sawmills → export pellet mills",
         "description": "Wood feedstock from within a radius of about 120 km"},
        {"coupling": "intracoupling", "category": "capital",
         "direction": "Pellet mills → forest owners",
         "description": "Income from sales of wood for bioenergy"},
        {"coupling": "telecoupling", "category": "energy",
         "direction": "SE US ports → the United Kingdom, Belgium and the Netherlands",
         "description": "Wood pellets: 4.6 million tonnes to 34 countries in 2015, 84% of them to the UK and 13% to Belgium"},
        {"coupling": "telecoupling", "category": "capital",
         "direction": "European buyers → SE US exporters",
         "description": "Payments for pellets; the export value rose from US$258 million in 2012 to US$683 million in 2015"},
        {"coupling": "telecoupling", "category": "matter",
         "direction": "Pellet shipping → the Atlantic Ocean and the global atmosphere",
         "description": "Shipping emissions, which one cited estimate puts at 44% of the supply chain's greenhouse-gas emissions"},
    ],
    agents=[
        {"coupling": "intracoupling", "level": "individuals / households",
         "name": "Family forest owners", "description": "Own 87% of SE US forest and decide when to harvest"},
        {"coupling": "intracoupling", "level": "firms / traders / corporations",
         "name": "Pellet producers such as Enviva and Drax Biomass", "description": "Run export mills and have set up funds to conserve bottomland forests"},
        {"coupling": "telecoupling", "level": "firms / traders / corporations",
         "name": "Drax and other European power producers", "description": "Burn imported pellets in converted coal plants"},
        {"coupling": "telecoupling", "level": "governments / policymakers",
         "name": "European Union and member-state governments", "description": "Set renewable-energy targets and subsidise biomass power"},
        {"coupling": "telecoupling", "level": "organizations / NGOs",
         "name": "Environmental NGOs and forest-certification programs (FSC, SFI, PEFC)", "description": "Contest or certify the sustainability of pellet feedstock"},
    ],
    causes=[
        {"coupling": "intracoupling", "category": "economic",
         "description": "After the 2007 housing crash and the decline of pulp and paper, nearly 1,000 US wood-processing mills closed, so new wood markets help keep land in forest"},
        {"coupling": "telecoupling", "category": "political / institutional",
         "description": "EU targets of 20% renewable energy by 2020 and 27% by 2030, and accounting rules that treat biomass energy as carbon neutral"},
        {"coupling": "telecoupling", "category": "technological / infrastructural",
         "description": "Retrofitting coal plants to burn biomass is relatively simple and cheap"},
        {"coupling": "telecoupling", "category": "ecological / biological",
         "description": "Europe's limited forest resources"},
    ],
    effects=[
        {"coupling": "intracoupling", "category": "ecological / biological", "system": "SE US forests, focal system",
         "description": "Hardwood inventory grew 0.1% a year in the Atlantic region and 1.3% a year in the Gulf region in 2000-2014, despite claims that pine plantations replace hardwoods; pellets are only about 3% of timberland removals, so their effect is hard to isolate"},
        {"coupling": "telecoupling", "category": "technological / infrastructural", "system": "United Kingdom, receiving system",
         "description": "UK biomass capacity rose 16-fold and coal-fired generation fell 88% since 2010"},
        {"coupling": "telecoupling", "category": "climatic / atmospheric", "system": "the global atmosphere, spillover system",
         "description": "Importing countries can report greenhouse-gas reductions of up to 78% for electricity, but the net reduction is uncertain"},
        {"coupling": "telecoupling", "category": "economic", "system": "the US coal industry, spillover system",
         "description": "Europe cut its imports of Appalachian coal by 50% in five years, partly because of renewables"},
    ],
    cross_coupling=[
        ("Displacement of coal",
         "Pellets displaced coal in European power plants, and coal demand fell for US producers, though "
         "the paper cannot separate the pellets' share from cheap natural gas and other renewables."),
        ("Amplification between telecoupling and intracoupling",
         "European demand gave SE US forest owners a market after mill closures, which the authors, citing "
         "earlier work, see as critical to keeping land in forest."),
    ],
    adaptation_note=(
        "The paper analyses telecoupling only; the intracoupling section is built from the flows it "
        "describes within the sending system."
    ),
    keywords=["energy", "bioenergy", "biomass", "renewable", "wood", "pellet",
              "forest", "forestry", "timber", "coal", "carbon", "emissions",
              "climate", "electricity", "power", "Europe", "United Kingdom"],
)


# ---------------------------------------------------------------------------
# All Examples
# ---------------------------------------------------------------------------

# Order matters: when a research description matches no keyword, the first
# examples are used.
ALL_EXAMPLES: list[FrameworkExample] = [
    WOLONG_METACOUPLING,
    SOUTH_AMERICA_FOOD_TRADE,
    GLOBAL_MARINE_FISHING,
    MICHIGAN_GROUNDWATER_STREAMS,
    TRANSATLANTIC_WOOD_PELLETS,
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


def _system_label(coupling: str, role: str) -> str:
    if role == "focal":
        return "Focal System"
    if role == "spillover":
        return "Spillover System"
    scale = " (adjacent)" if coupling == "pericoupling" else " (distant)"
    return f"{role.title()} System{scale}"


def _by_category(items: list[dict[str, str]]) -> list[tuple[str, list[dict[str, str]]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for item in items:
        groups.setdefault(item["category"], []).append(item)
    order = list(CATEGORIES)
    return sorted(groups.items(), key=lambda kv: order.index(kv[0]))


def format_example(example: FrameworkExample) -> str:
    """Format a single example for prompt injection, in the analysis's own structure."""
    lines: list[str] = [f"### {example.title}", f"Source: {example.source}"]
    if example.adaptation_note:
        lines.append(f"Adaptation: {example.adaptation_note}")
    lines.append(f"\n{example.description}\n")

    lines.append("**Coupling Classification**")
    for coupling in example.coupling_types:
        lines.append(f"- **{coupling.title()}**: {example.classification[coupling]}")

    for coupling in example.coupling_types:
        lines.append(f"\n**{_SECTION_HEADINGS[coupling]}**")

        systems = [s for s in example.systems if s["coupling"] == coupling]
        if systems:
            lines.append("*Systems*")
        for system in systems:
            lines.append(f"**{_system_label(coupling, system['role'])}**: {system['name']}")
            lines.append(f"- **Human subsystem**: {system['human_subsystem']}")
            lines.append(f"- **Natural subsystem**: {system['natural_subsystem']}")
            lines.append(f"- **Geographic scope**: {system['geographic_scope']}")

        flows = [f for f in example.flows if f["coupling"] == coupling]
        if flows:
            lines.append("*Flows*")
        for flow in flows:
            lines.append(f"**{flow['category'].title()} Flow**")
            lines.append(f"- **Direction**: {flow['direction']}")
            lines.append(f"- **Description**: {flow['description']}")

        agents = [a for a in example.agents if a["coupling"] == coupling]
        if agents:
            lines.append("*Agents*")
        for agent in agents:
            lines.append(
                f"- [{AGENT_LEVELS[agent['level']]}] {agent['name']}: {agent['description']}"
            )

        causes = [c for c in example.causes if c["coupling"] == coupling]
        if causes:
            lines.append("*Causes*")
        for category, items in _by_category(causes):
            lines.append(f"**{CATEGORIES[category]}**")
            lines.extend(f"- {item['description']}" for item in items)

        effects = [e for e in example.effects if e["coupling"] == coupling]
        if effects:
            lines.append("*Effects*")
        for category, items in _by_category(effects):
            lines.append(f"**{CATEGORIES[category]}**")
            lines.extend(
                f"- {item['description']} (applies to: {item['system']})" for item in items
            )

    if example.cross_coupling:
        lines.append("\n**Cross-coupling Interactions**")
        lines.extend(f"- **{label}**: {text}" for label, text in example.cross_coupling)

    return "\n".join(lines)

"""English search aliases and general business descriptions, not live filings."""

PROFILES = {
    '005930': ('Samsung Electronics', 'Designs and manufactures semiconductors, smartphones, televisions, home appliances, and other electronics.'),
    '005935': ('Samsung Electronics Preferred', 'Preferred shares of Samsung Electronics, an electronics and semiconductor manufacturer.'),
    '000660': ('SK hynix', 'Develops and manufactures memory semiconductors including DRAM, NAND flash, and high-bandwidth memory.'),
    '035420': ('NAVER', 'Operates internet search, digital advertising, commerce, content, and cloud services.'),
    '035720': ('Kakao', 'Operates messaging, digital content, advertising, commerce, and other platform services.'),
    '005380': ('Hyundai Motor', 'Manufactures passenger and commercial vehicles, including electric and hydrogen vehicles.'),
    '000270': ('Kia', 'Designs, manufactures, and sells passenger vehicles and mobility products.'),
    '373220': ('LG Energy Solution', 'Develops and manufactures rechargeable batteries for electric vehicles, energy storage, and other applications.'),
    '006400': ('Samsung SDI', 'Manufactures rechargeable batteries and electronic materials.'),
    '051910': ('LG Chem', 'Produces petrochemicals, advanced materials, and life-science products.'),
    '207940': ('Samsung Biologics', 'Provides contract development and manufacturing services for biologic medicines.'),
    '068270': ('Celltrion', 'Develops, manufactures, and markets biologic medicines and biosimilars.'),
    '247540': ('Ecopro BM', 'Produces cathode materials used in rechargeable lithium-ion batteries.'),
    '086520': ('Ecopro', 'A holding company with businesses in battery materials and environmental technologies.'),
    '005490': ('POSCO Holdings', 'A holding company with steel, battery-materials, and related industrial businesses.'),
    '055550': ('Shinhan Financial Group', 'A financial holding group with banking, securities, insurance, and card businesses.'),
    '105560': ('KB Financial Group', 'A financial holding group with banking, securities, insurance, and asset-management businesses.'),
    '086790': ('Hana Financial Group', 'A financial group providing banking, securities, credit-card, and related services.'),
    '010950': ('S-Oil', 'Operates oil refining, petrochemicals, and lubricant-related businesses.'),
    '010955': ('S-Oil Preferred', 'Preferred shares of S-Oil, an oil refining and petrochemicals company.'),
    '003490': ('Korean Air', 'Provides passenger and cargo air transportation and aerospace-related services.'),
    '180640': ('Hanjin KAL', 'A holding company with aviation and related transportation interests.'),
    '009150': ('Samsung Electro-Mechanics', 'Manufactures electronic components including capacitors, camera modules, and package substrates.'),
    '402340': ('SK Square', 'An investment company with technology and semiconductor-related holdings.'),
    '021240': ('Coway', 'Provides water and air purification products and home-appliance rental services.'),
    '282330': ('BGF Retail', 'Operates convenience-store retail and related distribution businesses.'),
    '031980': ('PSK Holdings', 'Supplies semiconductor manufacturing equipment and related services.'),
    '067290': ('JW Shinyak', 'Develops and markets pharmaceutical products.'),
}


def english_name(code, original):
    return PROFILES.get(str(code), (str(original), ''))[0]


def overview(code):
    profile = PROFILES.get(str(code))
    if profile:
        return profile[1] + ' This is a general business summary; confirm current segments in the latest company disclosures.'
    return 'An English business summary is not available for this stock. Use the source link to review the original company overview and current disclosures.'

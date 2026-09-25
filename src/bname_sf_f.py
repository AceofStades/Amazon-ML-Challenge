import pandas as pd

# Paths
input_freq_tsv = r"D:\claudesplan\data\dataset_cleaned\short_forms_frequency.tsv"
output_valid_map_tsv = r"D:\claudesplan\data\dataset_cleaned\valid_sf_lf_map.tsv"

# Comprehensive Master Business & Corporate Abbreviation Map
MASTER_SF_TO_LF = {
    # --- Legal & Entity Structures ---
    "pvt": "private", "prvt": "private", "p": "private",
    "ltd": "limited", "lt": "limited",
    "inc": "incorporated", "corp": "corporation", "co": "company",
    "llc": "limited liability company", "llp": "limited liability partnership",
    "plc": "public limited company", "gmbh": "gesellschaft mit beschrankter haftung",
    "sdn": "sendirian", "bhd": "berhad", "sa": "sociedad anonima",
    
    # --- Operations & Structure ---
    "grp": "group", "ent": "enterprises", "ep": "enterprises",
    "assoc": "associates", "assn": "association", "soc": "society",
    "dept": "department", "div": "division", "inst": "institute",
    "ctr": "center", "cntr": "center", "centre": "center",
    
    # --- Industry & Sector Terms ---
    "mfg": "manufacturing", "mfr": "manufacturer", "mfrs": "manufacturers",
    "tech": "technologies", "techno": "technology",
    "intl": "international", "int": "international",
    "serv": "services", "svc": "services", "svcs": "services",
    "soln": "solution", "solns": "solution", "solutions": "solution",
    "dev": "development", "mgmt": "management", "mgt": "management",
    "ind": "industries", "inds": "industries", "indust": "industries",
    "eng": "engineering", "engg": "engineering", "engin": "engineering",
    "sys": "systems", "syst": "systems", "prod": "products", "prods": "products",
    "trad": "trading", "trdg": "trading", "dist": "distribution", "distr": "distributors",
    "supp": "supply", "supps": "supplies", "log": "logistics", "logist": "logistics",
    "comm": "communications", "telecom": "telecommunication",
    "cons": "consulting", "consult": "consultants", "cnstr": "construction", "const": "construction",

    # --- Finance, Real Estate & Health ---
    "fin": "financial", "cap": "capital", "inv": "investments", "invest": "investments",
    "prop": "properties", "realty": "real estate", "re": "real estate", "ins": "insurance",
    "pharma": "pharmaceuticals", "pharm": "pharmaceutical", "med": "medical",
    "hosp": "hospital", "lab": "laboratories", "labs": "laboratories", "chem": "chemicals"
}

# 1. Read frequency file
freq_df = pd.read_csv(input_freq_tsv, sep="\t")

# 2. Filter valid short forms present in dataset
freq_df["long_form"] = freq_df["short_form"].map(MASTER_SF_TO_LF)
valid_df = freq_df.dropna(subset=["long_form"])[["short_form", "long_form", "frequency"]]

# 3. Save valid mappings
valid_df.to_csv(output_valid_map_tsv, sep="\t", index=False)

print(f"--- Short Form Extraction Complete ---")
print(f"Total Valid Short Forms Found in Data: {len(valid_df):,}")
print(f"Saved Valid Map to: {output_valid_map_tsv}\n")

print(valid_df.head(20))
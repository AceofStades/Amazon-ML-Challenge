import os
import re
import polars as pl

# -------------------------------------------------------------------------
# Directory Configuration
# -------------------------------------------------------------------------
CLEAN_DIR = r"D:\claudesplan\data\dataset_clean\train"

# -------------------------------------------------------------------------
# Gazetteers & Standardizers
# -------------------------------------------------------------------------
COUNTRY_MAP = {
    "us": "united states", "usa": "united states", "in": "india", "ind": "india",
    "uk": "united kingdom", "gb": "united kingdom", "ca": "canada", "au": "australia",
    "de": "germany", "fr": "france"
}

INDIA_STATES = {
    "gj": "gujarat", "dl": "delhi", "ka": "karnataka", "mh": "maharashtra", "tn": "tamil nadu",
    "kl": "kerala", "wb": "west bengal", "ts": "telangana", "ap": "andhra pradesh", "up": "uttar pradesh",
    "mp": "madhya pradesh", "rj": "rajasthan", "pb": "punjab", "hr": "haryana", "or": "odisha",
    "od": "odisha", "br": "bihar", "jh": "jharkhand", "cg": "chhattisgarh", "ct": "chhattisgarh",
    "ga": "goa", "hp": "himachal pradesh", "jk": "jammu and kashmir", "uk": "uttarakhand", "ut": "uttarakhand",
    "as": "assam", "ml": "meghalaya", "mn": "manipur", "mz": "mizoram", "nl": "nagaland",
    "tr": "tripura", "sk": "sikkim", "py": "puducherry", "ch": "chandigarh"
}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri",
    "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio",
    "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming"
}

STATE_MAP = {**INDIA_STATES, **US_STATES}
CITY_ALIASES = {
    "ahmedbad": "ahmedabad", "trivandrum": "thiruvananthapuram", 
    "baroda": "vadodara", "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai"
}

ROAD_KEYWORDS = r"\b(road|rd|marg|path|lane|street|st|highway|hwy|bypass|expressway|flyover|circle|cross)\b"
AREA_KEYWORDS = r"\b(nagar|colony|society|soc|enclave|layout|sector|sec|phase|block|dist|district|east|west|north|south|extension|extn|gali|vihar|park|bazar|bazaar|market|mrg)\b"
BUILDING_LANDMARK_KEYWORDS = r"\b(flat|shop|plot|house|h\.no|no|bldg|building|tower|complex|plaza|premise|premises|near|opp|opposite|behind|beside|adjacent|above|below)\b"

def sanitize_str(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() in ["nan", "null", "none", ""]:
        return ""
    return s

def standardize_country(val) -> str:
    cleaned = sanitize_str(val).lower()
    if not cleaned:
        return ""
    return COUNTRY_MAP.get(cleaned, cleaned)

def parse_address_robust(raw_address, existing_country="", existing_state="") -> dict:
    country = standardize_country(existing_country)
    raw = sanitize_str(raw_address).lower()

    if not raw:
        clean_state = sanitize_str(existing_state).lower()
        return {
            "building_landmark": "", "road_area": "", "street_address": "",
            "city": "", "state": clean_state, "country": country, "address_cleaned": country
        }

    # Extract country if missing
    if not country:
        for code, full_c in COUNTRY_MAP.items():
            if re.search(rf"\b{code}\b$", raw) or re.search(rf"\b{full_c}\b$", raw):
                country = full_c
                raw = re.sub(rf"\b({code}|{full_c})\b$", "", raw).strip()
                break

    # Determine state map subset to prevent foreign state match (e.g., missouri in India)
    allowed_states = set(INDIA_STATES.values()) if country == "india" else (set(US_STATES.values()) if country == "united states" else set(STATE_MAP.values()))

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    cleaned_parts = []
    found_state = sanitize_str(existing_state).lower()
    if country == "india" and found_state not in set(INDIA_STATES.values()):
        found_state = ""

    for part in parts:
        tokens = part.split()
        new_tokens = []
        for t in tokens:
            t = t.strip()
            if t in CITY_ALIASES:
                t = CITY_ALIASES[t]
            if t in STATE_MAP:
                st_candidate = STATE_MAP[t]
                if st_candidate in allowed_states:
                    if not found_state:
                        found_state = st_candidate
                    t = st_candidate
            elif t in allowed_states:
                if not found_state:
                    found_state = t
            new_tokens.append(t)

        cleaned_str = " ".join(new_tokens)
        if cleaned_str and cleaned_str != "null":
            cleaned_parts.append(cleaned_str)

    # Deduplicate parts preserving order
    unique_parts = []
    seen = set()
    for p in cleaned_parts:
        if p not in seen:
            unique_parts.append(p)
            seen.add(p)

    non_state_country_parts = [p for p in unique_parts if p not in allowed_states and p != country]

    building_landmark_parts = []
    road_area_parts = []
    city = ""

    if country == "india":
        if non_state_country_parts:
            # City is typically the last element among geographic parts in Indian addresses
            city = non_state_country_parts[-1]
            remaining = non_state_country_parts[:-1]
        else:
            remaining = []

        for part in remaining:
            if re.search(ROAD_KEYWORDS, part) or re.search(AREA_KEYWORDS, part):
                road_area_parts.append(part)
            elif re.search(BUILDING_LANDMARK_KEYWORDS, part) or re.search(r"\d+", part):
                building_landmark_parts.append(part)
            else:
                road_area_parts.append(part)

        building_landmark = ", ".join(building_landmark_parts)
        road_area = ", ".join(road_area_parts)
        street_address = ", ".join([p for p in [building_landmark, road_area] if p])
    else:
        building_landmark = ""
        road_area = ""
        
        # Guard against inverted street/city formats in US addresses
        if non_state_country_parts:
            candidate = non_state_country_parts[-1]
            if re.search(r"\d+\s+", candidate) or re.search(ROAD_KEYWORDS, candidate):
                # Candidate is actually a street name, swap order
                street_address = candidate
                city = non_state_country_parts[0] if len(non_state_country_parts) > 1 else ""
            else:
                city = candidate
                street_address = ", ".join(non_state_country_parts[:-1]) if len(non_state_country_parts) > 1 else non_state_country_parts[0]
        else:
            street_address = ""

    components = [c for c in [street_address, city, found_state, country] if c]
    address_cleaned = ", ".join(components)

    return {
        "building_landmark": building_landmark,
        "road_area": road_area,
        "street_address": street_address,
        "city": city,
        "state": found_state,
        "country": country,
        "address_cleaned": address_cleaned
    }

# -------------------------------------------------------------------------
# Master Schema Specification
# -------------------------------------------------------------------------
MASTER_COLUMNS = [
    "id",
    "business_name",
    "building_landmark",
    "road_area",
    "street_address",
    "city",
    "state",
    "country",
    "address_cleaned",
    "business_name_is_modified",
    "business_name_word_count",
    "address_cleaned_word_count"
]

# -------------------------------------------------------------------------
# Pipeline Execution (In-Place Fix)
# -------------------------------------------------------------------------
partitions = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]

for filename in partitions:
    file_path = os.path.join(CLEAN_DIR, filename)

    if not os.path.exists(file_path):
        continue

    print(f"Applying robustness fixes to: {file_path}...")
    df = pl.read_csv(file_path, separator="\t")

    id_col = "id" if "id" in df.columns else df.columns[0]
    if id_col != "id":
        df = df.rename({id_col: "id"})

    addr_col = "address_cleaned" if "address_cleaned" in df.columns else "address"

    parsed_records = []
    for addr_val, c_val, s_val in zip(df[addr_col], df["country"], df["state"]):
        parsed_records.append(parse_address_robust(addr_val, existing_country=c_val, existing_state=s_val))

    addr_struct = pl.Series(name="addr_struct", values=parsed_records)

    df = df.with_columns([
        addr_struct.struct.field("building_landmark").alias("building_landmark"),
        addr_struct.struct.field("road_area").alias("road_area"),
        addr_struct.struct.field("street_address").alias("street_address"),
        addr_struct.struct.field("city").alias("city"),
        addr_struct.struct.field("state").alias("state"),
        addr_struct.struct.field("country").alias("country"),
        addr_struct.struct.field("address_cleaned").alias("address_cleaned"),
    ])

    # Recalculate Word Counts
    df = df.with_columns([
        pl.col("address_cleaned")
        .fill_null("").str.strip_chars().str.split(by=" ")
        .list.eval(pl.element().filter(pl.element() != "")).list.len()
        .cast(pl.Int16).alias("address_cleaned_word_count"),
        
        pl.col("business_name")
        .fill_null("").str.strip_chars().str.split(by=" ")
        .list.eval(pl.element().filter(pl.element() != "")).list.len()
        .cast(pl.Int16).alias("business_name_word_count")
    ])

    # Standardize Column Output
    exprs = []
    for col_name in MASTER_COLUMNS:
        if col_name in ["business_name_is_modified", "business_name_word_count", "address_cleaned_word_count"]:
            exprs.append(pl.col(col_name).fill_null(0).cast(pl.Int16))
        else:
            exprs.append(pl.col(col_name).fill_null("").cast(pl.Utf8))

    df_clean = df.select(exprs)[MASTER_COLUMNS]
    df_clean.write_csv(file_path, separator="\t")
    print(f"Refined successfully: {file_path}")

print("--- Robustness fixes applied to dataset_clean/train ---")
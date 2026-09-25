import os
import re
import polars as pl

# -------------------------------------------------------------------------
# Directory & File Configuration
# -------------------------------------------------------------------------
CLEAN_DIR = r"D:\claudesplan\data\dataset_clean\train"
PARTITIONS = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]

# -------------------------------------------------------------------------
# Gazetteers & Keywords
# -------------------------------------------------------------------------
COUNTRY_MAP = {"us": "united states", "usa": "united states", "in": "india", "ind": "india"}

INDIA_STATES = {
    "gj": "gujarat", "dl": "delhi", "ka": "karnataka", "mh": "maharashtra", "tn": "tamil nadu",
    "kl": "kerala", "wb": "west bengal", "ts": "telangana", "ap": "andhra pradesh", "up": "uttar pradesh",
    "mp": "madhya pradesh", "rj": "rajasthan", "pb": "punjab", "hr": "haryana", "or": "odisha",
    "br": "bihar", "jh": "jharkhand", "cg": "chhattisgarh", "ga": "goa", "hp": "himachal pradesh",
    "jk": "jammu and kashmir", "uk": "uttarakhand", "as": "assam", "ch": "chandigarh"
}

ROAD_KEYWORDS = r"\b(road|rd|marg|path|lane|street|st|highway|hwy|bypass|expressway|flyover|circle|cross)\b"
AREA_KEYWORDS = r"\b(nagar|colony|society|soc|enclave|layout|sector|sec|phase|block|dist|district|east|west|north|south|extension|extn|gali|vihar|park|bazar|bazaar|market|mrg)\b"
BUILDING_LANDMARK_KEYWORDS = r"\b(flat|shop|plot|house|h\.no|hn|no|bldg|building|tower|complex|plaza|premise|premises|near|opp|opposite|behind|beside|adjacent|above|below)\b"

def sanitize_str(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() in ["nan", "null", "none", ""]:
        return ""
    return s

def deduplicate_address_fields(building_lm, road_ar, street_addr, city_val, state_val, country_val):
    """
    Cleans and segregates address components without repeating text across columns.
    """
    bldg = sanitize_str(building_lm).lower()
    road = sanitize_str(road_ar).lower()
    street = sanitize_str(street_addr).lower()
    city = sanitize_str(city_val).lower()
    state = INDIA_STATES.get(sanitize_str(state_val).lower(), sanitize_str(state_val).lower())
    country = COUNTRY_MAP.get(sanitize_str(country_val).lower(), sanitize_str(country_val).lower())

    # Collect all unique tokens/parts across existing fields
    combined_raw = ", ".join([p for p in [bldg, road, street] if p])
    raw_parts = [p.strip() for p in combined_raw.split(",") if p.strip()]

    # Deduplicate sequential raw parts
    unique_parts = []
    seen = set()
    for p in raw_parts:
        norm = " ".join(p.split())
        if norm and norm not in seen and norm not in [city, state, country]:
            unique_parts.append(norm)
            seen.add(norm)

    bldg_parts = []
    road_parts = []
    street_parts = []

    for part in unique_parts:
        if re.search(BUILDING_LANDMARK_KEYWORDS, part) or re.search(r"\b(hn|h\.no|no|tower|flat|plot)\b", part) or re.search(r"\d+", part):
            bldg_parts.append(part)
        elif re.search(ROAD_KEYWORDS, part) or re.search(AREA_KEYWORDS, part):
            road_parts.append(part)
        else:
            street_parts.append(part)

    final_bldg = ", ".join(bldg_parts)
    final_road = ", ".join(road_parts)
    final_street = ", ".join(street_parts)

    # Build street_address carefully: if street_parts is empty, keep it distinct
    if not final_street:
        # street_address receives combined structure only if needed for standard fallback
        full_street_line = ", ".join([p for p in [final_bldg, final_road] if p])
    else:
        full_street_line = final_street

    # Build address_cleaned without ANY duplicate tokens
    clean_components = []
    for comp in [final_bldg, final_road, final_street, city, state, country]:
        if comp and comp not in clean_components:
            clean_components.append(comp)

    address_cleaned = ", ".join(clean_components)

    return {
        "building_landmark": final_bldg,
        "road_area": final_road,
        "street_address": full_street_line,
        "city": city,
        "state": state,
        "country": country,
        "address_cleaned": address_cleaned
    }

# -------------------------------------------------------------------------
# Main Execution Strategy
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

for filename in PARTITIONS:
    file_path = os.path.join(CLEAN_DIR, filename)
    if not os.path.exists(file_path):
        continue

    print(f"Fixing duplication in: {filename}...")
    df = pl.read_csv(file_path, separator="\t")

    parsed_records = []
    for b_lm, r_ar, s_ad, c_val, st_val, co_val in zip(
        df["building_landmark"], df["road_area"], df["street_address"],
        df["city"], df["state"], df["country"]
    ):
        parsed_records.append(deduplicate_address_fields(b_lm, r_ar, s_ad, c_val, st_val, co_val))

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

    # Re-calculate correct word count metrics
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

    # Cast types and enforce strict schema
    exprs = []
    for col in MASTER_COLUMNS:
        if col in ["business_name_is_modified", "business_name_word_count", "address_cleaned_word_count"]:
            exprs.append(pl.col(col).fill_null(0).cast(pl.Int16))
        else:
            exprs.append(pl.col(col).fill_null("").cast(pl.Utf8))

    df_clean = df.select(exprs)[MASTER_COLUMNS]
    df_clean.write_csv(file_path, separator="\t")
    print(f"Successfully processed and overwritten: {filename}")

print("\n--- Fix complete! Duplication resolved across all partition files. ---")
import os
import re
import polars as pl

# -------------------------------------------------------------------------
# Directory Configuration
# -------------------------------------------------------------------------
INPUT_DIR = r"D:\claudesplan\data\dataset_formatted\train"
OUTPUT_DIR = r"D:\claudesplan\data\address_formatted\train"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Dynamic Gazetteers & Extensible Mappings
# -------------------------------------------------------------------------
COUNTRY_MAP = {
    "us": "united states",
    "usa": "united states",
    "in": "india",
    "ind": "india",
    "uk": "united kingdom",
    "gb": "united kingdom",
    "ca": "canada",
    "au": "australia",
    "de": "germany",
    "fr": "france"
}

STATE_MAP = {
    # India
    "gj": "gujarat", "dl": "delhi", "ka": "karnataka", "mh": "maharashtra", "tn": "tamil nadu",
    "kl": "kerala", "wb": "west bengal", "ts": "telangana", "ap": "andhra pradesh", "up": "uttar pradesh",
    "mp": "madhya pradesh", "rj": "rajasthan", "pb": "punjab", "hr": "haryana", "or": "odisha",
    "od": "odisha", "br": "bihar", "jh": "jharkhand", "cg": "chhattisgarh", "ct": "chhattisgarh",
    "ga": "goa", "hp": "himachal pradesh", "jk": "jammu and kashmir", "uk": "uttarakhand", "ut": "uttarakhand",
    "as": "assam", "ml": "meghalaya", "mn": "manipur", "mz": "mizoram", "nl": "nagaland",
    "tr": "tripura", "sk": "sikkim", "py": "puducherry", "ch": "chandigarh",
    # US
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

ALL_KNOWN_STATES = set(STATE_MAP.values())
CITY_ALIASES = {
    "ahmedbad": "ahmedabad", "trivandrum": "thiruvananthapuram", 
    "baroda": "vadodara", "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai"
}

# -------------------------------------------------------------------------
# Dynamic Normalizers (Future-Proof Passthrough)
# -------------------------------------------------------------------------
def standardize_country(val: str) -> str:
    """Standardizes known codes/abbreviations while preserving ANY arbitrary country string."""
    if not val or str(val).strip().lower() in ["nan", "null", "none", ""]:
        return ""
    
    cleaned_val = str(val).strip().lower()
    # Normalize known aliases (us -> united states), else retain original value exactly
    return COUNTRY_MAP.get(cleaned_val, cleaned_val)

def parse_address_dynamic(raw_address: str, existing_country: str = "", existing_state: str = "") -> dict:
    """Parses address components dynamically while protecting existing country & state values."""
    country = standardize_country(existing_country)
    
    if not raw_address or str(raw_address).strip().lower() in ["nan", "null", ""]:
        return {
            "street_address": "",
            "city": "",
            "state": existing_state.strip().lower() if existing_state else "",
            "country": country,
            "address_cleaned": country
        }

    raw = str(raw_address).lower().strip()

    # Fallback country extraction from address text ONLY if country was completely missing
    if not country:
        for code, full_c in COUNTRY_MAP.items():
            if re.search(rf"\b{code}\b$", raw) or re.search(rf"\b{full_c}\b$", raw):
                country = full_c
                raw = re.sub(rf"\b({code}|{full_c})\b$", "", raw).strip()
                break

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    cleaned_parts = []
    found_state = existing_state.strip().lower() if existing_state else ""

    for part in parts:
        tokens = part.split()
        new_tokens = []
        for t in tokens:
            t = t.strip()
            if t in CITY_ALIASES:
                t = CITY_ALIASES[t]
            if t in STATE_MAP:
                if not found_state:
                    found_state = STATE_MAP[t]
                t = STATE_MAP[t]
            elif t in ALL_KNOWN_STATES:
                if not found_state:
                    found_state = t
            new_tokens.append(t)

        cleaned_str = " ".join(new_tokens)
        if cleaned_str and cleaned_str != "null":
            cleaned_parts.append(cleaned_str)

    unique_parts = []
    seen = set()
    for p in cleaned_parts:
        if p not in seen:
            unique_parts.append(p)
            seen.add(p)

    street_parts = [p for p in unique_parts if p not in ALL_KNOWN_STATES and p != country]
    
    if street_parts:
        city = street_parts[-1]
        street = ", ".join(street_parts[:-1]) if len(street_parts) > 1 else street_parts[0]
    else:
        street = ""
        city = ""

    # Build clean full address line without introducing blank commas
    components = [c for c in [street, city, found_state, country] if c]
    address_cleaned = ", ".join(components)

    return {
        "street_address": street,
        "city": city,
        "state": found_state,
        "country": country,
        "address_cleaned": address_cleaned
    }

# -------------------------------------------------------------------------
# Pipeline Execution
# -------------------------------------------------------------------------
partitions = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]

for filename in partitions:
    input_file = os.path.join(INPUT_DIR, filename)
    output_file = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(input_file):
        print(f"Skipping missing partition: {input_file}")
        continue

    print(f"Processing {input_file} -> Generating {output_file}...")
    df = pl.read_csv(input_file, separator="\t")

    addr_col = "address" if "address" in df.columns else ("business_address" if "business_address" in df.columns else None)
    has_country = "country" in df.columns
    has_state = "state" in df.columns

    if addr_col:
        parsed_records = []
        
        # Zip row values dynamically depending on existing columns
        country_vals = df["country"] if has_country else [None] * len(df)
        state_vals = df["state"] if has_state else [None] * len(df)

        for addr_val, c_val, s_val in zip(df[addr_col], country_vals, state_vals):
            parsed_records.append(parse_address_dynamic(addr_val, existing_country=c_val, existing_state=s_val))

        addr_struct = pl.Series(name="addr_struct", values=parsed_records)

        df = df.with_columns([
            addr_struct.struct.field("street_address").alias("street_address"),
            addr_struct.struct.field("city").alias("city"),
            addr_struct.struct.field("state").alias("state"),
            addr_struct.struct.field("country").alias("country"),
            addr_struct.struct.field("address_cleaned").alias("address_cleaned"),
        ])

    # Word count metadata
    df = df.with_columns(
        pl.col("address_cleaned")
        .fill_null("").str.strip_chars().str.split(by=" ")
        .list.eval(pl.element().filter(pl.element() != "")).list.len()
        .cast(pl.Int16).alias("address_cleaned_word_count")
    )

    df.write_csv(output_file, separator="\t")
    print(f"Saved: {output_file} | Total Rows: {df.height:,}\n")

print("--- Successfully processed address_formatted with full future-proof location protection! ---")
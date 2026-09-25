import os
import re
import polars as pl

CLEAN_DIR = r"D:\claudesplan\data\dataset_clean\train"
PARTITIONS = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]

# General patterns for components (Language/Location Agnostic)
ROAD_KEYWORDS = r"\b(road|rd|drive|dr|avenue|ave|street|st|lane|ln|boulevard|blvd|way|court|ct|circle|cir|highway|hwy|marg|path|bypass|expressway|crossing|cross|route|rte|str|strasse|rue|avenida|piazza)\b"
AREA_KEYWORDS = r"\b(nagar|colony|society|soc|enclave|layout|sector|sec|phase|block|vihar|park|bazar|bazaar|market|estate|town|district|dist|ward|zone|suburb|borough)\b"
BLDG_KEYWORDS = r"\b(flat|shop|plot|house|h\.no|hn|no|unit|apt|apartment|bldg|building|tower|complex|plaza|premise|premises|floor|level|suite|ste|room|rm|near|opp|opposite|behind|beside|adjacent|above|below|sakar|bhawan|hospital|hotel|hall|center|centre|hub)\b"

# Pattern to detect numeric/unit numbers accidentally put in City/State
NUMBER_OR_CODE_PAT = r"^#?\s*[\d/-]+[a-zA-Z]?$"

def extract_tokens(text: str) -> set:
    """Extract normalized alphanumeric words from text."""
    return set(re.findall(r"\b\w+\b", str(text).lower()))

def dynamic_rebalance_row(row: dict) -> dict:
    bldg = str(row.get("building_landmark") or "").strip()
    road = str(row.get("road_area") or "").strip()
    city = str(row.get("city") or "").strip()
    state = str(row.get("state") or "").strip()
    country = str(row.get("country") or "").strip()

    # -----------------------------------------------------------------
    # Rule 1: Fix City Field Contamination
    # If `city` looks like a plot/house number (e.g. "# 53/1", "123"), move to bldg
    # -----------------------------------------------------------------
    if re.match(NUMBER_OR_CODE_PAT, city):
        if city and city not in bldg:
            bldg = f"{bldg}, {city}".strip(", ")
        city = ""

    # -----------------------------------------------------------------
    # Rule 2: Token Matching for Dynamic City Deduplication
    # If the exact city name (or parts of it) exists inside building/road, remove it from bldg/road
    # -----------------------------------------------------------------
    city_tokens = extract_tokens(city)
    
    # Analyze parts in building_landmark and road_area
    raw_parts = [p.strip() for p in f"{bldg}, {road}".split(",") if p.strip()]
    
    clean_bldg_parts = []
    clean_road_parts = []

    for part in raw_parts:
        part_lower = part.lower()
        part_tokens = extract_tokens(part)

        # Check if this part is strictly a duplicate of the city field
        if city_tokens and part_tokens and part_tokens.issubset(city_tokens):
            # Drop from address body to avoid duplicate weighting
            continue

        # Route to road_area if it contains thoroughfare/area keywords
        has_road = bool(re.search(ROAD_KEYWORDS, part_lower))
        has_area = bool(re.search(AREA_KEYWORDS, part_lower))
        has_bldg = bool(re.search(BLDG_KEYWORDS, part_lower))

        if (has_road or has_area) and not (has_bldg and not has_road):
            if part not in clean_road_parts:
                clean_road_parts.append(part)
        else:
            if part not in clean_bldg_parts:
                clean_bldg_parts.append(part)

    final_bldg = ", ".join(clean_bldg_parts)
    final_road = ", ".join(clean_road_parts)

    # -----------------------------------------------------------------
    # Rule 3: Dynamic Cross-Validation (Re-Sync address_cleaned)
    # -----------------------------------------------------------------
    clean_parts = []
    for comp in [final_bldg, final_road, city, state, country]:
        comp_clean = comp.strip(", ").strip()
        if comp_clean and comp_clean not in clean_parts:
            clean_parts.append(comp_clean)
            
    address_cleaned = ", ".join(clean_parts)

    return {
        "building_landmark": final_bldg,
        "road_area": final_road,
        "city": city,
        "state": state,
        "country": country,
        "address_cleaned": address_cleaned
    }

# -----------------------------------------------------------------
# Main Processing Engine
# -----------------------------------------------------------------
for filename in PARTITIONS:
    file_path = os.path.join(CLEAN_DIR, filename)
    if not os.path.exists(file_path):
        continue

    print(f"Applying dynamic re-segregation on: {filename}...")
    df = pl.read_csv(file_path, separator="\t")

    updated_rows = [dynamic_rebalance_row(row) for row in df.iter_rows(named=True)]
    struct_series = pl.Series(name="updated", values=updated_rows)

    df = df.with_columns([
        struct_series.struct.field("building_landmark").alias("building_landmark"),
        struct_series.struct.field("road_area").alias("road_area"),
        struct_series.struct.field("city").alias("city"),
        struct_series.struct.field("state").alias("state"),
        struct_series.struct.field("country").alias("country"),
        struct_series.struct.field("address_cleaned").alias("address_cleaned"),
    ])

    # Recalculate word counts
    df = df.with_columns([
        pl.col("address_cleaned")
        .fill_null("").str.strip_chars().str.split(by=" ")
        .list.eval(pl.element().filter(pl.element() != "")).list.len()
        .cast(pl.Int16).alias("address_cleaned_word_count"),
    ])

    df.write_csv(file_path, separator="\t")
    print(f"Successfully processed {filename}")

print("\n--- Universal Dynamic Cleaning Completed! ---")
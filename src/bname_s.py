import os
import re
import polars as pl

formatted_dir = r"D:\claudesplan\data\dataset_formatted\train"

# 1. Domain Extensions (.com, .org, .net, etc.)
DOMAIN_REGEX = re.compile(
    r"\b\w+\.(?:com|org|net|io|co|in|ai|biz|info|edu|gov)\b|\.(?:com|org|net|io|co|in|ai|biz|info|edu|gov)\b", 
    re.IGNORECASE
)

# 2. Special Characters and Symbols (@, #, --, -, ., .., etc.)
SYMBOLS_REGEX = re.compile(r"[@#_\-/\\&\+\*!\?\|~\$%:;\.]+")

# 3. Fuzzy/Stretched Legal Suffixes & Types
FUZZY_LEGAL_TOKENS = [
    # Private / Limited variations
    r"\bp+r+a+i+v+a+t+e+\b", r"\bp+r+a+a+i+v+e+t+t+\b", r"\bp+v+t+\b", r"\bp+r+i+v+a+t+e+\b",
    r"\bl+i+m+i+t+e+d+\b", r"\bl+i+m+i+t+t+e+d+d+\b", r"\bl+t+d+\b", r"\bl+i+m+t+d+\b",
    r"\bp+r+a+a+i+v+e+t+t+d+d+\b", r"\bp+r+i+v+a+t+e+l+i+m+i+t+e+d+\b",
    
    # Incorporated / Corporation / Corp variations
    r"\bi+n+c+o+r+p+o+r+a+t+e+d+\b", r"\bi+n+c+o+r+p+\b", r"\bi+n+c+\b", r"\bc+o+r+p+o+r+a+t+i+o+n+\b",
    
    # Company / Co variations
    r"\bc+o+m+p+a+n+y+\b", r"\bc+m+p+n+y+\b", r"\bc+o+\b", r"\bc+o+m+p+\b",
    
    # LLC / LLP / PLC variations
    r"\bl+l+c+\b", r"\bl+i+m+i+t+e+d+l+i+a+b+i+l+i+t+y+c+o+m+p+a+n+y+\b",
    
    # Holdings / Solutions / Services / Enterprises
    r"\bh+o+l+d+i+n+g+s+\b", r"\bs+e+r+v+i+c+e+s+\b", r"\bs+o+l+u+t+i+o+n+s+\b", r"\be+n+t+e+r+p+r+i+s+e+s+\b"
]

# 4. Generic Facility / Venue Tokens (Centres, Centers, Facilities, Branches)
FACILITY_TOKENS = [
    r"\bc+e+n+t+r+e+s?\b",
    r"\bc+e+n+t+e+r+s?\b",
    r"\bf+a+c+i+l+i+t+y+\b",
    r"\bf+a+c+i+l+i+t+i+e+s+\b",
    r"\bb+r+a+n+c+h+\b",
    r"\bb+r+a+n+c+h+e+s+\b",
    r"\bb+u+i+l+d+i+n+g+\b"
]

# Combined Noise Regex Pattern
ALL_NOISE_TOKENS = FUZZY_LEGAL_TOKENS + FACILITY_TOKENS
NOISE_REGEX = re.compile("|".join(ALL_NOISE_TOKENS), re.IGNORECASE)

# Regex for stripping brackets (), [], {}
BRACKET_REGEX = re.compile(r"[\(\[\{].*?[\)\]\}]", re.IGNORECASE)

# Regex for consecutive duplicate words ("acme acme" -> "acme")
DUPLICATE_WORDS_REGEX = re.compile(r"\b(\w+)(?:\s+\1)+\b", re.IGNORECASE)


def clean_fuzzy_business_name(text: str) -> tuple[str, int]:
    """Cleans domains, symbols (@, #, --, .), legal noise, facility words, and duplicates."""
    if not text or text == "nan":
        return text if text else "nan", 0
    
    original = text
    cleaned = text
    
    # Step 1: Remove bracketed text
    cleaned = BRACKET_REGEX.sub("", cleaned)
    
    # Step 2: Remove domain suffixes (.com, .org, .co, etc.)
    cleaned = DOMAIN_REGEX.sub("", cleaned)
    
    # Step 3: Replace special characters (@, #, --, -, ., ..) with space
    cleaned = SYMBOLS_REGEX.sub(" ", cleaned)
    
    # Step 4: Strip legal suffixes and facility tokens (centres, centers, etc.)
    cleaned = NOISE_REGEX.sub("", cleaned)
    
    # Step 5: Remove duplicate consecutive words
    cleaned = DUPLICATE_WORDS_REGEX.sub(r"\1", cleaned)
    
    # Step 6: Normalize whitespace
    cleaned = " ".join(cleaned.split())
    
    is_modified = 1 if cleaned != original else 0
    return cleaned, is_modified


def batch_clean_names(series: pl.Series) -> pl.Series:
    """Batch cleans a Polars Series."""
    cleaned_list = []
    flags_list = []
    
    for text in series:
        cleaned, flag = clean_fuzzy_business_name(text)
        cleaned_list.append(cleaned)
        flags_list.append(flag)

    return pl.Series(
        name="cleaning_result",
        values=[
            {"cleaned_name": c, "is_modified": f} 
            for c, f in zip(cleaned_list, flags_list)
        ],
        dtype=pl.Struct([
            pl.Field("cleaned_name", pl.String),
            pl.Field("is_modified", pl.Int8)
        ])
    )


sources = {
    "source1": os.path.join(formatted_dir, "train_source1.tsv"),
    "source2": os.path.join(formatted_dir, "train_source2.tsv"),
    "source3": os.path.join(formatted_dir, "train_source3.tsv")
}

for name, path in sources.items():
    if not os.path.exists(path):
        print(f"Skipping missing partition: {path}")
        continue

    print(f"Applying domain & symbol cleaning to {name} in dataset_formatted...")
    
    df = pl.read_csv(path, separator="\t")
    
    # Apply cleaning logic
    df = df.with_columns(
        pl.col("business_name")
        .map_batches(
            batch_clean_names,
            return_dtype=pl.Struct([
                pl.Field("cleaned_name", pl.String),
                pl.Field("is_modified", pl.Int8)
            ])
        )
        .alias("res")
    )
    
    # Update business_name and modification flag
    df = df.with_columns([
        pl.col("res").struct.field("cleaned_name").alias("business_name"),
        pl.col("res").struct.field("is_modified").alias("business_name_is_modified"),
    ]).drop("res")
    
    # Recalculate word count
    df = df.with_columns(
        pl.col("business_name")
        .fill_null("")
        .str.strip_chars()
        .str.split(by=" ")
        .list.eval(pl.element().filter(pl.element() != ""))
        .list.len()
        .cast(pl.Int16)
        .alias("business_name_cleaned_word_count")
    )
    
    # Safe overwrite via temporary file (.tmp)
    temp_path = path + ".tmp"
    df.write_csv(temp_path, separator="\t")
    os.replace(temp_path, path)
    print(f"Successfully updated partition: {path} (Rows: {df.height:,})")

print("\n--- Cleaning Complete! Domains (.com), symbols (@, #, --, -), and centres removed. ---")
import os
import polars as pl
from unidecode import unidecode

# Original file sources
sources = {
    "source1": r"D:\claudesplan\data\dataset\train\train_source1.tsv",
    "source2": r"D:\claudesplan\data\dataset\train\train_source2.tsv",
    "source3": r"D:\claudesplan\data\dataset\train\train_source3.tsv"
}

# Output directory for cleaned datasets (never overwrites original data)
output_dir = r"D:\claudesplan\data\dataset_cleaned\train"
os.makedirs(output_dir, exist_ok=True)

cols = ["entity_id", "business_name", "business_address", "country"]
text_cols = ["business_name", "business_address", "country"]

def to_english_ascii(val: str | None) -> str:
    """Converts non-ASCII / non-English scripts (Hindi, Tamil, accents) to Latin ASCII."""
    if val is None or val == "":
        return ""
    return unidecode(str(val))

# Process each partition independently
for name, input_path in sources.items():
    print(f"Processing {name}...")
    
    # 1. Read original dataset as String types
    df = pl.read_csv(
        input_path,
        separator="\t",
        schema_overrides={c: pl.String for c in cols}
    )
    
    # 2. Add missing address flag BEFORE filling missing values
    # (1 if business_address is null, empty string, or whitespace; 0 otherwise)
    df = df.with_columns(
        pl.when(
            pl.col("business_address").is_null() | 
            (pl.col("business_address").str.strip_chars() == "")
        )
        .then(1)
        .otherwise(0)
        .alias("business_address_is_missing")
    )
    
    # 3. Transliterate text columns to English ASCII
    for col_name in text_cols:
        df = df.with_columns(
            pl.col(col_name)
            .map_elements(to_english_ascii, return_dtype=pl.String)
            .alias(col_name)
        )
    
    # 4. Convert TEXT columns to LOWERCASE & replace missing values with "NaN"
    # (entity_id is IGNORED to preserve original casing)
    for col_name in text_cols:
        df = df.with_columns(
            pl.when(
                pl.col(col_name).is_null() | 
                (pl.col(col_name).str.strip_chars() == "")
            )
            .then(pl.lit("NaN"))
            .otherwise(pl.col(col_name).str.to_lowercase().str.strip_chars())
            .alias(col_name)
        )
    
    # Handle entity_id missing values if any, but IGNORE lowercasing
    df = df.with_columns(
        pl.when(
            pl.col("entity_id").is_null() | 
            (pl.col("entity_id").str.strip_chars() == "")
        )
        .then(pl.lit("NaN"))
        .otherwise(pl.col("entity_id").str.strip_chars())
        .alias("entity_id")
    )
    
    # 5. Save output as TAB-SEPARATED (.tsv) preserving original partition structure
    output_path = os.path.join(output_dir, f"train_{name}.tsv")
    df.write_csv(output_path, separator="\t")
    print(f"Saved cleaned TSV file: {output_path} (Shape: {df.shape})")

print("\n--- Cleaning Complete! Original datasets remain unchanged. ---")
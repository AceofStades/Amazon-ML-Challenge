import os
import polars as pl

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
CLEAN_DIR = r"D:\claudesplan\data\dataset_clean\train"
PARTITIONS = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]

TEXT_FIELDS = ["building_landmark", "road_area", "city", "state"]

def sanitize_and_add_missing_flags(df: pl.DataFrame) -> pl.DataFrame:
    """
    1. Converts empty/whitespace/literal "null" strings into explicit pl.Null.
    2. Creates binary missingness indicator flags (1 = Missing, 0 = Present).
    """
    exprs = []
    
    for col in TEXT_FIELDS:
        if col in df.columns:
            # Step A: Clean string field to true null if empty/whitespace
            cleaned_col = (
                pl.when(
                    (pl.col(col).str.strip_chars() == "") | 
                    (pl.col(col).str.to_lowercase().is_in(["null", "nan", "none"]))
                )
                .then(None)
                .otherwise(pl.col(col).str.strip_chars())
            )
            exprs.append(cleaned_col.alias(col))
            
            # Step B: Add explicit missingness indicator flag for ML tree splits
            missing_flag = (
                pl.when(
                    (pl.col(col).is_null()) | 
                    (pl.col(col).str.strip_chars() == "") |
                    (pl.col(col).str.to_lowercase().is_in(["null", "nan", "none"]))
                )
                .then(1)
                .otherwise(0)
                .cast(pl.Int8)
                .alias(f"{col}_is_missing")
            )
            exprs.append(missing_flag)

    return df.with_columns(exprs)

# -------------------------------------------------------------------------
# Processing Execution
# -------------------------------------------------------------------------
for filename in PARTITIONS:
    file_path = os.path.join(CLEAN_DIR, filename)
    if not os.path.exists(file_path):
        continue

    print(f"Pre-loading missing-class safety schema for: {filename}...")
    df = pl.read_csv(file_path, separator="\t", infer_schema_length=10000)
    
    # Apply missingness transformation
    df_processed = sanitize_and_add_missing_flags(df)
    
    # Overwrite dataset partition with safe schema
    df_processed.write_csv(file_path, separator="\t")
    print(f"Saved: {filename} (Total columns: {len(df_processed.columns)})")

print("\n--- Missing-Class Safety Transformations Applied Successfully! ---")
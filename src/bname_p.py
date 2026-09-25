import os
import re
import polars as pl

# Directory Paths
expanded_dir = r"D:\claudesplan\data\dataset_expanded\train"
output_dir = r"D:\claudesplan\data\dataset_expanded\train"  # Overwrite in-place

os.makedirs(output_dir, exist_ok=True)

def normalize_text(text: str) -> tuple[str, int]:
    """Replaces '&' with 'and', removes commas and apostrophes, and collapses spaces."""
    if not text or text == "nan":
        return text if text else "nan", 0
    
    # Replace ampersands with ' and '
    cleaned = re.sub(r"&", " and ", text)
    # Remove commas and apostrophes
    cleaned = re.sub(r"[,']", "", cleaned)
    # Collapse multiple whitespaces
    cleaned = " ".join(cleaned.split())
    
    # Flag 1 if any symbol cleaning occurred, else 0
    is_modified = 1 if cleaned != text else 0
    return cleaned, is_modified

def batch_clean_symbols(series: pl.Series) -> pl.Series:
    """Vectorized batch symbol normalization on a Polars Series."""
    cleaned_list = []
    flags_list = []
    
    for text in series:
        cleaned, flag = normalize_text(text)
        cleaned_list.append(cleaned)
        flags_list.append(flag)

    return pl.Series(
        name="expansion_result",
        values=[
            {"expanded_name": c, "symbol_cleaned": f} 
            for c, f in zip(cleaned_list, flags_list)
        ],
        dtype=pl.Struct([
            pl.Field("expanded_name", pl.String),
            pl.Field("symbol_cleaned", pl.Int8)
        ])
    )

# Reading directly from dataset_expanded
sources = {
    "source1": os.path.join(expanded_dir, "train_source1.tsv"),
    "source2": os.path.join(expanded_dir, "train_source2.tsv"),
    "source3": os.path.join(expanded_dir, "train_source3.tsv")
}

for name, path in sources.items():
    if not os.path.exists(path):
        print(f"Skipping missing partition: {path}")
        continue

    print(f"Processing partition {name} from dataset_expanded...")
    
    # Read already expanded TSV
    df = pl.read_csv(path, separator="\t")
    
    # Apply symbol cleaning on business_name from dataset_expanded
    df = df.with_columns(
        pl.col("business_name")
        .map_batches(
            batch_clean_symbols,
            return_dtype=pl.Struct([
                pl.Field("expanded_name", pl.String),
                pl.Field("symbol_cleaned", pl.Int8)
            ])
        )
        .alias("res")
    )
    
    # Update business_name and set business_name_is_expanded to 1 
    # if either a short-form expansion OR a symbol replacement (&, ,, ') occurred
    df = df.with_columns([
        pl.col("res").struct.field("expanded_name").alias("business_name"),
        (
            (pl.col("business_name_is_expanded") == 1) | 
            (pl.col("res").struct.field("symbol_cleaned") == 1)
        ).cast(pl.Int8).alias("business_name_is_expanded")
    ]).drop("res")
    
    # Write updated TSV back to dataset_expanded
    output_path = os.path.join(output_dir, f"train_{name}.tsv")
    df.write_csv(output_path, separator="\t")
    print(f"Updated partition: {output_path} (Rows: {df.height:,}, Columns: {df.width})")

print("\n--- Symbol Clean Complete! Updated dataset_expanded with business_name_is_expanded = 1 ---")
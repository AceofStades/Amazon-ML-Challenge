import os
import polars as pl
from flashtext import KeywordProcessor

# Directory Paths
cleaned_dir = r"D:\claudesplan\data\dataset_cleaned\train"
output_dir = r"D:\claudesplan\data\dataset_expanded\train"
map_tsv = r"D:\claudesplan\data\dataset_cleaned\valid_sf_lf_map.tsv"

os.makedirs(output_dir, exist_ok=True)

# 1. Initialize FlashText Keyword Processor from external TSV file
mapping_df = pl.read_csv(map_tsv, separator="\t")
kp = KeywordProcessor(case_sensitive=False)

for row in mapping_df.iter_rows(named=True):
    kp.add_keyword(row["short_form"], row["long_form"])

def batch_expand_and_flag(series: pl.Series) -> pl.Series:
    """Vectorized batch expansion using FlashText on a Polars Series."""
    expanded_list = []
    flags_list = []
    
    for text in series:
        if not text or text == "nan":
            expanded_list.append(text if text else "nan")
            flags_list.append(0)
        else:
            exp = kp.replace_keywords(text)
            expanded_list.append(exp)
            flags_list.append(1 if exp != text else 0)

    # Return a Polars Struct Series matching expected fields
    return pl.Series(
        name="expansion_result",
        values=[
            {"expanded_name": e, "is_expanded": f} 
            for e, f in zip(expanded_list, flags_list)
        ],
        dtype=pl.Struct([
            pl.Field("expanded_name", pl.String),
            pl.Field("is_expanded", pl.Int8)
        ])
    )

# 2. Dataset Partition Mapping
sources = {
    "source1": os.path.join(cleaned_dir, "train_source1.tsv"),
    "source2": os.path.join(cleaned_dir, "train_source2.tsv"),
    "source3": os.path.join(cleaned_dir, "train_source3.tsv")
}

cols = ["entity_id", "business_name", "business_address", "country", "business_address_is_missing"]

# 3. Process each partition independently
for name, path in sources.items():
    if not os.path.exists(path):
        print(f"Skipping missing partition: {path}")
        continue

    print(f"Processing partition {name} from dataset_cleaned...")
    
    # Read TSV preserving original schema
    df = pl.read_csv(
        path,
        separator="\t",
        schema_overrides={c: pl.String for c in cols if c != "business_address_is_missing"}
    )
    
    # Apply batch expansion
    df = df.with_columns(
        pl.col("business_name")
        .map_batches(
            batch_expand_and_flag,
            return_dtype=pl.Struct([
                pl.Field("expanded_name", pl.String),
                pl.Field("is_expanded", pl.Int8)
            ])
        )
        .alias("expansion_result")
    )
    
    # Unnest struct into top-level business_name and business_name_is_expanded
    df = df.with_columns([
        pl.col("expansion_result").struct.field("expanded_name").alias("business_name"),
        pl.col("expansion_result").struct.field("is_expanded").alias("business_name_is_expanded")
    ]).drop("expansion_result")
    
    # Save output as tab-separated TSV with exact original partition name
    output_path = os.path.join(output_dir, f"train_{name}.tsv")
    df.write_csv(output_path, separator="\t")
    print(f"Saved expanded partition: {output_path} (Rows: {df.height:,}, Columns: {df.width})")

print("\n--- Expansion & Flagging Complete! Output saved to dataset_expanded/train ---")
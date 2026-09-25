import os
import re
import polars as pl

# Directory Paths
input_dir = r"D:\claudesplan\data\dataset_expanded\train"
output_dir = r"D:\claudesplan\data\dataset_formatted\train"

os.makedirs(output_dir, exist_ok=True)

# Full expanded legal entity terms specifically targeted at trailing positions
# (Ensures leading/middle terms like "company" in "company store" are NOT touched)
EXPANDED_SUFFIX_TOKENS = [
    r"private limited", r"public limited company", r"limited liability company",
    r"limited liability partnership", r"general partnership", r"sole proprietorship",
    r"joint stock company", r"corporation", r"incorporated", r"incorporation",
    r"limited", r"private", r"company", r"corporation", r"holdings", r"enterprises",
    r"services", r"solutions", r"group", r"gesellschaft mit beschraenkter haftung",
    r"aktiengesellschaft", r"societe anonyme", r"besloten vennootschap"
]

# Regex anchored to match trailing suffixes at the end of string ($)
SUFFIX_PATTERN = r"(?:\s+\b(?:" + "|".join(EXPANDED_SUFFIX_TOKENS) + r")\b)+$"
SUFFIX_REGEX = re.compile(SUFFIX_PATTERN, re.IGNORECASE)

def strip_trailing_legal_suffixes(text: str) -> tuple[str, int]:
    """Trims expanded legal entity suffixes from the right end of business_name only."""
    if not text or text == "nan":
        return text if text else "nan", 0
    
    cleaned = text
    # Iteratively trim trailing legal suffixes from the right end
    while True:
        new_text = SUFFIX_REGEX.sub("", cleaned).strip()
        if new_text == cleaned:
            break
        cleaned = new_text

    # Collapse any residual double spaces
    cleaned = " ".join(cleaned.split())
    
    is_modified = 1 if cleaned != text else 0
    return cleaned, is_modified

def batch_clean_suffixes(series: pl.Series) -> pl.Series:
    """Vectorized batch trailing suffix stripping on a Polars Series."""
    cleaned_list = []
    flags_list = []
    
    for text in series:
        cleaned, flag = strip_trailing_legal_suffixes(text)
        cleaned_list.append(cleaned)
        flags_list.append(flag)

    return pl.Series(
        name="cleaning_result",
        values=[
            {"cleaned_name": c, "is_suffix_removed": f} 
            for c, f in zip(cleaned_list, flags_list)
        ],
        dtype=pl.Struct([
            pl.Field("cleaned_name", pl.String),
            pl.Field("is_suffix_removed", pl.Int8)
        ])
    )

sources = {
    "source1": os.path.join(input_dir, "train_source1.tsv"),
    "source2": os.path.join(input_dir, "train_source2.tsv"),
    "source3": os.path.join(input_dir, "train_source3.tsv")
}

for name, input_path in sources.items():
    if not os.path.exists(input_path):
        print(f"Skipping missing partition: {input_path}")
        continue

    print(f"Processing partition {name} from dataset_expanded -> dataset_suffix_cleaned...")
    
    df = pl.read_csv(input_path, separator="\t")
    
    # Clean trailing suffixes from business_name
    df = df.with_columns(
        pl.col("business_name")
        .map_batches(
            batch_clean_suffixes,
            return_dtype=pl.Struct([
                pl.Field("cleaned_name", pl.String),
                pl.Field("is_suffix_removed", pl.Int8)
            ])
        )
        .alias("res")
    )
    
    # Apply cleaned name, tracking flag, and calculate word count
    df = df.with_columns([
        pl.col("res").struct.field("cleaned_name").alias("business_name"),
        pl.col("res").struct.field("is_suffix_removed").alias("business_name_is_legal_suffix_removed"),
    ]).drop("res")
    
    # Calculate word count after suffix removal
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
    
    # Write directly to the NEW dataset folder
    output_path = os.path.join(output_dir, f"train_{name}.tsv")
    df.write_csv(output_path, separator="\t")
    print(f"Saved new dataset partition: {output_path} (Rows: {df.height:,}, Columns: {df.width})")

print("\n--- Process Complete! New dataset created at data/dataset_formatted/train ---")
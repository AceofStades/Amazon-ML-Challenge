import os
import re
import polars as pl

INPUT_EXPANDED_DIR = r"D:\claudesplan\data\dataset_expanded\train"
OUTPUT_FORMATTED_DIR = r"D:\claudesplan\data\dataset_formatted\train"

os.makedirs(OUTPUT_FORMATTED_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Regex Definitions
# -------------------------------------------------------------------------

# Pass 1: Remove all punctuation, dots, brackets, and symbols
PUNCTUATION_BRACKETS_REGEX = re.compile(r"[^\w\s]")

# Pass 2: Comprehensive list including liability, partnership, and corporate entity words
LEGAL_AND_FACILITY_TOKENS = [
    # Private / Limited variations
    r"\bprivate\b", r"\bpvt\b", r"\blimited\b", r"\bltd\b", r"\bpvtltd\b", r"\bprivatelimited\b",
    
    # Liability & Partnership variations
    r"\bliability\b", r"\bpartnership\b", r"\bpartnerships\b", r"\bpartners\b", r"\bpartner\b",
    r"\bllp\b", r"\blp\b", r"\bproprietorship\b", r"\bproprietor\b",
    
    # Incorporated / Corp / Co
    r"\bincorporated\b", r"\binc\b", r"\bcorporation\b", r"\bcorp\b",
    r"\bcompany\b", r"\bco\b", r"\bcmpny\b",
    
    # LLC / PLC / Holdings / Services / Entities
    r"\bllc\b", r"\bplc\b", r"\bholdings\b", r"\bservices\b", 
    r"\bsolutions\b", r"\benterprises\b", r"\bassociation\b", r"\bsociety\b", 
    r"\bfoundation\b", r"\btrust\b", r"\bcooperative\b", r"\bcoop\b",
    
    # Centres / Centers / Facilities
    r"\bcentres\b", r"\bcentre\b", r"\bcenters\b", r"\bcenter\b", 
    r"\bfacility\b", r"\bfacilities\b", r"\bbranch\b", r"\bbranches\b", r"\bbuilding\b"
]

LEGAL_REGEX = re.compile(r"\b(?:" + "|".join(LEGAL_AND_FACILITY_TOKENS) + r")\b", re.IGNORECASE)

# Collapse repeated characters (3+ consecutive characters -> 1 character)
REPEATED_CHARS_REGEX = re.compile(r"(.)\1{2,}")

# Remove consecutive duplicate words
DUPLICATE_WORDS_REGEX = re.compile(r"\b(\w+)(?:\s+\1)+\b", re.IGNORECASE)


def process_business_name(text: str) -> tuple[str, int]:
    """Two-pass cleaning pipeline with guaranteed non-empty output."""
    if not text or text.strip() == "" or str(text).lower() == "nan":
        return "nan", 0

    original = str(text).strip()
    
    # STEP 1: Remove ALL punctuation, dots, brackets, and symbols
    pass1_text = PUNCTUATION_BRACKETS_REGEX.sub(" ", original)
    pass1_text = " ".join(pass1_text.split())
    
    if not pass1_text:
        return original, 1

    # STEP 2: Collapse stretched/repeated letters
    pass2_text = REPEATED_CHARS_REGEX.sub(r"\1", pass1_text)

    # STEP 3: Remove legal suffixes, liability/partnership terms, and facility words
    pass2_text = LEGAL_REGEX.sub("", pass2_text)

    # STEP 4: Remove duplicate consecutive words
    pass2_text = DUPLICATE_WORDS_REGEX.sub(r"\1", pass2_text)

    # STEP 5: Normalize spaces
    cleaned = " ".join(pass2_text.split())

    # GUARDRAIL: Prevent empty strings by falling back to Pass 1 output
    if not cleaned:
        cleaned = pass1_text

    is_modified = 1 if cleaned != original else 0
    return cleaned, is_modified


def batch_clean_names(series: pl.Series) -> pl.Series:
    cleaned_list = []
    flags_list = []

    for text in series:
        cleaned, flag = process_business_name(text)
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


# -------------------------------------------------------------------------
# Rebuild Partitions
# -------------------------------------------------------------------------

partitions = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]

for filename in partitions:
    input_file = os.path.join(INPUT_EXPANDED_DIR, filename)
    output_file = os.path.join(OUTPUT_FORMATTED_DIR, filename)

    if not os.path.exists(input_file):
        print(f"Skipping missing file: {input_file}")
        continue

    print(f"Rebuilding {filename} from dataset_expanded...")
    df = pl.read_csv(input_file, separator="\t")

    # Apply cleaning
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

    # Update columns
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

    # Write output
    df.write_csv(output_file, separator="\t")
    
    empty_count = df.filter(pl.col("business_name") == "").height
    null_count = df.filter(pl.col("business_name").is_null()).height
    print(f"Done: {output_file} | Total Rows: {df.height:,} | Empty: {empty_count} | Nulls: {null_count}\n")

print("--- Pipeline Complete! Rebuilt dataset_formatted including liability & partnership removals. ---")
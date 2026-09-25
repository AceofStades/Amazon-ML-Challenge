import polars as pl
import re

# File paths
sources = {
    "source1": r"D:\claudesplan\data\dataset_cleaned\train\train_source1.tsv",
    "source2": r"D:\claudesplan\data\dataset_cleaned\train\train_source2.tsv",
    "source3": r"D:\claudesplan\data\dataset_cleaned\train\train_source3.tsv"
}

cols = ["entity_id", "business_name", "business_address", "country"]

# 1. Load all datasets into one combined Polars DataFrame
df_list = []
for name, path in sources.items():
    df = pl.read_csv(
        path, 
        separator="\t", 
        schema_overrides={c: pl.String for c in cols}
    ).with_columns(pl.lit(name).alias("source_file"))
    df_list.append(df)

combined_df = pl.concat(df_list)

# Fill nulls with empty string
combined_df = combined_df.with_columns(
    pl.col("business_name").fill_null("")
)

# ---------------------------------------------------------
# STEP 1: Word Count Analysis
# ---------------------------------------------------------
# Clean string temporarily to count real words
combined_df = combined_df.with_columns(
    pl.col("business_name")
    .str.to_lowercase()
    .str.replace_all(r"[^\w\s]", " ")
    .str.strip_chars()
    .alias("clean_name")
)

# Calculate word count per business name
combined_df = combined_df.with_columns(
    pl.col("clean_name")
    .str.split(" ")
    .list.eval(pl.element().filter(pl.element() != ""))
    .list.len()
    .alias("word_count")
)

print("--- 1. Business Name Word Count Summary ---")
print(combined_df.select([
    pl.mean("word_count").alias("mean_words"),
    pl.median("word_count").alias("median_words"),
    pl.min("word_count").alias("min_words"),
    pl.max("word_count").alias("max_words"),
]))

print("\n--- Word Count Distribution ---")
print(
    combined_df["word_count"]
    .value_counts(sort=True)
    .head(10)
)

# ---------------------------------------------------------
# STEP 2: Common Terms & Legal Suffix Frequency
# ---------------------------------------------------------
# Unnest all words across the dataset to find the top common terms
words_df = (
    combined_df
    .select(pl.col("clean_name").str.split(" "))
    .explode("clean_name")
    .filter(pl.col("clean_name") != "")
    .rename({"clean_name": "word"})
)

print("\n--- 2. Top 30 Most Frequent Words in Business Names ---")
print(words_df["word"].value_counts(sort=True).head(30))

# ---------------------------------------------------------
# STEP 3: Language & Character Set Check (English / ASCII check)
# ---------------------------------------------------------
# Check how many business names contain non-ASCII characters
combined_df = combined_df.with_columns(
    pl.col("business_name")
    .str.contains(r"[^\x00-\x7F]")
    .alias("has_non_ascii")
)

non_ascii_count = combined_df["has_non_ascii"].sum()
total_records = combined_df.height
non_ascii_pct = round((non_ascii_count / total_records) * 100, 2)

print("\n--- 3. Language / Character Set Check ---")
print(f"Total Records: {total_records}")
print(f"Records with Non-ASCII / Accented / Special Characters: {non_ascii_count} ({non_ascii_pct}%)")

if non_ascii_count > 0:
    print("\nSample Non-ASCII Business Names:")
    print(combined_df.filter(pl.col("has_non_ascii")).select(["business_name", "country"]).head(10))
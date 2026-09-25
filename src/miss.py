import polars as pl

# sources = {
#     "source1": r"D:\claudesplan\data\dataset\train\train_source1.tsv",
#     "source2": r"D:\claudesplan\data\dataset\train\train_source2.tsv",
#     "source3": r"D:\claudesplan\data\dataset\train\train_source3.tsv"
# }

sources = {
    "source1": r"D:\claudesplan\data\dataset\test\test_source1.tsv",
    "source2": r"D:\claudesplan\data\dataset\test\test_source2.tsv",
    "source3": r"D:\claudesplan\data\dataset\test\test_source3.tsv"
}


cols = ["entity_id", "business_name", "business_address", "country"]

# 1. Load dataframes forcing all columns to String type
dfs = {}
for name, path in sources.items():
    df = pl.read_csv(
        path,
        separator="\t",
        schema_overrides={c: pl.String for c in cols}
    )
    
    # Standardize empty strings or whitespace to null
    df = df.with_columns([
        pl.when(pl.col(c).str.strip_chars() == "")
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in cols
    ])
    dfs[name] = df

# 2. Summary table: Null counts per column
summary = pl.DataFrame({
    "source": list(dfs.keys()),
    **{c: [df[c].null_count() for df in dfs.values()] for c in cols}
})

print("--- Missing Values Count ---")
print(summary)

# 3. Summary table: Null percentage (%)
pct_summary = pl.DataFrame({
    "source": list(dfs.keys()),
    **{c: [round((df[c].null_count() / df.height) * 100, 2) for df in dfs.values()] for c in cols}
})

print("\n--- Missing Values Percentage (%) ---")
print(pct_summary)
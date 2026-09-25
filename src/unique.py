import polars as pl

sources = {
    "source1": r"D:\claudesplan\data\dataset\train\train_source1.tsv",
    "source2": r"D:\claudesplan\data\dataset\train\train_source2.tsv",
    "source3": r"D:\claudesplan\data\dataset\train\train_source3.tsv"
}

cols = ["entity_id", "business_name", "business_address", "country"]

# 1. Read files forcing string types and converting blank strings to null
dfs = {}
for name, path in sources.items():
    df = pl.read_csv(
        path,
        separator="\t",
        schema_overrides={c: pl.String for c in cols}
    )
    # Standardize empty strings or whitespace to null
    dfs[name] = df.with_columns([
        pl.when(pl.col(c).str.strip_chars() == "")
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in cols
    ])

# 2. Count unique values per column (excluding nulls)
unique_summary = pl.DataFrame({
    "source": list(dfs.keys()),
    **{c: [df[c].drop_nulls().n_unique() for df in dfs.values()] for c in cols}
})

print("--- Unique Values Count Per Column (Excluding Nulls) ---")
print(unique_summary)
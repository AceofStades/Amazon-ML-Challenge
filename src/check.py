import os
import polars as pl

# Configure Polars display limits for clean console output
pl.Config.set_tbl_rows(-1)          # Display ALL rows
pl.Config.set_tbl_cols(-1)          # Display ALL columns
pl.Config.set_tbl_width_chars(120)  # Expand width to avoid line wrapping

CLEAN_DIR = r"D:\claudesplan\data\dataset_clean\train"
PARTITIONS = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]

def analyze_dataset_read_only(file_path: str, filename: str):
    if not os.path.exists(file_path):
        print(f"File not found: {filename}")
        return

    # Read dataset into RAM only
    df = pl.read_csv(file_path, separator="\t", infer_schema_length=10000)
    n_rows, n_cols = df.shape

    print("\n" + "=" * 80)
    print(f" READ-ONLY DATASET SUMMARY (NULL AS CLASS): {filename}")
    print("=" * 80)
    print(f"Shape: {n_rows:,} rows × {n_cols} columns\n")

    summary_data = []

    for col_name in df.columns:
        col = df[col_name]
        
        # Count native nulls as distinct class occurrences
        null_count = col.null_count()

        # Check empty/placeholder strings strictly for String columns
        if col.dtype == pl.String:
            empty_count = (
                (col == "") | 
                (col.str.to_lowercase() == "null") | 
                (col.str.to_lowercase() == "nan")
            ).sum()
            total_null_class = null_count + (empty_count if empty_count is not None else 0)
        else:
            total_null_class = null_count

        null_pct = (total_null_class / n_rows) * 100
        unique_count = col.n_unique()

        summary_data.append({
            "Column Name": col_name,
            "Data Type": str(col.dtype),
            "Null Class Count": total_null_class,
            "Null Class (%)": f"{null_pct:.2f}%",
            "Unique Values": unique_count
        })

    summary_df = pl.DataFrame(summary_data)
    print(summary_df)

if __name__ == "__main__":
    for filename in PARTITIONS:
        full_path = os.path.join(CLEAN_DIR, filename)
        analyze_dataset_read_only(full_path, filename)
        
    print("\n--- Check complete. ZERO files were modified or written to disk. ---")
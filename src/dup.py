import os
import polars as pl

# Configure Polars display formatting
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(120)

CLEAN_DIR = r"D:\claudesplan\data\dataset_clean\train"
PARTITIONS = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]

def check_id_duplication(clean_dir: str, partitions: list[str]):
    source_dfs = {}
    id_counts_summary = []

    print("=" * 80)
    print(" 1. INTRA-SOURCE ID DUPLICATION CHECK (Within Same File)")
    print("=" * 80)

    for filename in partitions:
        file_path = os.path.join(clean_dir, filename)
        if not os.path.exists(file_path):
            print(f"File not found: {filename}")
            continue

        # Load ID column only for fast execution
        df = pl.read_csv(file_path, separator="\t", columns=["id"])
        source_dfs[filename] = df

        total_rows = len(df)
        unique_ids = df["id"].n_unique()
        duplicate_rows = total_rows - unique_ids
        duplicate_pct = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0.0

        id_counts_summary.append({
            "File Name": filename,
            "Total Rows": f"{total_rows:,}",
            "Unique IDs": f"{unique_ids:,}",
            "Duplicate Rows": f"{duplicate_rows:,}",
            "Duplicate (%)": f"{duplicate_pct:.4f}%",
            "Has Duplicates": "YES" if duplicate_rows > 0 else "NO"
        })

    summary_df = pl.DataFrame(id_counts_summary)
    print(summary_df)

    print("\n" + "=" * 80)
    print(" 2. INTER-SOURCE ID OVERLAP CHECK (Across Different Sources)")
    print("=" * 80)

    loaded_files = list(source_dfs.keys())
    if len(loaded_files) < 2:
        print("Need at least 2 loaded files to check inter-source overlap.")
        return

    # Check pairwise ID overlap between sources
    overlap_summary = []
    for i in range(len(loaded_files)):
        for j in range(i + 1, len(loaded_files)):
            file_a = loaded_files[i]
            file_b = loaded_files[j]

            ids_a = source_dfs[file_a]["id"].unique()
            ids_b = source_dfs[file_b]["id"].unique()

            # Find intersection of IDs
            shared_ids_count = ids_a.filter(ids_a.is_in(ids_b)).len()

            overlap_summary.append({
                "Source A": file_a,
                "Source B": file_b,
                "Shared Overlapping IDs": f"{shared_ids_count:,}",
                "Overlap Status": "OVERLAPPING" if shared_ids_count > 0 else "DISJOINT"
            })

    overlap_df = pl.DataFrame(overlap_summary)
    print(overlap_df)

if __name__ == "__main__":
    check_id_duplication(CLEAN_DIR, PARTITIONS)
    print("\n--- Check complete. ZERO files were modified or written to disk. ---")
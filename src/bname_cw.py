import os
import polars as pl

# Directory Paths
expanded_dir = r"D:\claudesplan\data\dataset_expanded\train"
output_dir = r"D:\claudesplan\data\dataset_expanded\train"

os.makedirs(output_dir, exist_ok=True)

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
    
    # Read expanded TSV
    df = pl.read_csv(path, separator="\t")
    
    # Calculate word count for business_name
    df = df.with_columns(
        pl.col("business_name")
        .fill_null("")
        .str.strip_chars()
        .str.split(by=" ")
        .list.eval(pl.element().filter(pl.element() != ""))
        .list.len()
        .cast(pl.Int16)
        .alias("business_name_word_count")
    )
    
    # Write to temporary path to prevent Windows mmap lock error (os error 1224)
    output_path = os.path.join(output_dir, f"train_{name}.tsv")
    temp_path = output_path + ".tmp"
    
    df.write_csv(temp_path, separator="\t")
    
    # Safely replace original file
    os.replace(temp_path, output_path)
    print(f"Saved partition with word count: {output_path} (Rows: {df.height:,}, Columns: {df.width})")

print("\n--- Word Count Feature Addition Complete! ---")
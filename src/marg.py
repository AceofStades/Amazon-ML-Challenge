import os
import polars as pl

CLEAN_DIR = r"D:\claudesplan\data\dataset_clean\train"
PARTITIONS = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]

ROAD_KEYWORDS = r"(?i)\b(marg|road|rd|path|lane|gali|mrg)\b"

def reassign_road_tokens(df: pl.DataFrame) -> pl.DataFrame:
    landmark_has_road = pl.col("building_landmark").str.contains(ROAD_KEYWORDS)
    
    df_corrected = df.with_columns([
        # 1. Update road_area: Append or assign the road token from landmark if present
        pl.when(landmark_has_road)
        .then(
            pl.when(pl.col("road_area").is_null())
            .then(pl.col("building_landmark"))
            .otherwise(pl.concat_str([pl.col("building_landmark"), pl.col("road_area")], separator=" "))
        )
        .otherwise(pl.col("road_area"))
        .alias("road_area"),

        # 2. Nullify building_landmark if it was purely a road string
        pl.when(landmark_has_road)
        .then(None)
        .otherwise(pl.col("building_landmark"))
        .alias("building_landmark")
    ])

    # Recalculate missing indicators after re-alignment
    return df_corrected.with_columns([
        pl.col("building_landmark").is_null().cast(pl.Int64).alias("building_landmark_is_missing"),
        pl.col("road_area").is_null().cast(pl.Int64).alias("road_area_is_missing")
    ])

for filename in PARTITIONS:
    file_path = os.path.join(CLEAN_DIR, filename)
    temp_path = os.path.join(CLEAN_DIR, f"temp_{filename}")
    
    if os.path.exists(file_path):
        print(f"Fixing 'Marg' token misalignments in {filename}...")
        
        # Read from original file
        df = pl.read_csv(file_path, separator="\t", infer_schema_length=10000)
        df_fixed = reassign_road_tokens(df)
        
        # Write to temporary file to avoid Windows memory-map lock conflicts
        df_fixed.write_csv(temp_path, separator="\t")
        
        # Explicitly release dataframe reference before replacing file
        del df, df_fixed
        
        # Atomic replace on Windows
        os.replace(temp_path, file_path)
        print(f"Updated {filename} successfully.")

print("\n--- Token Realignment Completed! 'Marg' pushed to road_area ---")
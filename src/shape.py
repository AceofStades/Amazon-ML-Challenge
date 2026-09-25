import polars as pl

print("Train")

# Read the TSV file with tab separator
df = pl.read_csv(r"D:\claudesplan\data\dataset_clean\train\train_source1.tsv", separator="\t")

# Get the shape as a tuple: (num_rows, num_columns)
shape = df.shape
print(shape)

# Or print rows and columns individually
print("S1")
print(f"Rows: {df.height}, Columns: {df.width}")



# Read the TSV file with tab separator
df = pl.read_csv(r"D:\claudesplan\data\dataset_clean\train\train_source2.tsv", separator="\t")

# Get the shape as a tuple: (num_rows, num_columns)
shape = df.shape
print(shape)

# Or print rows and columns individually
print("S2")
print(f"Rows: {df.height}, Columns: {df.width}")


# Read the TSV file with tab separator
df = pl.read_csv(r"D:\claudesplan\data\dataset_clean\train\train_source3.tsv", separator="\t")

# Get the shape as a tuple: (num_rows, num_columns)
shape = df.shape
print(shape)

# Or print rows and columns individually
print("S3")
print(f"Rows: {df.height}, Columns: {df.width}")


print("ground truth")

# Read the TSV file with tab separator
df = pl.read_csv(r"D:\claudesplan\data\dataset\train\train_ground_truth.tsv", separator="\t")

# Get the shape as a tuple: (num_rows, num_columns)
shape = df.shape
print(shape)

# Or print rows and columns individually
print("GT")
print(f"Rows: {df.height}, Columns: {df.width}")



print("Test")

# Read the TSV file with tab separator
df = pl.read_csv(r"D:\claudesplan\data\dataset\test\test_source1.tsv", separator="\t")

# Get the shape as a tuple: (num_rows, num_columns)
shape = df.shape
print(shape)

# Or print rows and columns individually
print("S1")
print(f"Rows: {df.height}, Columns: {df.width}")

# Read the TSV file with tab separator
df = pl.read_csv(r"D:\claudesplan\data\dataset\test\test_source2.tsv", separator="\t")

# Get the shape as a tuple: (num_rows, num_columns)
shape = df.shape
print(shape)

# Or print rows and columns individually
print("S2")
print(f"Rows: {df.height}, Columns: {df.width}")

# Read the TSV file with tab separator
df = pl.read_csv(r"D:\claudesplan\data\dataset\test\test_source3.tsv", separator="\t")

# Get the shape as a tuple: (num_rows, num_columns)
shape = df.shape
print(shape)

# Or print rows and columns individually
print("S3")
print(f"Rows: {df.height}, Columns: {df.width}")

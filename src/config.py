# config.py
import os

# --------------------------------------------------
# BASE DIRECTORIES
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "testdata")


def resolve_dataset_file(directory, filename_prefix):
    """Return the first matching TSV file in a directory.

    The real workspace filenames include suffixes like "(2)" and "(1)",
    so a strict filename match is too fragile.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    matches = [
        os.path.join(directory, file_name)
        for file_name in os.listdir(directory)
        if file_name.lower().startswith(filename_prefix.lower())
        and file_name.lower().endswith(".tsv")
    ]

    if not matches:
        raise FileNotFoundError(
            f"Could not find a TSV file starting with '{filename_prefix}' in {directory}"
        )

    return sorted(matches)[0]


# --------------------------------------------------
# TRAINING FILES
# --------------------------------------------------
TRAIN_SOURCE1 = resolve_dataset_file(TRAIN_DIR, "train_source1")
TRAIN_SOURCE2 = resolve_dataset_file(TRAIN_DIR, "train_source2")
TRAIN_SOURCE3 = resolve_dataset_file(TRAIN_DIR, "train_source3")
GROUND_TRUTH = resolve_dataset_file(TRAIN_DIR, "train_ground_truth")

# --------------------------------------------------
# TEST FILES
# --------------------------------------------------
TEST_SOURCE1 = resolve_dataset_file(TEST_DIR, "test_source1")
TEST_SOURCE2 = resolve_dataset_file(TEST_DIR, "test_source2")
TEST_SOURCE3 = resolve_dataset_file(TEST_DIR, "test_source3")

# --------------------------------------------------
# OUTPUT DIRECTORY
# --------------------------------------------------
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# OUTPUT FILES
# --------------------------------------------------
MATCHING_RESULTS = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
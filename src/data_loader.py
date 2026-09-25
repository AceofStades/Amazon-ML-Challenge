# data_loader.py
import pandas as pd
from config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    GROUND_TRUTH,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3
)
# --------------------------------------------------
# TRAINING DATA
# --------------------------------------------------
def load_training_data(max_rows=None):
    source1 = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        nrows=max_rows
    )
    source2 = pd.read_csv(
        TRAIN_SOURCE2,
        sep="\t",
        nrows=max_rows
    )
    source3 = pd.read_csv(
        TRAIN_SOURCE3,
        sep="\t",
        nrows=max_rows
    )
    ground_truth = pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        nrows=max_rows
    )
    return source1, source2, source3, ground_truth
# --------------------------------------------------
# TEST DATA
# --------------------------------------------------
def load_test_data(max_rows=None):
    source1 = pd.read_csv(
        TEST_SOURCE1,
        sep="\t",
        nrows=max_rows
    )
    source2 = pd.read_csv(
        TEST_SOURCE2,
        sep="\t",
        nrows=max_rows
    )
    source3 = pd.read_csv(
        TEST_SOURCE3,
        sep="\t",
        nrows=max_rows
    )
    return source1, source2, source3
# --------------------------------------------------
# DISPLAY BASIC INFORMATION
# --------------------------------------------------
def display_data_information(
    source1,
    source2,
    source3
):
    print("\n========== SOURCE 1 ==========")
    print("Rows:", len(source1))
    print("Columns:", list(source1.columns))
    print("\n========== SOURCE 2 ==========")
    print("Rows:", len(source2))
    print("Columns:", list(source2.columns))
    print("\n========== SOURCE 3 ==========")
    print("Rows:", len(source3))
    print("Columns:", list(source3.columns))
# --------------------------------------------------
# TESTING
# --------------------------------------------------
if __name__ == "__main__":
    source1, source2, source3, ground_truth = load_training_data()
    display_data_information(
        source1,
        source2,
        source3
    )
    print("\n========== GROUND TRUTH ==========")
    print("Rows:", len(ground_truth))
    print("Columns:", list(ground_truth.columns))
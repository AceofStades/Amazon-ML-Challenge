from data_loader import load_training_data
from preprocessing import preprocess_sources
from blocking import generate_all_candidates


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("\nLoading training data...")

source1, source2, source3, ground_truth = load_training_data()

print("Training data loaded.")


# ---------------------------------------------------------
# Preprocess data
# ---------------------------------------------------------

print("\nPreprocessing data...")

source1, source2, source3 = preprocess_sources(
    source1,
    source2,
    source3
)

print("Preprocessing completed.")


# ---------------------------------------------------------
# Generate candidates
# ---------------------------------------------------------

print("\nGenerating candidates...")

candidate_df = generate_all_candidates(
    source1,
    source2,
    source3
)

print("\nCandidate generation completed.")


# ---------------------------------------------------------
# Candidate counts
# ---------------------------------------------------------

candidate_df["candidate_count"] = (
    candidate_df["candidate_entity_ids"]
    .apply(len)
)


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

print("\n========================================")
print("       BLOCKING RESULTS")
print("========================================")

print(
    "\nSource 1 entities:",
    len(candidate_df)
)

print(
    "Source 2 entities:",
    len(source2)
)

print(
    "Source 3 entities:",
    len(source3)
)

print(
    "Total candidates:",
    candidate_df["candidate_count"].sum()
)

print(
    "Average candidates per S1:",
    round(
        candidate_df["candidate_count"].mean(),
        2
    )
)

print(
    "Median candidates per S1:",
    candidate_df["candidate_count"].median()
)

print(
    "Minimum candidates for one S1:",
    candidate_df["candidate_count"].min()
)

print(
    "Maximum candidates for one S1:",
    candidate_df["candidate_count"].max()
)

print(
    "S1 entities with zero candidates:",
    (
        candidate_df["candidate_count"] == 0
    ).sum()
)

print(
    "S1 entities with candidates:",
    (
        candidate_df["candidate_count"] > 0
    ).sum()
)


# ---------------------------------------------------------
# Candidate distribution
# ---------------------------------------------------------

print("\n========================================")
print("       CANDIDATE DISTRIBUTION")
print("========================================")

print(
    "\nCandidates <= 5:",
    (
        candidate_df["candidate_count"] <= 5
    ).sum()
)

print(
    "Candidates 6-10:",
    (
        candidate_df["candidate_count"].between(6, 10)
    ).sum()
)

print(
    "Candidates 11-20:",
    (
        candidate_df["candidate_count"].between(11, 20)
    ).sum()
)

print(
    "Candidates 21-50:",
    (
        candidate_df["candidate_count"].between(21, 50)
    ).sum()
)

print(
    "Candidates > 50:",
    (
        candidate_df["candidate_count"] > 50
    ).sum()
)


# ---------------------------------------------------------
# Sample results
# ---------------------------------------------------------

print("\n========================================")
print("          SAMPLE RESULTS")
print("========================================")

sample = candidate_df.head(10).copy()

for _, row in sample.iterrows():

    print(
        "\nS1:",
        row["source1_entity_id"]
    )

    print(
        "Candidate count:",
        row["candidate_count"]
    )

    print(
        "Candidates:",
        row["candidate_entity_ids"]
    )


# ---------------------------------------------------------
# Save diagnostic output
# ---------------------------------------------------------

candidate_df.to_csv(
    "blocking_test_results.tsv",
    sep="\t",
    index=False
)

print(
    "\nDiagnostic results saved to "
    "blocking_test_results.tsv"
)
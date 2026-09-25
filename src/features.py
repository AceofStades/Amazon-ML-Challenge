import math
import random

import pandas as pd
from rapidfuzz.fuzz import ratio


# ---------------------------------------------------------
# MCMC missing-value model
# ---------------------------------------------------------

def _is_missing(value):
    return pd.isna(value) or str(value).strip() == ""


def _mcmc_posterior_mean(
    successes,
    failures,
    iterations=1500,
    burn_in=300,
    random_state=42
):
    """Estimate a Beta posterior mean with a Metropolis-Hastings chain."""
    rng = random.Random(random_state)
    current = (successes + 1.0) / (successes + failures + 2.0)
    samples = []

    def log_posterior(probability):
        if probability <= 0.0 or probability >= 1.0:
            return float("-inf")
        return (
            successes * math.log(probability)
            + failures * math.log1p(-probability)
        )

    for step in range(iterations):
        proposal = current + rng.gauss(0.0, 0.08)
        if 0.0 < proposal < 1.0:
            log_ratio = (
                log_posterior(proposal)
                - log_posterior(current)
            )
            if math.log(rng.random()) < min(0.0, log_ratio):
                current = proposal

        if step >= burn_in:
            samples.append(current)

    return sum(samples) / len(samples)


def _missing_match_probability(values):
    observed = [
        str(value).strip()
        for value in values
        if not _is_missing(value)
    ]
    if len(observed) < 2:
        return 0.5

    counts = pd.Series(observed).value_counts()
    pair_count = len(observed) * (len(observed) - 1)
    matching_pairs = sum(count * (count - 1) for count in counts)

    agreement_rate = matching_pairs / pair_count
    evidence = min(len(observed), 20)
    successes = agreement_rate * evidence
    failures = (1.0 - agreement_rate) * evidence
    return _mcmc_posterior_mean(successes, failures)


def _build_missing_value_profile(*dataframes):
    return {
        field: _missing_match_probability(
            pd.concat([frame[field] for frame in dataframes])
        )
        for field in (
            "business_name_normalized",
            "business_address_normalized",
            "country_normalized"
        )
    }


# ---------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------

def jaccard_similarity(set1, set2):

    if not set1 or not set2:
        return 0.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        return 0.0

    return intersection / union


# ---------------------------------------------------------
# Character similarity
# ---------------------------------------------------------

def character_similarity(text1, text2):

    if not text1 or not text2:
        return 0.0

    return ratio(
        str(text1),
        str(text2)
    ) / 100.0


# ---------------------------------------------------------
# Calculate features for one pair
# ---------------------------------------------------------

def calculate_pair_features(
    source1_row,
    candidate_row,
    missing_value_profile=None
):

    name1 = source1_row["business_name_normalized"]
    name2 = candidate_row["business_name_normalized"]

    address1 = source1_row["business_address_normalized"]
    address2 = candidate_row["business_address_normalized"]

    country1 = source1_row["country_normalized"]
    country2 = candidate_row["country_normalized"]

    if missing_value_profile is None:
        missing_value_profile = {}

    name_missing = _is_missing(name1) or _is_missing(name2)
    address_missing = _is_missing(address1) or _is_missing(address2)
    country_missing = _is_missing(country1) or _is_missing(country2)

    name_missing_probability = missing_value_profile.get(
        "business_name_normalized",
        0.5
    )
    address_missing_probability = missing_value_profile.get(
        "business_address_normalized",
        0.5
    )
    country_missing_probability = missing_value_profile.get(
        "country_normalized",
        0.5
    )

    name1 = "" if _is_missing(name1) else str(name1)
    name2 = "" if _is_missing(name2) else str(name2)
    address1 = "" if _is_missing(address1) else str(address1)
    address2 = "" if _is_missing(address2) else str(address2)
    country1 = "" if _is_missing(country1) else str(country1)
    country2 = "" if _is_missing(country2) else str(country2)

    # Token sets
    name_tokens1 = set(name1.split())
    name_tokens2 = set(name2.split())

    address_tokens1 = set(address1.split())
    address_tokens2 = set(address2.split())

    features = {

        # -----------------------------
        # Business name features
        # -----------------------------

        "name_exact": (
            name_missing_probability
            if name_missing
            else int(name1 == name2 and name1 != "")
        ),

        "name_jaccard": (
            name_missing_probability
            if name_missing
            else jaccard_similarity(name_tokens1, name_tokens2)
        ),

        "name_sequence": (
            name_missing_probability
            if name_missing
            else character_similarity(name1, name2)
        ),

        "name_missing_probability": (
            name_missing_probability if name_missing else 0.0
        ),

        # -----------------------------
        # Address features
        # -----------------------------

        "address_exact": (
            address_missing_probability
            if address_missing
            else int(address1 == address2 and address1 != "")
        ),

        "address_jaccard": (
            address_missing_probability
            if address_missing
            else jaccard_similarity(address_tokens1, address_tokens2)
        ),

        "address_sequence": (
            address_missing_probability
            if address_missing
            else character_similarity(address1, address2)
        ),

        "address_missing_probability": (
            address_missing_probability if address_missing else 0.0
        ),

        # -----------------------------
        # Country feature
        # -----------------------------

        "country_match": (
            country_missing_probability
            if country_missing
            else int(
                country1 != ""
                and country2 != ""
                and country1 == country2
            )
        ),

        "country_missing_probability": (
            country_missing_probability if country_missing else 0.0
        ),

        # -----------------------------
        # Length features
        # -----------------------------

        "name_length_diff": abs(
            len(name1) - len(name2)
        ),

        "address_length_diff": abs(
            len(address1) - len(address2)
        )
    }

    return features


# ---------------------------------------------------------
# Create feature dataframe
# ---------------------------------------------------------

def create_feature_dataframe(
    source1,
    source2,
    source3,
    candidate_df
):

    # Make lookup dictionaries
    source2_lookup = source2.set_index(
        "entity_id"
    ).to_dict("index")

    source3_lookup = source3.set_index(
        "entity_id"
    ).to_dict("index")

    missing_value_profile = _build_missing_value_profile(
        source1,
        source2,
        source3
    )

    source1_lookup = source1.set_index(
        "entity_id"
    ).to_dict("index")

    feature_rows = []

    total_pairs = 0

    for _, candidate_row in candidate_df.iterrows():

        source1_id = candidate_row[
            "source1_entity_id"
        ]

        candidate_ids = candidate_row[
            "candidate_entity_ids"
        ]

        source1_data = source1_lookup[
            source1_id
        ]

        for candidate_id in candidate_ids:

            # Determine whether candidate belongs
            # to Source 2 or Source 3
            if str(candidate_id).startswith("S2-"):

                candidate_data = source2_lookup.get(
                    candidate_id
                )

                candidate_source = "S2"

            else:

                candidate_data = source3_lookup.get(
                    candidate_id
                )

                candidate_source = "S3"

            if candidate_data is None:
                continue

            features = calculate_pair_features(
                source1_data,
                candidate_data,
                missing_value_profile
            )

            # Store IDs along with features
            features["source1_entity_id"] = source1_id
            features["candidate_entity_id"] = candidate_id
            features["candidate_source"] = candidate_source
            feature_rows.append(features)
            total_pairs += 1
            if total_pairs % 10000 == 0:
                print(
                    "Feature pairs created:",
                    total_pairs
                )
    return pd.DataFrame(feature_rows)
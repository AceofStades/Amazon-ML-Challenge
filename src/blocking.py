import re
import pandas as pd
# ---------------------------------------------------------
# 1. Get tokens from text
# ---------------------------------------------------------
def get_tokens(text):
    if pd.isna(text):
        return set()
    text = str(text).lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return set(tokens)
# ---------------------------------------------------------
# 2. Create blocking information
# ---------------------------------------------------------
def create_blocking_keys(df):
    df = df.copy()
    df["name_tokens"] = df["business_name_normalized"].apply(get_tokens)
    df["address_tokens"] = df["business_address_normalized"].apply(get_tokens)
    # Remove spaces before creating prefixes
    df["name_compact"] = (
        df["business_name_normalized"]
        .str.replace(" ", "", regex=False)
    )
    df["address_compact"] = (
        df["business_address_normalized"]
        .str.replace(" ", "", regex=False)
    )
    # Name prefixes
    df["name_prefix_3"] = df["name_compact"].str[:3]
    df["name_prefix_4"] = df["name_compact"].str[:4]
    # Address prefixes
    df["address_prefix_4"] = df["address_compact"].str[:4]
    df["address_prefix_6"] = df["address_compact"].str[:6]
    return df
# ---------------------------------------------------------
# 3. Build blocking indexes
# ---------------------------------------------------------
def build_blocking_index(df):
    index = {
        "name_token": {},
        "name_prefix_3": {},
        "name_prefix_4": {},
        "address_token": {},
        "address_prefix_4": {},
        "address_prefix_6": {}
    }
    for idx, row in df.iterrows():
        # -------------------------
        # Name tokens
        # -------------------------
        for token in row["name_tokens"]:
            if len(token) < 2:
                continue
            if token not in index["name_token"]:
                index["name_token"][token] = set()

            index["name_token"][token].add(idx)
        # -------------------------
        # Name prefixes
        # -------------------------
        prefix = row["name_prefix_3"]
        if prefix:
            if prefix not in index["name_prefix_3"]:
                index["name_prefix_3"][prefix] = set()
            index["name_prefix_3"][prefix].add(idx)
        prefix = row["name_prefix_4"]
        if prefix:
            if prefix not in index["name_prefix_4"]:
                index["name_prefix_4"][prefix] = set()
            index["name_prefix_4"][prefix].add(idx)
        # -------------------------
        # Address tokens
        # -------------------------
        for token in row["address_tokens"]:
            if len(token) < 2:
                continue
            if token not in index["address_token"]:
                index["address_token"][token] = set()
            index["address_token"][token].add(idx)
        # -------------------------
        # Address prefixes
        # -------------------------
        prefix = row["address_prefix_4"]
        if prefix:
            if prefix not in index["address_prefix_4"]:
                index["address_prefix_4"][prefix] = set()
            index["address_prefix_4"][prefix].add(idx)
        prefix = row["address_prefix_6"]
        if prefix:
            if prefix not in index["address_prefix_6"]:
                index["address_prefix_6"][prefix] = set()
            index["address_prefix_6"][prefix].add(idx)
    return index
# ---------------------------------------------------------
# 4. Generate candidates for one S1 record
# ---------------------------------------------------------
def generate_candidates(source1_row, source_df, index):
    candidate_scores = {}
    # -----------------------------------------
    # Name token matches
    # -----------------------------------------
    for token in source1_row["name_tokens"]:
        if len(token) < 2:
            continue
        matching_indices = index["name_token"].get(token, set())
        for idx in matching_indices:
            if idx not in candidate_scores:
                candidate_scores[idx] = 0
            candidate_scores[idx] += 3
    # -----------------------------------------
    # Strong name prefix
    # -----------------------------------------
    name_prefix = source1_row["name_prefix_4"]
    if name_prefix:
        matching_indices = index["name_prefix_4"].get(
            name_prefix,
            set()
        )
        for idx in matching_indices:
            if idx not in candidate_scores:
                candidate_scores[idx] = 0
            candidate_scores[idx] += 3
    # -----------------------------------------
    # Shorter name prefix
    # -----------------------------------------
    name_prefix = source1_row["name_prefix_3"]
    if name_prefix:
        matching_indices = index["name_prefix_3"].get(
            name_prefix,
            set()
        )
        for idx in matching_indices:
            if idx not in candidate_scores:
                candidate_scores[idx] = 0
            candidate_scores[idx] += 1
    # -----------------------------------------
    # Address token matches
    # -----------------------------------------
    for token in source1_row["address_tokens"]:
        if len(token) < 2:
            continue
        matching_indices = index["address_token"].get(token, set())
        for idx in matching_indices:
            if idx not in candidate_scores:
                candidate_scores[idx] = 0
            candidate_scores[idx] += 2
    # -----------------------------------------
    # Address prefix
    # -----------------------------------------
    address_prefix = source1_row["address_prefix_6"]
    if address_prefix:
        matching_indices = index["address_prefix_6"].get(
            address_prefix,
            set()
        )
        for idx in matching_indices:
            if idx not in candidate_scores:
                candidate_scores[idx] = 0
            candidate_scores[idx] += 2
    # -----------------------------------------
    # Shorter address prefix
    # -----------------------------------------
    address_prefix = source1_row["address_prefix_4"]
    if address_prefix:
        matching_indices = index["address_prefix_4"].get(
            address_prefix,
            set()
        )
        for idx in matching_indices:
            if idx not in candidate_scores:
                candidate_scores[idx] = 0
            candidate_scores[idx] += 1
    # -----------------------------------------
    # Country is only a bonus
    # -----------------------------------------
    source_country = source1_row["country_normalized"]
    if source_country:
        for idx in list(candidate_scores.keys()):
            candidate_country = source_df.iloc[idx]["country_normalized"]
            if candidate_country == source_country:
                candidate_scores[idx] += 2
            else:
                # Different countries are not immediately removed.
                # This helps with noisy/missing country information.
                candidate_scores[idx] -= 1
    # -----------------------------------------
    # Keep candidates with enough evidence
    # -----------------------------------------
    selected_candidates = []
    for idx, score in candidate_scores.items():
        if score >= 3:
            selected_candidates.append(
                (idx, score)
            )
    # Sort strongest candidates first
    selected_candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )
    # Convert indices to entity IDs
    candidate_entity_ids = []
    for idx, score in selected_candidates:
        entity_id = source_df.iloc[idx]["entity_id"]
        candidate_entity_ids.append(entity_id)
    return candidate_entity_ids
# ---------------------------------------------------------
# 5. Generate candidates for all S1 records
# ---------------------------------------------------------
def generate_all_candidates(source1, source2, source3):
    source1 = create_blocking_keys(source1)
    source2 = create_blocking_keys(source2)
    source3 = create_blocking_keys(source3)
    index2 = build_blocking_index(source2)
    index3 = build_blocking_index(source3)
    results = []
    total = len(source1)
    for count, (_, row) in enumerate(
        source1.iterrows(),
        start=1
    ):
        if count % 1000 == 0:
            print(
                "Processed:",
                count,
                "/",
                total
            )
        # Generate candidates separately for S2 and S3
        candidates2 = generate_candidates(
            row,
            source2,
            index2
        )
        candidates3 = generate_candidates(
            row,
            source3,
            index3
        )
        all_candidates = candidates2 + candidates3
        results.append({
            "source1_entity_id": row["entity_id"],
            "candidate_entity_ids": all_candidates
        })
    return pd.DataFrame(results)
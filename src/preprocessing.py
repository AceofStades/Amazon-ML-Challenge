# preprocessing.py
import re
import pandas as pd
# --------------------------------------------------
# BASIC TEXT NORMALIZATION
# --------------------------------------------------
def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Replace '&' with 'and'
    text = text.replace("&", " and ")
    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )
    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )
    return text.strip()
# --------------------------------------------------
# NORMALIZE BUSINESS NAME
# --------------------------------------------------
def normalize_business_name(name):
    name = normalize_text(name)
    # Common business abbreviations
    replacements = {
        "corporation": "corp",
        "company": "co",
        "limited": "ltd",
        "private": "pvt",
        "incorporated": "inc"
    }
    words = name.split()
    normalized_words = []
    for word in words:
        if word in replacements:
            word = replacements[word]
        normalized_words.append(word)
    return " ".join(normalized_words)
# --------------------------------------------------
# NORMALIZE BUSINESS ADDRESS
# --------------------------------------------------
def normalize_business_address(address):
    address = normalize_text(address)
    # Common address abbreviations
    replacements = {
        "road": "rd",
        "street": "st",
        "avenue": "ave",
        "boulevard": "blvd",
        "lane": "ln",
        "building": "bldg",
        "apartment": "apt",
        "highway": "hwy"
    }
    words = address.split()
    normalized_words = []
    for word in words:
        if word in replacements:
            word = replacements[word]
        normalized_words.append(word)
    return " ".join(normalized_words)
# --------------------------------------------------
# PROCESS COMPLETE DATAFRAME
# --------------------------------------------------
def preprocess_dataframe(df):
    df = df.copy()
    df["business_name_normalized"] = (
        df["business_name"]
        .apply(normalize_business_name)
    )
    df["business_address_normalized"] = (
        df["business_address"]
        .apply(normalize_business_address)
    )
    # Country is kept as a string.
    # We do NOT restrict it to US/India because
    # the test set also contains France.
    df["country_normalized"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    return df
# --------------------------------------------------
# PROCESS ALL THREE SOURCES
# --------------------------------------------------
def preprocess_sources(
    source1,
    source2,
    source3
):
    source1 = preprocess_dataframe(source1)
    source2 = preprocess_dataframe(source2)
    source3 = preprocess_dataframe(source3)
    return source1, source2, source3
# --------------------------------------------------
# TESTING
# --------------------------------------------------
if __name__ == "__main__":
    sample_data = pd.DataFrame({
        "entity_id": ["S1-00001"],
        "business_name": ["ABC Pvt. Ltd."],
        "business_address": [
            "123, MG Road, Mumbai"
        ],
        "country": ["India"]
    })
    result = preprocess_dataframe(
        sample_data
    )
    print(result)
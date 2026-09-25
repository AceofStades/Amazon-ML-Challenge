# Entity Resolution & Business Matching Dataset

## Overview
This directory contains the cleaned, pre-processed, and schema-standardized datasets for the **Business Entity Matching / Record Linkage Pipeline**. The dataset is structured specifically for tree-based Machine Learning models (e.g., XGBoost, LightGBM) and pairwise record-linkage algorithms optimized for $F_{0.5}$ scoring.

All raw address components have been parsed, token-deduplicated, and segregated without information loss across all sources.

---

## Directory Structure
dataset_clean/
└── train/
├── train_source1.tsv   # Reference source dataset
├── train_source2.tsv   # Target source dataset 2
└── train_source3.tsv   # Target source dataset 3

> **Note on Data Access:** Due to GitHub's 100 MB file size limit, the raw `.tsv` files (~1.9 GB total) and `dataset_clean.zip` are excluded from Git history. Please download the dataset zip from our [GitHub Release / Google Drive Link] and extract it into `dataset_clean/train/`.

---

## Master Schema Specification (11 Columns)

The dataset strictly adheres to an 11-column ML-optimized schema with zero field redundancy. Redundant full-address strings (`address_cleaned`) and redundant concatenated fields (`street_address`) have been dropped to prevent artificial feature importance weighting in gradient boosted trees.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `id` | `String` | Unique record identifier (e.g., `S1-925783039`, `S2-294835210`) |
| `business_name` | `String` | Raw or normalized title of the business entity |
| `building_landmark` | `String` | Sub-building specs, plot/flat numbers, landmarks, and suites |
| `road_area` | `String` | Street names, thoroughfares, avenues, sectors, and localities |
| `city` | `String` | Standardized city / municipality name |
| `state` | `String` | Standardized state, province, or region |
| `country` | `String` | Country name |
| `business_name_is_modified` | `Int8` / `Flag` | Binary flag (`1` or `0`) indicating whether the business name was modified |
| `business_name_word_count` | `Int16` | Token count of non-empty words in `business_name` |
| `building_landmark_word_count` | `Int16` | Token count of non-empty words in `building_landmark` |
| `road_area_word_count` | `Int16` | Token count of non-empty words in `road_area` |

## Business Name (`business_name`) Processing Strategy

Business names present unique challenges in record linkage due to varying legal suffixes, abbreviations, typos, and formatting noise. The pipeline processes entity names using three key transformations:

1. **Text Standardization & Suffix Handling:**
   - Case normalization and removal of non-informative punctuation/symbols.
   - Selective cleaning of generic entity noise (e.g., common legal identifiers like *LLC*, *Inc.*, *Pvt Ltd*, *Corp*) while retaining distinctive brand tokens.

2. **Modification Audit Tracking (`business_name_is_modified`):**
   - A binary tracking flag (`1` = Modified, `0` = Untouched) records whether the raw name underwent standardization or token removal.
   - **ML Utility:** Serves as an explicit feature signaling to tree models whether a partial string mismatch might be an artifact of normalization rather than a distinct entity.

3. **Token Density Granularity (`business_name_word_count`):**
   - Calculates the exact token count of the cleaned business title.
   - **ML Utility:** Assists in adjusting fuzzy matching distance scales—mismatches in single-token names (e.g., *"Cobalt"*) carry higher weight penalties than single-token differences in long multi-word titles (e.g., *"Piedmont Complete Healthcare Solution LLC"*).

---

## Cleaning & Address Parsing Architecture

1. **Dynamic Multi-Tier Address Segregation:**
   - Automatically segregates building/landmark specifics from thoroughfares and road names across both Indian and international (e.g., US) address formats without relying on static city lookup lists.
2. **Token-Level Deduplication:**
   - Eliminates repeated city/state names or duplicate street tokens from appearing simultaneously in `building_landmark` and `road_area`.
3. **Lossless Audit Verification:**
   - 100% token preservation audit verified prior to dropping `street_address`. All unassigned street tokens were dynamically rescued into `building_landmark` or `road_area`.
4. **ML Feature Readiness:**
   - Individual word count metrics (`*_word_count`) enable models to evaluate field density and missingness during candidate pair feature scoring.

---

## Quickstart & Loading with Polars

```python
import polars as pl

# Load cleaned partitions
source1 = pl.read_csv("dataset_clean/train/train_source1.tsv", separator="\t")
source2 = pl.read_csv("dataset_clean/train/train_source2.tsv", separator="\t")
source3 = pl.read_csv("dataset_clean/train/train_source3.tsv", separator="\t")

print(f"Source 1 Schema ({len(source1.columns)} columns):")
print(source1.schema)
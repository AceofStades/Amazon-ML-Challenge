"""Recall rule for countries without labels: add back high-precision noise patterns.

Calibrated decisions for an unlabelled country (France) drop mid-score pairs, and
the raw stage-1 decisions add them back indiscriminately (about half are decoys:
generic words swapped at the same address). This keeps the calibrated decisions
and adds a raw-only pair only when the target sits at the Source 1 address and
its name follows a noise pattern that decoys do not use: an acronym, a web
domain or handle, a DBA/f.k.a. alias, a near-typo of the name, or the Source 1
name plus extra words. Targets already assigned to another entity are skipped.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import polars as pl

ALIAS = re.compile(r"\b(dba|d/b/a|d\.b\.a|f/k/a|fka|formerly|aka|a/k/a|nee|née|doing business|trading as|t/a)\b")
DOMAIN = re.compile(r"\.(com|fr|net|org)\b|^www\.|^#")


def read(path: str) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)


def pairs(path: str) -> pl.DataFrame:
    return (read(path).filter(pl.col("matched_entity_ids") != "")
            .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids"))


def kind(name: str | None) -> str:
    text = (name or "").strip()
    low = text.lower()
    if DOMAIN.search(low):
        return "domain"
    if ALIAS.search(low):
        return "alias"
    if re.fullmatch(r"[A-Z0-9]{1,4}", text):
        return "acronym"
    return "other"


STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "and", "the", "of", "&"}


def initials(name: str, stop: bool) -> str:
    words = re.findall(r"[a-z0-9]+", (name or "").lower())
    return "".join(w[0] for w in words if not (stop and w in STOP))


def consistent(kind_: str, target: str, source: str, core: str) -> bool:
    """An acronym must spell the Source 1 initials; a domain must carry a Source 1 word or its initials.
    Decoy businesses at the same address have their own names, so their acronyms and domains differ."""
    t = re.sub(r"[^a-z0-9]", "", (target or "").lower())
    if kind_ == "alias":
        return True
    if kind_ == "acronym":
        return any(t and (init.startswith(t) or t.startswith(init[:2])) and init[:1] == t[:1]
                   for init in (initials(source, True), initials(source, False)))
    words = [w for w in (core or "").split() if len(w) >= 4]
    return any(w in t for w in words) or any(len(i) >= 2 and t.startswith(i)
                                             for i in (initials(source, True), initials(source, False)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated", required=True, help="matching_results.tsv with calibrated decisions")
    parser.add_argument("--raw", required=True, help="matching_results.tsv with raw decisions for the country")
    parser.add_argument("--country", default="France")
    parser.add_argument("--output", required=True)
    parser.add_argument("--typo-jw", type=float, default=0.92)
    args = parser.parse_args()

    s1 = read("dataset/test/test_source1.tsv").with_row_index("si")
    targets = pl.concat([read(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti")
    norm_s = read("cache/test_source1.tsv").select("entity_id", pl.col("name_core").alias("s_core"))
    norm_t = pl.concat([read(f"cache/test_source{i}.tsv") for i in (2, 3)]).select(
        "entity_id", pl.col("name_core").alias("t_core"))
    calibrated, raw = pairs(args.calibrated), pairs(args.raw)
    gap = (raw.join(calibrated, on=["source1_entity_id", "matched_entity_ids"], how="anti")
           .join(s1.select("entity_id", "si", "country"), left_on="source1_entity_id", right_on="entity_id")
           .filter(pl.col("country") == args.country)
           .join(targets.select("entity_id", "ti", pl.col("business_name").alias("t_name")),
                 left_on="matched_entity_ids", right_on="entity_id")
           .join(norm_s, left_on="source1_entity_id", right_on="entity_id")
           .join(norm_t, left_on="matched_entity_ids", right_on="entity_id"))
    features = pl.concat([
        pl.read_parquet(p, columns=["si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_compact_jw"])
        .join(gap.select("si", "ti"), on=["si", "ti"], how="semi")
        for p in sorted(glob.glob("cache/v3/test/part*.parquet"))])
    gap = gap.join(features, on=["si", "ti"], how="left").with_columns(
        pl.col("t_name").map_elements(kind, return_dtype=pl.String).alias("kind"),
        ((pl.col("house_conflict") != 1)
         & ((pl.col("house_match") == 1) | (pl.col("address_idf_cos") > 0.8))).alias("same_address"),
        pl.struct("s_core", "t_core").map_elements(
            lambda r: bool(set((r["s_core"] or "").split()) and set((r["s_core"] or "").split())
                           <= set((r["t_core"] or "").split())), return_dtype=pl.Boolean).alias("contains"))
    s_names = s1.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("business_name").alias("s_name"))
    gap = gap.join(s_names, on="source1_entity_id", how="left").with_columns(
        pl.struct("kind", "t_name", "s_name", "s_core").map_elements(
            lambda r: consistent(r["kind"], r["t_name"], r["s_name"], r["s_core"]), return_dtype=pl.Boolean)
        .alias("consistent"))
    keep = gap.filter(pl.col("same_address") & (
        (pl.col("kind").is_in(["acronym", "domain", "alias"]) & pl.col("consistent"))
        | ((pl.col("kind") == "other") & ((pl.col("name_compact_jw") >= args.typo_jw) | pl.col("contains")))))
    taken = calibrated.select("matched_entity_ids").unique()
    keep = keep.join(taken, on="matched_entity_ids", how="anti").unique("matched_entity_ids", keep="first")
    print(f"{args.country} raw-only pairs {gap.height:,}; added back {keep.height:,}", flush=True)
    print(keep.group_by("kind").len().sort("len", descending=True).rows(), flush=True)

    merged = pl.concat([calibrated.select("source1_entity_id", "matched_entity_ids"),
                        keep.select("source1_entity_id", "matched_entity_ids")])
    lists = merged.group_by("source1_entity_id").agg(
        pl.col("matched_entity_ids").unique().sort().str.join(","))
    out = s1.select(pl.col("entity_id").alias("source1_entity_id")).join(
        lists, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
    out.write_csv(args.output, separator="\t", quote_style="never")
    print(f"wrote {args.output}: {out.height:,} rows, {merged.height:,} matches", flush=True)


if __name__ == "__main__":
    main()

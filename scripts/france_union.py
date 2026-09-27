"""France blend union: add France pairs that the cross-encoder blend scores above a threshold but the France decisions
(stage 1 + calibration + rules) reject, when no other entity holds the target.

Training out-of-fold blend scores above 0.98 are 99.9-100% true in every house-relation x name-change class, and the
blend rejects France's decoy classes (of ~296k France near conflicts that add words, 113 score above 0.98). The
blend was not used as the France scorer because it under-scores true records with French legal-suffix changes; that
is a recall problem, so adding its confident pairs is safe. Excluded: common-word swaps and single made-up brands
(France-specific decoy-heavy classes, kept to the existing France rules) and near conflicts that add words (the
France decoy signature). One entity per target (highest p2).
Usage: python scripts/france_union.py --base A/matching_results.tsv --scores output/blend_X/test_scored.parquet \
           --output B/matching_results.tsv [--threshold 0.98] [--with-swaps]
"""

import argparse

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.98)
    ap.add_argument("--with-swaps", action="store_true", help="also add common-word swaps")
    args = ap.parse_args()

    s1 = R("dataset/test/test_source1.tsv", ["entity_id", "country"]).with_row_index("si")
    tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
    fr = s1.filter(pl.col("country") == "France").select("si")
    sc = pl.read_parquet(args.scores, columns=["si", "ti", "p2"]).join(fr, on="si", how="semi").filter(pl.col("p2") > args.threshold)
    links = (R(args.base).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids").join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
             .join(tg.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))
    cand = sc.join(links.select("ti").unique(), on="ti", how="anti")
    gen = pl.read_parquet("cache/gen/test.parquet", columns=["si", "ti", "g_swap_common", "g_brand1"])
    cls = pl.read_parquet("cache/class_audit_fr.parquet", columns=["si", "ti", "h", "nd"])
    cand = cand.join(gen, on=["si", "ti"], how="left").join(cls, on=["si", "ti"], how="left")
    bad = (pl.col("g_brand1") == 1) | ((pl.col("h") == "near") & pl.col("nd").is_in(["add", "adddrop"]))
    if not args.with_swaps:
        bad = bad | (pl.col("g_swap_common") == 1)
    add = cand.filter(~bad.fill_null(False)).sort(["ti", "p2"], descending=[False, True]).unique("ti", keep="first", maintain_order=True)
    print(f"France pairs with p2 > {args.threshold} on free targets: {cand.height:,}; added after exclusions: {add.height:,}")
    print("  by class:", add.group_by(pl.col("h").fill_null("other"), pl.col("nd").fill_null("-")).len().sort("len", descending=True).head(12).rows())

    new = pl.concat([links, add.select("si", "ti")])
    new = new.join(s1.select("si", pl.col("entity_id").alias("source1_entity_id")), on="si").join(
        tg.select("ti", pl.col("entity_id").alias("t")), on="ti")
    grouped = new.group_by("source1_entity_id").agg(pl.col("t").sort().str.join(",").alias("matched_entity_ids"))
    out = s1.select(pl.col("entity_id").alias("source1_entity_id")).join(grouped, on="source1_entity_id", how="left").with_columns(
        pl.col("matched_entity_ids").fill_null(""))
    out.write_csv(args.output, separator="\t", quote_style="never")
    print(f"wrote {args.output}: {out.height:,} rows, {new.height:,} matches (base {links.height:,})")


if __name__ == "__main__":
    main()

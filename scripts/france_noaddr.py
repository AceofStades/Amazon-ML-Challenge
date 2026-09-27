"""France address-less targets whose normalized name core belongs to exactly one France Source 1 entity: assign the
target to that entity when no entity holds it and the pair is a scored candidate. Address-less test targets are
almost all true links (their count per entity equals training's, where 97.7% are true), and with a unique name core
the owner is the unique entity. Training precision of the same rule is printed first (--check).
Usage: python scripts/france_noaddr.py --base A/matching_results.tsv --scores output/blend_X/test_scored.parquet \
           --output B/matching_results.tsv [--min-p1 0.05] [--check]
"""

import argparse
import glob

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def unique_core_pairs(split: str, country: str | None) -> pl.DataFrame:
    s1 = R(f"cache/{split}_source1.tsv", ["country", "name_core"]).with_row_index("si")
    tg = pl.concat([R(f"cache/{split}_source{i}.tsv", ["country", "address", "name_core"]) for i in (2, 3)]).with_row_index("ti")
    if country:
        s1 = s1.filter(pl.col("country") == country)
        tg = tg.filter(pl.col("country") == country)
    k = s1.group_by("country", "name_core").agg(pl.len().alias("k"), pl.col("si").first())
    na = tg.filter(pl.col("address") == "").select("ti", "country", "name_core")
    return na.join(k.filter(pl.col("k") == 1), on=["country", "name_core"]).select("si", "ti")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base")
    ap.add_argument("--scores")
    ap.add_argument("--output")
    ap.add_argument("--min-p1", type=float, default=0.05)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        pairs = unique_core_pairs("train", None)
        s1 = R("dataset/train/train_source1.tsv", ["entity_id"]).with_row_index("si")
        tg = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
        gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids").join(s1, left_on="source1_entity_id", right_on="entity_id")
              .join(tg, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti").with_columns(pl.lit(1).alias("label")))
        p1 = pl.concat([pl.read_parquet(f).join(pairs, on=["si", "ti"], how="semi") for f in sorted(glob.glob("output/v7/train_p1/*.parquet"))])
        x = pairs.join(p1, on=["si", "ti"], how="left").join(gt, on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0))
        x = x.with_columns(pl.col("p1").is_not_null().alias("cand"))
        print("train unique-core address-less pairs:", x.height, "candidates:", x["cand"].sum(),
              "precision (all):", round(x["label"].mean(), 4), "precision (candidate, p1 >= min):",
              round(x.filter(pl.col("p1") >= args.min_p1)["label"].mean(), 4))
        print(x.filter(pl.col("cand")).with_columns(pl.col("p1").cut([0.05, 0.2, 0.5, 0.9]).alias("b")).group_by("b").agg(
            pl.len(), pl.col("label").mean().round(4)).sort("b").rows())
        if not args.base:
            return

    s1 = R("dataset/test/test_source1.tsv", ["entity_id"]).with_row_index("si")
    tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
    pairs = unique_core_pairs("test", "france")
    sc = pl.read_parquet(args.scores, columns=["si", "ti", "p1", "p2"]).join(pairs, on=["si", "ti"], how="semi")
    links = (R(args.base).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids").join(s1, left_on="source1_entity_id", right_on="entity_id")
             .join(tg, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))
    add = sc.join(links.select("ti").unique(), on="ti", how="anti").filter(pl.col("p1") >= args.min_p1)
    print(f"France unique-core address-less targets: {pairs.height:,}; scored candidates: {sc.height:,}; added: {add.height:,}")
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

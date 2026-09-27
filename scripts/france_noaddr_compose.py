"""File 17's France step: take France links to address-less records from the context composition model instead of the
stage-1 France decisions. The composition model is trained on the stage-1-only cross-validated scores of US/India
(output/cv_p1only.parquet) and applied to France stage-1 scores; only its decisions on address-less targets are used.
  1. python scripts/france_noaddr_compose.py prepare        -> cache/fr_p1_as_calibrated.parquet
  2. python scripts/compose.py test --cv-scores output/cv_p1only.parquet --test-calibrated cache/fr_p1_as_calibrated.parquet \
         --countries France --extra --output output/compose_fr_p1/matching_results.tsv
  3. python scripts/france_noaddr_compose.py merge BASE.tsv output/compose_fr_p1/matching_results.tsv OUT.tsv
"""
import glob
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
COL = "matched_entity_ids"


def prepare():
    ids = pl.read_parquet("cache/embeddings/test_source1_ids.parquet").select("country").with_row_index("si").with_columns(pl.col("si").cast(pl.UInt32))
    fr = ids.filter(pl.col("country") == "France").select("si")
    p = pl.concat([pl.read_parquet(f).join(fr, on="si", how="semi") for f in sorted(glob.glob("output/v7/test_p1/*.parquet"))])
    p = p.select(pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32), pl.col("p1").alias("p"), pl.col("p1").alias("raw"))
    p.write_parquet("cache/fr_p1_as_calibrated.parquet")
    print(f"wrote cache/fr_p1_as_calibrated.parquet: {p.height:,} France pairs")


def links(path, s1):
    return (R(path).filter(pl.col(COL) != "").with_columns(pl.col(COL).str.split(",")).explode(COL)
            .select("source1_entity_id", pl.col(COL).alias("t")).join(s1, left_on="source1_entity_id", right_on="entity_id"))


def merge(base_path, comp_path, out_path):
    s1 = R("dataset/test/test_source1.tsv", ["entity_id", "country"])
    tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id", "business_address"]) for i in (2, 3)])
    noaddr = tg.filter(pl.col("business_address") == "").select(pl.col("entity_id").alias("t"))
    base = links(base_path, s1)
    comp = links(comp_path, s1).filter(pl.col("country") == "France").join(noaddr, on="t", how="semi")
    keep = base.filter(~((pl.col("country") == "France") & pl.col("t").is_in(noaddr["t"].implode())))
    new = pl.concat([keep, comp.select(keep.columns)]).unique(["source1_entity_id", "t"])
    assert new.group_by("t").len().filter(pl.col("len") > 1).height == 0, "a target is assigned twice"
    g = new.group_by("source1_entity_id").agg(pl.col("t").sort().str.join(",").alias(COL))
    out = s1.select(pl.col("entity_id").alias("source1_entity_id")).join(g, on="source1_entity_id", how="left").with_columns(pl.col(COL).fill_null(""))
    out.write_csv(out_path, separator="\t", quote_style="never")
    print(f"wrote {out_path}: {new.height:,} links (base {base.height:,}); France address-less links "
          f"{base.filter((pl.col('country') == 'France') & pl.col('t').is_in(noaddr['t'].implode())).height:,} -> {comp.height:,}")


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        merge(*sys.argv[2:5])

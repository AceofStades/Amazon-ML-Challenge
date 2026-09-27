"""File 18: file 17 plus France same-house pure made-up brand records that the 29-model blend scores >= 0.95 and no
entity holds (one entity per target, the highest blend score). The source-balance test (scripts/fr_cand_balance.py)
estimates these at about 83% +/- 11% true (brands below blend 0.5: 6%; common-word swaps above 0.95: about 0%).
Usage: python scripts/france_brand_add.py BASE.tsv OUT.tsv [MIN_P2]"""
import glob
import re
import sys
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
COL = "matched_entity_ids"


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def pure_brand(sn, tn):
    t = re.findall(r"[A-Za-z]+", fold(tn))
    if len(t) != 1 or len(t[0]) < 6 or re.search(r"[#@.]|com$", tn or ""):
        return False
    s = re.findall(r"[a-z]+", fold(sn))
    return t[0] not in s and not any(len(x) >= 4 and x in t[0] for x in s)


base_path, out_path = sys.argv[1], sys.argv[2]
min_p2 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.95
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "business_name", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id", "business_name"]) for i in (2, 3)]).with_row_index("ti")
base = (R(base_path).filter(pl.col(COL) != "").with_columns(pl.col(COL).str.split(",")).explode(COL)
        .select("source1_entity_id", pl.col(COL).alias("t")))
held = base.select("t").unique()
fr = s1.filter(pl.col("country") == "France").select("si")
sc = pl.read_parquet("output/blend_fleetZ/test_scored.parquet", columns=["si", "ti", "p2"]).join(fr, on="si", how="semi").filter(pl.col("p2") >= min_p2)
f = pl.concat([pl.read_parquet(p, columns=["si", "ti", "house_match", "address_idf_cos"]).join(sc.select("si", "ti"), on=["si", "ti"], how="semi")
               .filter((pl.col("house_match") == 1) & (pl.col("address_idf_cos") >= 0.6)) for p in sorted(glob.glob("cache/v3/test/part*.parquet"))])
c = sc.join(f.select("si", "ti"), on=["si", "ti"], how="semi")
c = c.with_columns(s1["business_name"].gather(c["si"].to_numpy()).alias("sn"), tg["business_name"].gather(c["ti"].to_numpy()).alias("tn"),
                   s1["entity_id"].gather(c["si"].to_numpy()).alias("source1_entity_id"), tg["entity_id"].gather(c["ti"].to_numpy()).alias("t"))
c = c.filter(pl.struct("sn", "tn").map_elements(lambda r: pure_brand(r["sn"], r["tn"]), return_dtype=pl.Boolean))
c = c.join(held, on="t", how="anti").sort("p2", descending=True).unique("t", keep="first")
print(f"adding {c.height:,} France brand links (blend >= {min_p2})")
for r in c.sample(min(8, c.height), seed=2).iter_rows(named=True):
    print(f"  {r['p2']:.3f} {r['sn'][:36]:36} || {r['tn']}")
new = pl.concat([base, c.select("source1_entity_id", "t")])
assert new.group_by("t").len().filter(pl.col("len") > 1).height == 0
g = new.group_by("source1_entity_id").agg(pl.col("t").sort().str.join(",").alias(COL))
out = s1.select(pl.col("entity_id").alias("source1_entity_id")).join(g, on="source1_entity_id", how="left").with_columns(pl.col(COL).fill_null(""))
out.write_csv(out_path, separator="\t", quote_style="never")
print(f"wrote {out_path}: {new.height:,} links (base {base.height:,})")

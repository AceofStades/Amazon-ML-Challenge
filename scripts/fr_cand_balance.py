"""Source-balance check of unassigned France candidates (free targets): brands and common-word swaps at the Source 1
house, by blend score. For an entity whose accepted links (2 or 3) all sit in one source, a true extra record lands in
the other source about as often as France's exact-name links do (reference), a decoy half the time. The common-word
swaps with blend >= 0.9 are the class file 12 added and the leaderboard measured at about 25% true: a calibration.
Usage: python scripts/fr_cand_balance.py SUBMISSION"""
import glob
import re
import sys
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def pure_brand(sn, tn):
    t = re.findall(r"[A-Za-z]+", fold(tn))
    if len(t) != 1 or len(t[0]) < 6 or re.search(r"[#@.]|com$", tn or ""):
        return False
    s = re.findall(r"[a-z]+", fold(sn))
    return t[0] not in s and not any(len(x) >= 4 and x in t[0] for x in s)


sub = sys.argv[1]
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "business_name", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id", "business_name"]) for i in (2, 3)]).with_row_index("ti")
tg = tg.with_columns(pl.col("entity_id").str.starts_with("S3").cast(pl.Int32).alias("t3"))
col = "matched_entity_ids"
l = (R(sub).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
     .join(s1.select("entity_id", "si", "country"), left_on="source1_entity_id", right_on="entity_id")
     .join(tg.select("entity_id", "ti", "t3"), left_on=col, right_on="entity_id").select("si", "ti", "t3", "country"))
fr_l = l.filter(pl.col("country") == "France")
held = l.select("ti").unique()
ent = fr_l.group_by("si").agg(pl.len().alias("k"), pl.col("t3").sum().alias("n3"))
ent = ent.filter(pl.col("k").is_in([2, 3]) & ((pl.col("n3") == 0) | (pl.col("n3") == pl.col("k"))))
# reference: France accepted exact-name-like links in entities with k+1 links whose other k are one-source
# (computed with the same conditioning as the candidates: fill of a held-out accepted link)
g = pl.read_parquet("cache/gen/test.parquet", columns=["si", "ti", "g_exact", "g_brand1", "g_swap_common", "g_swap_inplace"])
ex = fr_l.join(g.filter(pl.col("g_exact") > 0.5).select("si", "ti"), on=["si", "ti"], how="semi")
e2 = fr_l.group_by("si").agg(pl.len().alias("K"), pl.col("t3").sum().alias("N3"))
ex = ex.join(e2, on="si").filter(pl.col("K").is_in([3, 4])).with_columns((pl.col("K") - 1).alias("kr"), (pl.col("N3") - pl.col("t3")).alias("n3r"))
ex = ex.filter((pl.col("n3r") == 0) | (pl.col("n3r") == pl.col("kr")))
ref = ex.with_columns(pl.when(pl.col("n3r") == 0).then(pl.col("t3") == 1).otherwise(pl.col("t3") == 0).alias("fill"))["fill"].mean()
print(f"France reference (accepted exact links): fill {ref:.3f}")
sc = pl.read_parquet("output/blend_fleetZ/test_scored.parquet", columns=["si", "ti", "p2"])
feats = pl.concat([pl.read_parquet(f, columns=["si", "ti", "house_match", "address_idf_cos"]).join(ent.select("si"), on="si", how="semi")
                   .filter((pl.col("house_match") == 1) & (pl.col("address_idf_cos") >= 0.6)) for f in sorted(glob.glob("cache/v3/test/part*.parquet"))])
c = feats.join(held, on="ti", how="anti").join(g, on=["si", "ti"], how="left").join(sc, on=["si", "ti"], how="left")
c = c.with_columns(s1["business_name"].gather(c["si"].to_numpy()).alias("sn"), tg["business_name"].gather(c["ti"].to_numpy()).alias("tn"),
                   tg["t3"].gather(c["ti"].to_numpy()).alias("t3"))
c = c.with_columns(pl.when(pl.struct("sn", "tn").map_elements(lambda r: pure_brand(r["sn"], r["tn"]), return_dtype=pl.Boolean)).then(pl.lit("brand"))
                   .when(pl.col("g_swap_common") > 0.5).then(pl.lit("swap_common")).otherwise(pl.lit(None)).alias("cls")).filter(pl.col("cls").is_not_null())
c = c.join(ent, on="si").with_columns(pl.when(pl.col("n3") == 0).then(pl.col("t3") == 1).otherwise(pl.col("t3") == 0).alias("fill"),
                                      pl.when(pl.col("p2") >= 0.95).then(pl.lit(">=0.95")).when(pl.col("p2") >= 0.5).then(pl.lit("0.5-0.95")).otherwise(pl.lit("<0.5")).alias("pb"))
t = c.group_by("cls", "pb").agg(pl.len().alias("n"), pl.col("fill").mean().round(3).alias("fill"))
t = t.with_columns(((pl.col("fill") - 0.5) / (ref - 0.5)).round(2).alias("pi"), (0.5 / (pl.col("n").sqrt() * (ref - 0.5))).round(2).alias("se"))
pl.Config.set_tbl_rows(20)
print(t.sort("cls", "pb"))

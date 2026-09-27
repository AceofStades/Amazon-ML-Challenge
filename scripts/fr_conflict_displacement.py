"""Displacement test on rejected same-street house-conflict candidates (target free, stage-1 p1 >= 0.3), split by raw
name difference and house-number gap, per country. True records with house-number noise make their entity short of
a link (negative delta, like accepted true classes at about -0.4); decoys leave it unchanged (US/India: about 0 or
positive). Writes the France candidates to cache/fr_conflict_cands.parquet.
Usage: python scripts/fr_conflict_displacement.py SUBMISSION"""
import glob
import re
import sys
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def toks(s):
    return frozenset(re.findall(r"[a-z0-9]+", fold(s)))


def gap(h, ta):
    try:
        hv = int(h)
    except (TypeError, ValueError):
        return None
    nums = [int(n) for n in re.findall(r"\d+", ta or "") if len(n) <= 5]
    return min(abs(n - hv) for n in nums) if nums else None


sub = sys.argv[1]
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "business_name", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id", "business_name", "business_address"]) for i in (2, 3)]).with_row_index("ti")
col = "matched_entity_ids"
l = (R(sub).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
     .join(s1.select("entity_id", "si", "country"), left_on="source1_entity_id", right_on="entity_id")
     .join(tg.select("entity_id", "ti"), left_on=col, right_on="entity_id").select("si", "ti", "country"))
held = l.select("ti").unique()
p1 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("output/v7/test_p1/*.parquet"))]).join(held, on="ti", how="anti").filter(pl.col("p1") >= 0.3)
c = pl.concat([pl.read_parquet(f, columns=["si", "ti", "house_conflict", "house_match", "address_idf_cos"]).join(p1, on=["si", "ti"], how="semi")
               for f in sorted(glob.glob("cache/v3/test/part*.parquet"))])
c = c.filter((pl.col("house_conflict") == 1) & (pl.col("house_match") != 1) & (pl.col("address_idf_cos") >= 0.6)).join(p1, on=["si", "ti"])
house = R("cache/test_source1.tsv", ["house"])["house"]
c = c.with_columns(s1["business_name"].gather(c["si"].to_numpy()).alias("sn"), tg["business_name"].gather(c["ti"].to_numpy()).alias("tn"),
                   house.gather(c["si"].to_numpy()).alias("hs"), tg["business_address"].gather(c["ti"].to_numpy()).alias("ta"),
                   s1["country"].gather(c["si"].to_numpy()).alias("country"))
nd = []
for a, b in zip(c["sn"].to_list(), c["tn"].to_list()):
    A, B = toks(a), toks(b)
    nd.append(("add" if B - A else "") + ("drop" if A - B else "") or "same")
c = c.with_columns(pl.Series("nd", nd), pl.struct("hs", "ta").map_elements(lambda s: gap(s["hs"], s["ta"]), return_dtype=pl.Int64).alias("gap"))
c = c.with_columns(pl.when(pl.col("gap").is_null()).then(pl.lit("?")).when(pl.col("gap") <= 2).then(pl.lit("1-2")).when(pl.col("gap") <= 10).then(pl.lit("3-10"))
                   .when(pl.col("gap") <= 30).then(pl.lit("11-30")).otherwise(pl.lit(">30")).alias("gb"))
out = []
for ctry in ("US", "India", "France"):
    ents = s1.filter(pl.col("country") == ctry).select("si")
    n = ents.join(l.filter(pl.col("country") == ctry).group_by("si").len().rename({"len": "n"}), on="si", how="left").with_columns(pl.col("n").fill_null(0))
    cc = c.filter(pl.col("country") == ctry)
    for (k, gb), grp in cc.group_by(["nd", "gb"]):
        e = grp.select("si").unique()
        if e.height < 150:
            continue
        a = n.join(e, on="si", how="semi"); b = n.join(e, on="si", how="anti")
        out.append((k, gb, ctry, grp.height, e.height, round(a["n"].mean() - b["n"].mean(), 3)))
t = pl.DataFrame(out, schema=["nd", "gap", "country", "pairs", "ents", "delta"], orient="row")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
print(t.pivot(on="country", index=["nd", "gap"], values="delta").sort("nd", "gap"))
print(t.pivot(on="country", index=["nd", "gap"], values="pairs").sort("nd", "gap"))
c.filter(pl.col("country") == "France").select("si", "ti", "p1", "nd", "gap", "gb", "sn", "tn", "hs", "ta").write_parquet("cache/fr_conflict_cands.parquet")

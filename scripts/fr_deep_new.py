"""New France pairs from the deep kNN (ranks 41..150) that are not candidates yet: how many look like true links
(identical raw name token set, same house number, same street), and whether the target is unclaimed."""
import glob
import re
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def tokset(s):
    out, run = [], ""
    for w in re.findall(r"[a-z0-9]+", fold(s)):
        if len(w) == 1 and w.isalpha():
            run += w
            continue
        if run:
            out.append(run); run = ""
        out.append(w)
    if run:
        out.append(run)
    return " ".join(sorted(set(out)))


deep = pl.read_parquet("cache/fr_deep_knn.parquet").filter(pl.col("rank") > 40).rename({"q": "si", "d": "ti"})
cand = pl.concat([pl.read_parquet(f, columns=["si", "ti"]).join(deep.select("si").unique(), on="si", how="semi") for f in sorted(glob.glob("cache/v3/test/part*.parquet"))])
new = deep.join(cand.with_columns(pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32)), on=["si", "ti"], how="anti")
print("deep pairs ranks 41-150:", deep.height, "not yet candidates:", new.height)
s1 = R("cache/test_source1.tsv", ["name_core", "address_core", "house"]).with_row_index("si").with_columns(
    R("dataset/test/test_source1.tsv", ["business_name"])["business_name"].alias("sn"))
tg = pl.concat([R(f"cache/test_source{i}.tsv", ["name_core", "address_core", "address"]) for i in (2, 3)]).with_row_index("ti").with_columns(
    pl.concat([R(f"dataset/test/test_source{i}.tsv", ["business_name"]) for i in (2, 3)])["business_name"].alias("tn"))
new = new.join(s1.rename({"name_core": "s_core", "address_core": "s_addr"}), on="si").join(tg.rename({"name_core": "t_core", "address_core": "t_addr"}), on="ti")
new = new.with_columns((pl.col("s_core") == pl.col("t_core")).alias("core_eq"),
                       (pl.col("address") == "").alias("noaddr"),
                       pl.struct("house", "address").map_elements(lambda r: bool(r["house"]) and r["house"] in re.findall(r"\d+", r["address"] or ""), return_dtype=pl.Boolean).alias("house_eq"),
                       pl.struct("s_addr", "t_addr").map_elements(lambda r: len(set((r["s_addr"] or "").split()) & set((r["t_addr"] or "").split())) / max(1, len(set((r["s_addr"] or "").split()))), return_dtype=pl.Float64).alias("addr_ov"))
sub = (R("output/final_ctxA/matching_results.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids"))
tgid = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
held = sub.join(tgid, left_on="matched_entity_ids", right_on="entity_id").select("ti").unique().with_columns(pl.lit(True).alias("held"))
new = new.join(held, on="ti", how="left").with_columns(pl.col("held").fill_null(False))
g = new.group_by("core_eq", "noaddr", "house_eq", (pl.col("addr_ov") >= 0.6).alias("street"), "held").len().sort("len", descending=True)
pl.Config.set_tbl_rows(30)
print(g.filter(pl.col("core_eq")))
strong = new.filter(pl.col("core_eq") & ~pl.col("held") & ((pl.col("house_eq") & (pl.col("addr_ov") >= 0.6)) | pl.col("noaddr")))
print("strong unclaimed: exact core + (same house & street, or no address):", strong.height, "entities:", strong["si"].n_unique())
new.select("si", "ti", "rank", "cosine", "core_eq", "noaddr", "house_eq", "addr_ov", "held").write_parquet("cache/fr_deep_new.parquet")
for r in strong.sample(min(10, strong.height), seed=1).iter_rows(named=True):
    print(f"  r={r['rank']} | {r['sn'][:30]:30} {r['house']:>5} {r['s_addr'][:30]:30} || {r['tn'][:30]:30} {r['address'][:34]}")

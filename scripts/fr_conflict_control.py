"""Control for the France house-conflict displacement signal: rejected same-street conflict candidates (target free, any
stage-1 score) grouped by the added word. Decoy-only words (international, holding, distribution, participations) set
the decoy baseline; shared words (groupe, developpement, france), legal forms and filler-only words (fils, associes, cie,
services) are compared against it. A group of true records shows a deficit beyond the decoy baseline."""
import glob
import re
import sys
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
DECOY = {"international", "holding", "distribution", "participations"}
SHARED = {"groupe", "developpement", "france"}
LF = {"sarl", "sas", "sasu", "eurl", "sa", "sci", "snc"}
FILLER = {"fils", "associes", "cie", "services", "compagnie"}


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def toks(s):
    return frozenset(re.findall(r"[a-z0-9]+", fold(s)))


def group(sn, tn):
    a, b = toks(sn), toks(tn)
    add = b - a
    if not add:
        return "no_add"
    if add & DECOY:
        return "decoy_word"
    if add <= SHARED | LF | {"et", "s", "a", "r", "l", "u", "e"}:
        if add & SHARED:
            return "shared_word"
        return "legal_form"
    if add & FILLER:
        return "filler_word"
    return "other_add"


sub = sys.argv[1] if len(sys.argv) > 1 else "output/final_ctxA/matching_results.tsv"
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "business_name", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id", "business_name"]) for i in (2, 3)]).with_row_index("ti")
fr = s1.filter(pl.col("country") == "France").select("si")
col = "matched_entity_ids"
l = (R(sub).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
     .join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
     .join(tg.select("entity_id", "ti"), left_on=col, right_on="entity_id").select("si", "ti"))
held = l.select("ti").unique()
c = pl.concat([pl.read_parquet(f, columns=["si", "ti", "house_conflict", "house_match", "address_idf_cos", "name_idf_cos"]).join(fr, on="si", how="semi")
               .filter((pl.col("house_conflict") == 1) & (pl.col("house_match") != 1) & (pl.col("address_idf_cos") >= 0.6) & (pl.col("name_idf_cos") >= 0.6))
               for f in sorted(glob.glob("cache/v3/test/part*.parquet"))]).join(held, on="ti", how="anti")
p1 = pl.concat([pl.read_parquet(f).join(c.select("si", "ti"), on=["si", "ti"], how="semi") for f in sorted(glob.glob("output/v7/test_p1/*.parquet"))])
c = c.join(p1, on=["si", "ti"], how="left").with_columns(pl.col("p1").fill_null(0.0))
c = c.with_columns(s1["business_name"].gather(c["si"].to_numpy()).alias("sn"), tg["business_name"].gather(c["ti"].to_numpy()).alias("tn"))
c = c.with_columns(pl.struct("sn", "tn").map_elements(lambda r: group(r["sn"], r["tn"]), return_dtype=pl.String).alias("grp"),
                   pl.when(pl.col("p1") >= 0.3).then(pl.lit("p>=0.3")).otherwise(pl.lit("p<0.3")).alias("pb"))
n = fr.join(l.group_by("si").len().rename({"len": "n"}), on="si", how="left").with_columns(pl.col("n").fill_null(0))
out = []
for (g, pb), grp in c.group_by(["grp", "pb"]):
    e = grp.select("si").unique()
    a = n.join(e, on="si", how="semi"); b = n.join(e, on="si", how="anti")
    out.append((g, pb, grp.height, e.height, round(a["n"].mean() - b["n"].mean(), 3)))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
print(pl.DataFrame(out, schema=["group", "p1", "pairs", "ents", "delta"], orient="row").sort("group", "p1"))
c.select("si", "ti", "p1", "grp", "sn", "tn").write_parquet("cache/fr_conflict_groups.parquet")

"""Same-street house conflicts whose raw names are identical after folding accents/case and merging dotted letters
(S.A.R.L. -> sarl). Training: true rate by gap and name difference. France test: count by target state (accepted,
unassigned, owned by another entity). Writes cache/fr_samename_conflicts.parquet (France candidates to restore)."""
import glob
import re
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def tokens(s):
    t = re.findall(r"[a-z0-9]+", fold(s))
    out, run = [], ""
    for w in t:  # merge runs of single letters (s a r l -> sarl)
        if len(w) == 1 and w.isalpha():
            run += w
            continue
        if run:
            out.append(run); run = ""
        out.append(w)
    if run:
        out.append(run)
    return set(out)


def diff(sn, tn):
    a, b = tokens(sn), tokens(tn)
    return ("add" if b - a else "") + ("drop" if a - b else "") or "same"


def gap(h, ta):
    try:
        hv = int(h)
    except ValueError:
        return None
    nums = [int(n) for n in re.findall(r"\d+", ta) if len(n) <= 5]
    return min(abs(n - hv) for n in nums) if nums else None


def frame(split, cond_country=None):
    cols = ["si", "ti", "house_conflict", "address_idf_cos", "name_idf_cos"]
    parts = []
    for f in sorted(glob.glob(f"cache/v3/{split}/part*.parquet")):
        parts.append(pl.read_parquet(f, columns=cols).filter((pl.col("house_conflict") == 1) & (pl.col("address_idf_cos") >= 0.6) & (pl.col("name_idf_cos") >= 0.9)))
    x = pl.concat(parts)
    s1 = R(f"cache/{split}_source1.tsv", ["country", "house"]).with_row_index("si")
    if cond_country:
        x = x.join(s1.filter(pl.col("country") == cond_country).select("si"), on="si", how="semi")
    addr = pl.concat([R(f"cache/{split}_source{i}.tsv", ["address"]) for i in (2, 3)])["address"].fill_null("")
    raw1 = R(f"dataset/{split}/{split}_source1.tsv", ["business_name"])["business_name"]
    raw2 = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["business_name"]) for i in (2, 3)])["business_name"]
    x = x.with_columns(raw1.gather(x["si"].to_numpy()).alias("sn"), raw2.gather(x["ti"].to_numpy()).alias("tn"),
                       s1["house"].fill_null("").gather(x["si"].to_numpy()).alias("hs"), addr.gather(x["ti"].to_numpy()).alias("ta"))
    x = x.with_columns(pl.struct("hs", "ta").map_elements(lambda s: gap(s["hs"], s["ta"]), return_dtype=pl.Int64).alias("gap"),
                       pl.struct("sn", "tn").map_elements(lambda s: diff(s["sn"], s["tn"]), return_dtype=pl.String).alias("nd"))
    return x.with_columns(pl.when(pl.col("gap") == 0).then(pl.lit("0")).when(pl.col("gap") <= 50).then(pl.lit("near"))
                          .otherwise(pl.lit("far")).alias("g"))


pl.Config.set_tbl_rows(40)
tr = frame("train")
s1id = R("dataset/train/train_source1.tsv", ["entity_id"]).with_row_index("si")
tgid = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").join(s1id, left_on="source1_entity_id", right_on="entity_id")
      .join(tgid, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti").with_columns(pl.lit(1).alias("label")))
tr = tr.join(gt, on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0))
print("train", tr.group_by("g", "nd").agg(pl.len().alias("n"), (pl.len() / 2206821).round(4).alias("per_E"),
                                           pl.col("label").mean().round(3).alias("true")).sort("g", "nd"))

fr = frame("test", "france")
s1t = R("dataset/test/test_source1.tsv", ["entity_id"]).with_row_index("si")
tgt = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
sub = (R("output/final_pre1/matching_results.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
       .explode("matched_entity_ids").join(s1t, left_on="source1_entity_id", right_on="entity_id")
       .join(tgt, left_on="matched_entity_ids", right_on="entity_id").select(pl.col("si").alias("owner"), "ti"))
fr = fr.join(sub, on="ti", how="left").with_columns(
    pl.when(pl.col("owner") == pl.col("si")).then(pl.lit("accepted")).when(pl.col("owner").is_not_null()).then(pl.lit("other")).otherwise(pl.lit("free")).alias("state"))
p1 = pl.concat([pl.read_parquet(f).join(fr.select("si", "ti"), on=["si", "ti"], how="semi") for f in sorted(glob.glob("output/v7/test_p1/*.parquet"))])
fr = fr.join(p1, on=["si", "ti"], how="left")
print("france", fr.group_by("g", "nd", "state").len().pivot(on="state", index=["g", "nd"], values="len").fill_null(0).sort("g", "nd"))
fr.select("si", "ti", "g", "nd", "gap", "state", "p1", "owner").write_parquet("cache/fr_samename_conflicts.parquet")

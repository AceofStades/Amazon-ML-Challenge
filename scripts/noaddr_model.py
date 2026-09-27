"""Feasibility: an owner model for address-less targets. Training, full competition (all entities, out-of-fold p1):
for each address-less target, its candidate entities (p1 >= 0.01) with features from the scores and raw-name
agreement; XGBoost on half of the targets, evaluated on the other half: precision/coverage of the arg-max owner
for targets the current scoring leaves ambiguous (top p1 < 0.9)."""
import glob
import re
import unicodedata

import numpy as np
import polars as pl
import xgboost as xgb

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


tg = pl.concat([R(f"cache/train_source{i}.tsv", ["address"]) for i in (2, 3)]).with_row_index("ti")
raw2 = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["business_name"]) for i in (2, 3)])["business_name"]
src = pl.concat([pl.Series("src", [i] * R(f"cache/train_source{i}.tsv", ["address"]).height, dtype=pl.Int8) for i in (2, 3)])
tg = tg.with_columns(raw2.alias("tn"), src)
na = tg.filter(pl.col("address") == "").select("ti", "tn", "src")
p = pl.concat([pl.read_parquet(f).join(na.select("ti"), on="ti", how="semi").filter(pl.col("p1") >= 0.01) for f in sorted(glob.glob("output/v7/train_p1/*.parquet"))])
s1id = R("dataset/train/train_source1.tsv", ["entity_id"]).with_row_index("si")
tgid = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").join(s1id, left_on="source1_entity_id", right_on="entity_id")
      .join(tgid, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))
p = p.join(gt.with_columns(pl.lit(1).alias("label")), on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0))
print("candidate pairs:", p.height, "targets:", p["ti"].n_unique(), "targets with owner among candidates:", p.filter(pl.col("label") == 1)["ti"].n_unique())

raw1 = R("dataset/train/train_source1.tsv", ["business_name"])["business_name"]
p = p.with_columns(raw1.gather(p["si"].to_numpy()).alias("sn")).join(na, on="ti")
p = p.with_columns(pl.col("sn").map_elements(tokset, return_dtype=pl.String).alias("s_tok"), pl.col("tn").map_elements(tokset, return_dtype=pl.String).alias("t_tok"),
                   pl.col("sn").map_elements(fold, return_dtype=pl.String).alias("s_fold"), pl.col("tn").map_elements(fold, return_dtype=pl.String).alias("t_fold"))
# entity's other (addressed) records: true links in training (the test analogue is the confident matches)
other = gt.join(tg.select("ti", "tn", "address", "src"), on="ti")
other = other.with_columns(pl.col("tn").map_elements(tokset, return_dtype=pl.String).alias("o_tok"))
ent = other.group_by("si").agg(pl.len().alias("e_n"), (pl.col("address") == "").sum().alias("e_noaddr"), (pl.col("src") == 2).sum().alias("e_s2"),
                               pl.col("o_tok"), pl.col("ti").alias("o_ti"))
p = p.join(ent, on="si", how="left")
p = p.with_columns(
    (pl.col("s_tok") == pl.col("t_tok")).cast(pl.Int8).alias("s_tok_eq"),
    (pl.col("s_fold") == pl.col("t_fold")).cast(pl.Int8).alias("s_fold_eq"),
    (pl.col("sn") == pl.col("tn")).cast(pl.Int8).alias("s_exact"),
    pl.struct("t_tok", "o_tok", "ti", "o_ti").map_elements(
        lambda r: int(any(o == r["t_tok"] for o, t in zip(r["o_tok"] or [], r["o_ti"] or []) if t != r["ti"])), return_dtype=pl.Int8).alias("o_tok_eq"),
    # counts that exclude the target itself (in test these come from predicted links)
    (pl.col("e_n").fill_null(0) - pl.col("label")).alias("e_n_other"),
    (pl.col("e_noaddr").fill_null(0) - pl.col("label")).alias("e_noaddr_other"),
    (pl.col("e_s2").fill_null(0) - pl.col("label") * (pl.col("src") == 2).cast(pl.Int64)).alias("e_s2_other"))
p = p.with_columns(pl.col("p1").rank("ordinal", descending=True).over("ti").alias("rank"), pl.len().over("ti").alias("m"),
                   pl.col("p1").max().over("ti").alias("top1"), (pl.col("p1") / pl.col("p1").sum().over("ti")).alias("q"),
                   pl.col("s_tok_eq").sum().over("ti").alias("n_s_tok_eq"), pl.col("s_fold_eq").sum().over("ti").alias("n_s_fold_eq"),
                   pl.col("o_tok_eq").sum().over("ti").alias("n_o_tok_eq"))
feats = ["p1", "rank", "m", "top1", "q", "s_tok_eq", "s_fold_eq", "s_exact", "o_tok_eq", "n_s_tok_eq", "n_s_fold_eq", "n_o_tok_eq",
         "e_n_other", "e_noaddr_other", "e_s2_other", "src"]
p = p.with_columns((pl.col("ti").hash(7) % 2).alias("half"))
tr, va = p.filter(pl.col("half") == 0), p.filter(pl.col("half") == 1)
d = xgb.DMatrix(tr.select(feats).to_numpy(), label=tr["label"].to_numpy())
m = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eta": 0.1, "subsample": 0.8, "eval_metric": "logloss", "nthread": 12}, d, 300)
va = va.with_columns(pl.Series("pm", m.predict(xgb.DMatrix(va.select(feats).to_numpy()))))
best = va.sort("pm", descending=True).group_by("ti").agg(pl.col("pm").first(), pl.col("label").first().alias("right"), pl.col("top1").first())
amb = best.filter(pl.col("top1") < 0.9)
print("validation targets:", best.height, "ambiguous (top p1 < 0.9):", amb.height)
for thr in (0.95, 0.9, 0.85, 0.8, 0.7):
    s = amb.filter(pl.col("pm") >= thr)
    print(f"  pm >= {thr}: coverage {s.height / max(amb.height, 1):.3f} ({s.height}), precision {s['right'].mean() if s.height else float('nan'):.3f}")
imp = sorted(zip(feats, m.get_score(importance_type="gain").values()), key=lambda t: -t[1]) if False else m.get_score(importance_type="gain")
print("gain importance:", sorted(imp.items(), key=lambda t: -t[1])[:10])

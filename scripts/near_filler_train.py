"""Training: same-street pairs whose target appends a filler word (center, services, service, partners) after the legal
form, by house relation and gap: true rate, and for false pairs whether another entity owns the target (so the
target would be claimed elsewhere at test time) or nobody does (a decoy)."""
import sys

import polars as pl

sys.argv = sys.argv[:1] + ["output/final_ctxA/matching_results.tsv"]
exec(open("scripts/class_audit.py").read().split("sub = sys.argv[1]")[0])
sys.path.insert(0, "scripts")
from suffix_position import position  # noqa: E402

FILL = {"center", "services", "service", "partners"}
tr = frame("train")
s1id = R("dataset/train/train_source1.tsv", ["entity_id"]).with_row_index("si")
tgid = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").join(s1id, left_on="source1_entity_id", right_on="entity_id")
      .join(tgid, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))
owned = gt.select("ti").unique().with_columns(pl.lit(1).alias("owned"))
tr = (tr.join(gt.with_columns(pl.lit(1).alias("label")), on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0))
      .join(owned, on="ti", how="left").with_columns(pl.col("owned").fill_null(0)).filter(pl.col("nd").is_in(["add", "adddrop"])))
raw1 = R("dataset/train/train_source1.tsv", ["business_name"])["business_name"]
raw2 = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["business_name"]) for i in (2, 3)])["business_name"]
res = [position(a, b) for a, b in zip(raw1.gather(tr["si"].to_numpy()).to_list(), raw2.gather(tr["ti"].to_numpy()).to_list())]
tr = tr.with_columns(pl.Series("word", [r[0] for r in res]), pl.Series("pos", [r[1] for r in res]))
x = tr.filter(pl.col("word").is_in(list(FILL)) & (pl.col("pos") == "after_lf"))
x = x.with_columns(pl.when(pl.col("h") == "gap0").then(pl.lit("0")).when(pl.col("h") == "near").then(pl.lit("1-30")).otherwise(pl.col("h")).alias("hh"))
x = x.with_columns(pl.when(pl.col("label") == 1).then(pl.lit("true")).when(pl.col("owned") == 1).then(pl.lit("other_owner"))
                   .otherwise(pl.lit("no_owner")).alias("state"))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
t = x.group_by("hh", "nd", "state").len().pivot(on="state", index=["hh", "nd"], values="len").fill_null(0)
t = t.with_columns((pl.col("true") / (pl.col("true") + pl.col("no_owner"))).round(3).alias("true_if_free"),
                   (pl.col("true") / (pl.col("true") + pl.col("no_owner") + pl.col("other_owner"))).round(3).alias("true_all"))
print(t.sort("hh", "nd"))

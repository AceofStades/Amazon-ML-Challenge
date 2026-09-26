"""Measure the literal ndeg house-prefix correction without changing decisions."""

from __future__ import annotations

import json

import polars as pl

from france_audit_restore import eligible, original_conflicts


def main() -> None:
    rejected = pl.read_parquet("output/france_audit_rejected.parquet")
    short = pl.read_parquet("output/france_audit_shortaddr.parquet")
    columns = ["si", "ti", "p1", "s_name", "t_name", "s_address", "t_address",
               *[column for column in short.columns if column.startswith("g_")]]
    frame = pl.concat([rejected.select(columns), short.select(columns)]).unique(["si", "ti"])
    frame = original_conflicts(frame).rename({"original_conflict": "before"})
    # Polars uses Rust regex: capturing the digits is equivalent to stripping
    # r'\bndeg(?=\d)', which would require unsupported lookahead syntax.
    frame = frame.with_columns(pl.col("t_address").alias("t_address_original"),
        pl.col("t_address").str.replace_all(r"\bndeg(\d+)", r"${1}").alias("t_address"))
    frame = original_conflicts(frame).rename({"original_conflict": "after"})
    frame = frame.with_columns(eligible(frame).alias("safe"))
    changed = frame.filter(pl.col("before") & ~pl.col("after"))
    print(json.dumps({"newly_nonconflicting_pairs": changed.height,
                      "newly_nonconflicting_targets": changed["ti"].n_unique()}), flush=True)
    for prob in (0.01, 0.5, 0.8, 0.9, 0.99, 0.999):
        for safe in (False, True):
            group = changed.filter((pl.col("p1") >= prob) & (pl.col("safe") == safe))
            print(json.dumps({"p1_min": prob, "safe": safe, "pairs": group.height,
                              "targets": group["ti"].n_unique(),
                              "exact": int(group["g_exact"].sum()), "drop": int(group["g_drop"].sum()),
                              "filler": int(group["g_swap_filler"].sum()),
                              "rare": int(group["g_swap_rare"].sum()),
                              "common": int(group["g_swap_common"].sum())}), flush=True)
    changed.write_parquet("output/france_audit_ndeg_recoveries.parquet")
    train = pl.read_parquet("output/france_audit_train_parser.parquet")
    print(json.dumps({"train_ndeg_targets": train.filter(
        pl.col("t_address").str.contains(r"\bndeg\d+")).height}), flush=True)
    train = original_conflicts(train.filter(eligible(train))).filter(pl.col("original_conflict"))
    print(json.dumps({"train_original_conflict_glued_number_targets": train.filter(
        pl.col("t_address").str.contains(r"\b[a-z]+\d+")).height}), flush=True)
    for prob in (0.5, 0.8, 0.9, 0.99):
        group = train.filter((pl.col("p1") >= prob) & pl.col("address_repaired_eq"))
        print(json.dumps({"train_p1_min": prob, "train_signature": "exact_address_original_conflict",
                          "pairs": group.height, "precision": group["label"].mean(),
                          "countries": group.group_by("country").agg(pl.len(), pl.col("label").mean()).rows()}),
              flush=True)


if __name__ == "__main__":
    main()

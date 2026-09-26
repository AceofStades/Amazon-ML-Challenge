"""Validate narrow address-parser recovery rules and draft France additions.

The audit never overwrites a baseline. Address equality ignores token order and
repairs only the literal French number prefix ``ndeg``. The second signature
requires identical nonnumeric address tokens and no new target numbers.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import polars as pl

from amzn.france_rule import typo_close

SAFE = ["g_exact", "g_drop", "g_typo1", "g_swap_filler", "g_add"]


def read(path: str, columns: list[str] | None = None) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       empty_string_is_null=False, columns=columns)


def signatures(frame: pl.DataFrame) -> pl.DataFrame:
    for side in ("s", "t"):
        frame = frame.with_columns(pl.col(f"{side}_address").str.replace_all(
            r"\bndeg(\d+)", r"${1}").alias(f"{side}_fixed"))
        frame = frame.with_columns(
            pl.col(f"{side}_fixed").str.split(" ").list.unique().list.sort().alias(f"{side}_tokens"),
            pl.col(f"{side}_fixed").str.extract_all(r"\d+").list.unique().list.sort().alias(f"{side}_nums"),
            pl.col(f"{side}_fixed").str.replace_all(r"\b\d+\b", "").str.replace_all(
                r"\b(frhdf|frpdl|frnaq|ndeg|no)\b", "").str.split(" ").list.eval(
                    pl.element().filter(pl.element() != "")).list.unique().list.sort().alias(f"{side}_words"))
    return frame.with_columns(
        (pl.col("s_tokens") == pl.col("t_tokens")).alias("address_repaired_eq"),
        ((pl.col("s_words") == pl.col("t_words"))
         & (pl.col("t_nums").list.set_difference(pl.col("s_nums")).list.len() == 0))
        .alias("address_number_subset"))


def eligible(frame: pl.DataFrame) -> pl.Expr:
    # One-token made-up brands also resemble rare-word substitutions. They are
    # decoy-heavy in France and cannot qualify as a spelling repair.
    safe = pl.any_horizontal(pl.col(column) == 1 for column in SAFE)
    rare = ((pl.col("g_swap_rare") == 1) & (pl.col("g_brand1") != 1)
            & (pl.col("g_s_ntok") > pl.col("g_n_miss")))
    return safe | rare


def original_conflicts(frame: pl.DataFrame) -> pl.DataFrame:
    """Replay the original pruning rule, including its first-digit source house."""
    standalone = re.compile(r"(?:^|[^a-z0-9])(\d+)")

    def conflict(source: str, target: str) -> bool:
        source_number = re.search(r"\d+", source)
        if source_number is None:
            return False
        house = source_number.group(0).lstrip("0") or "0"
        numbers = [number.lstrip("0") or "0" for number in standalone.findall(target)]
        return bool(numbers) and not any(typo_close(house, number) for number in numbers)

    return frame.with_columns(pl.Series("original_conflict", [conflict(source, target)
        for source, target in zip(frame["s_address"], frame["t_address"])]))


def report_train(frame: pl.DataFrame) -> None:
    frame = original_conflicts(frame.filter(eligible(frame)))
    for prob in (0.5, 0.8, 0.9, 0.99, 0.999):
        for signature in ("address_repaired_eq", "address_number_subset"):
            for dropped_only in (False, True):
                group = frame.filter((pl.col("p1") >= prob) & pl.col(signature)
                                     & (pl.col("s_address") != "") & (pl.col("t_address") != ""))
                if dropped_only:
                    group = group.filter(pl.col("original_conflict"))
                unique = group.group_by("ti").len().filter(pl.col("len") == 1)
                group = group.join(unique.select("ti"), on="ti", how="semi")
                print(json.dumps({"split": "train", "p1_min": prob, "signature": signature,
                                  "original_conflict_only": dropped_only,
                                  "pairs": group.height, "precision": group["label"].mean(),
                                  "countries": group.group_by("country").agg(
                                      pl.len(), pl.col("label").mean()).rows()}), flush=True)


def train() -> None:
    gen = pl.read_parquet("cache/gen/train.parquet")
    gen = gen.filter(eligible(gen))
    frames = []
    for path in sorted(glob.glob("output/v7/train_p1/*.parquet")):
        frame = (pl.scan_parquet(path).filter(pl.col("p1") >= 0.5)
                 .join(gen.lazy(), on=["si", "ti"], how="inner").collect())
        frames.append(frame)
    pairs = pl.concat(frames)
    print(f"TRAIN safe pairs p1>=.5: {pairs.height:,}", flush=True)
    s_ids = pl.read_parquet("cache/embeddings/train_source1_ids.parquet").with_row_index("si")
    t_ids = pl.concat([pl.read_parquet(f"cache/embeddings/train_source{i}_ids.parquet")
                      for i in (2, 3)]).with_row_index("ti")
    focus_s = pairs.select("si").unique()
    focus_t = pairs.select("ti").unique()
    focus_source_ids = s_ids.join(focus_s, on="si", how="semi")
    truth = (read("dataset/train/train_ground_truth.tsv")
             .join(focus_source_ids, left_on="source1_entity_id", right_on="entity_id")
             .filter(pl.col("matched_entity_ids") != "")
             .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .join(t_ids.join(focus_t, on="ti", how="semi"), left_on="matched_entity_ids", right_on="entity_id")
             .select("si", "ti").with_columns(pl.lit(1).alias("label")))
    pairs = pairs.join(truth, on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0))
    source = (pl.scan_csv("cache/train_source1.tsv", separator="\t", quote_char=None,
                          infer_schema=False, empty_string_is_null=False)
              .select(pl.col("address").alias("s_address"), "country").with_row_index("si")
              .join(focus_s.lazy(), on="si", how="semi").collect())
    target = (pl.concat([pl.scan_csv(f"cache/train_source{i}.tsv", separator="\t", quote_char=None,
                                    infer_schema=False, empty_string_is_null=False)
                         .select(pl.col("address").alias("t_address")) for i in (2, 3)])
              .with_row_index("ti").join(focus_t.lazy(), on="ti", how="semi").collect())
    frame = signatures(pairs.join(source, on="si").join(target, on="ti"))
    frame.write_parquet("output/france_audit_train_parser.parquet")
    report_train(frame)


def test(args: argparse.Namespace) -> None:
    frame = pl.read_parquet(args.candidates)
    frame = frame.filter(eligible(frame))
    frame = signatures(frame)
    for prob in (0.5, 0.8, 0.9, 0.99, 0.999):
        for signature in ("address_repaired_eq", "address_number_subset"):
            group = frame.filter((pl.col("p1") >= prob) & pl.col(signature))
            unique = group.group_by("ti").len().filter(pl.col("len") == 1)
            group = group.join(unique.select("ti"), on="ti", how="semi")
            print(json.dumps({"split": "test", "p1_min": prob, "signature": signature,
                              "pairs": group.height, "classes": group.group_by("class").len().rows()}), flush=True)
    choices = frame.filter((pl.col("p1") >= args.min_p) & pl.col(args.signature))
    unique = choices.group_by("ti").len().filter(pl.col("len") == 1)
    choices = choices.join(unique.select("ti"), on="ti", how="semi").select("si", "ti", "p1", "class",
        "s_name", "t_name", "s_address", "t_address")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    choices.write_parquet(output)
    print(f"Wrote {choices.height:,} proposed additions to {output}", flush=True)
    if args.matching_output:
        baseline = read(args.baseline)
        s_ids = pl.read_parquet("cache/embeddings/test_source1_ids.parquet").with_row_index("si")
        t_ids = pl.concat([pl.read_parquet(f"cache/embeddings/test_source{i}_ids.parquet")
                          for i in (2, 3)]).with_row_index("ti")
        additions = (choices.join(s_ids.select("si", pl.col("entity_id").alias("source1_entity_id")), on="si")
                     .join(t_ids.select("ti", pl.col("entity_id").alias("matched_entity_ids")), on="ti")
                     .select("source1_entity_id", "matched_entity_ids"))
        accepted = (baseline.filter(pl.col("matched_entity_ids") != "")
                    .with_columns(pl.col("matched_entity_ids").str.split(","))
                    .explode("matched_entity_ids"))
        additions = additions.join(accepted.select("matched_entity_ids"), on="matched_entity_ids", how="anti")
        context = (additions.join(baseline.with_columns(pl.when(pl.col("matched_entity_ids") == "").then(0)
                   .otherwise(pl.col("matched_entity_ids").str.split(",").list.len()).alias("accepted_count"))
                   .select("source1_entity_id", "accepted_count"), on="source1_entity_id"))
        print(json.dumps({"added_links": additions.height,
                          "added_owners": additions["source1_entity_id"].n_unique(),
                          "empty_owners": context.filter(pl.col("accepted_count") == 0)["source1_entity_id"].n_unique(),
                          "links_to_empty_owners": context.filter(pl.col("accepted_count") == 0).height}), flush=True)
        lists = pl.concat([accepted, additions]).group_by("source1_entity_id").agg(
            pl.col("matched_entity_ids").unique().sort().str.join(","))
        combined = (baseline.select("source1_entity_id").join(lists, on="source1_entity_id", how="left")
                    .with_columns(pl.col("matched_entity_ids").fill_null("")))
        matching_output = Path(args.matching_output)
        matching_output.parent.mkdir(parents=True, exist_ok=True)
        combined.write_csv(matching_output, separator="\t", quote_style="never")
        print(f"Wrote {matching_output}: {combined.height:,} rows, {additions.height:,} added links", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "report", "test"])
    parser.add_argument("--candidates", default="output/france_audit_rejected.parquet")
    parser.add_argument("--min-p", type=float, default=0.9)
    parser.add_argument("--signature", choices=["address_repaired_eq", "address_number_subset"],
                        default="address_repaired_eq")
    parser.add_argument("--output", default="output/france_audit_parser_additions.parquet")
    parser.add_argument("--baseline", default="output/sub_auto2_frsafe/matching_results.tsv")
    parser.add_argument("--matching-output", help="Optional new matching_results.tsv; baseline is never overwritten")
    args = parser.parse_args()
    if args.mode == "train":
        train()
    elif args.mode == "report":
        report_train(pl.read_parquet("output/france_audit_train_parser.parquet"))
    else:
        test(args)


if __name__ == "__main__":
    main()

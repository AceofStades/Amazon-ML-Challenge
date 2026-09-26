"""Recall rule for countries without labels: add back high-precision noise patterns.

Calibrated decisions for an unlabelled country (France) drop mid-score pairs, and
the raw stage-1 decisions add them back indiscriminately (about half are decoys:
generic words swapped at the same address). This keeps the calibrated decisions
and adds a raw-only pair only when the target sits at the Source 1 address and
its name follows a noise pattern that decoys do not use: an acronym, a web
domain or handle, a DBA/f.k.a. alias, a near-typo of the name, or the Source 1
name plus extra words. Targets already assigned to another entity are skipped.
"""

from __future__ import annotations

import argparse
import glob
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import polars as pl

ALIAS = re.compile(r"\b(dba|d/b/a|d\.b\.a|f/k/a|fka|formerly|aka|a/k/a|nee|née|doing business|trading as|t/a)\b")
DOMAIN = re.compile(r"\.(com|fr|net|org)\b|^www\.|^#")


def read(path: str) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)


def pairs(path: str) -> pl.DataFrame:
    return (read(path).filter(pl.col("matched_entity_ids") != "")
            .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids"))


def kind(name: str | None) -> str:
    text = (name or "").strip()
    low = text.lower()
    if DOMAIN.search(low):
        return "domain"
    if ALIAS.search(low):
        return "alias"
    if re.fullmatch(r"[A-Z0-9]{1,4}", text):
        return "acronym"
    return "other"


STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "and", "the", "of", "&"}


LEGAL = {"sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "scop", "gie", "selarl"}


def fold(text: str | None) -> str:
    """Lower case without accents: 'École' and 'ÉPL' keep their initial E."""
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()


def initials(name: str) -> set[str]:
    """Initials of a name, with and without articles and legal forms, reading hyphenated
    words either as one word ('Saint-Nazaire' gives S) or as several (S, N)."""
    low = fold(name)
    out = set()
    for pattern in (r"[a-z0-9]+", r"[a-z0-9]+(?:-[a-z0-9]+)*"):
        words = re.findall(pattern, low)
        for drop_legal in (False, True):
            for stop in (False, True):
                kept = [w for w in words if not (stop and w in STOP) and not (drop_legal and w in LEGAL)]
                if kept:
                    out.add("".join(w[0] for w in kept))
    return out


def consistent(kind_: str, target: str, source: str, core: str) -> bool:
    """An acronym must spell the Source 1 initials; a domain must carry a Source 1 word or its initials.
    Decoy businesses at the same address have their own names, so their acronyms and domains differ."""
    t = re.sub(r"[^a-z0-9]", "", fold(target))
    if kind_ == "alias":
        return True
    variants = initials(source)
    if kind_ == "acronym":
        return any(t and (init.startswith(t) or t.startswith(init[:2])) and init[:1] == t[:1] for init in variants)
    words = [w for w in fold(core).split() if len(w) >= 4]
    return any(w in t for w in words) or any(len(i) >= 2 and t.startswith(i) for i in variants)


def typo_close(a: str, b: str) -> bool:
    """House numbers that differ like a typo: same digits after zero padding, or equal length with one digit changed."""
    a, b = a.lstrip("0") or "0", b.lstrip("0") or "0"
    if a == b:
        return True
    if len(a) == len(b):
        return len(a) >= 2 and sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = sorted((a, b), key=len)
    # One digit dropped (a frequent noise on true records): 601 vs 60, 15911 vs 5911.
    return len(long_) - len(short) == 1 and len(short) >= 2 and any(
        long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def drop_house_conflicts(base: pl.DataFrame, s1: pl.DataFrame, targets: pl.DataFrame, country: str) -> pl.DataFrame:
    """For the unlabelled country, drop accepted pairs whose house numbers conflict unless typo-close:
    its decoys repeat the business name at another number on the same street."""
    def standalone(frame: pl.DataFrame) -> pl.Series:
        # Digits glued to letters ('cite1', 'espl1', 'chem1') are street-type artifacts, not house numbers.
        return frame["address"].fill_null("").str.extract_all(r"(?:^|[^a-z0-9])(\d+)").list.eval(
            pl.element().str.replace_all(r"[^0-9]", "").str.strip_chars_start("0").replace("", "0"))
    s_norm = read("cache/test_source1.tsv").select("house")
    t_norm = pl.concat([read(f"cache/test_source{i}.tsv").select("address") for i in (2, 3)])
    t_numbers = standalone(t_norm)
    rows = (base.join(s1.select("entity_id", "si", "country"), left_on="source1_entity_id", right_on="entity_id")
            .join(targets.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id"))
    focus = rows.filter(pl.col("country") == country)
    hs = s_norm["house"].fill_null("").to_numpy()[focus["si"].to_numpy()]
    nums = t_numbers.gather(focus["ti"].to_numpy()).to_list()
    conflict = [bool(h) and bool(n) and not any(typo_close(h, x) for x in n) for h, n in zip(hs, nums)]
    dropped = focus.filter(pl.Series(conflict))
    print(f"{country}: dropping {dropped.height:,} accepted pairs with conflicting house numbers", flush=True)
    return base.join(dropped.select("source1_entity_id", "matched_entity_ids"),
                     on=["source1_entity_id", "matched_entity_ids"], how="anti")


FILLERS = {"center", "services", "service", "partners", "france"}


def subsequence(short: str, long_: str) -> bool:
    it = iter(long_)
    return all(ch in it for ch in short)


def common_swap(s_core: str, t_core: str, df: Counter) -> bool:
    """One Source 1 word replaced by another common vocabulary word: the decoy perturbation
    (true records use a filler such as 'services', an abbreviation or a typo instead)."""
    s, t = (s_core or "").split(), (t_core or "").split()
    ss, ts = set(s), set(t)
    miss = [w for w in s if w not in ts]
    extra = [w for w in t if w not in ss]
    if len(miss) != 1 or len(extra) != 1:
        return False
    a, b = miss[0], extra[0]
    if b in FILLERS or df.get(b, 0) < 20 or SequenceMatcher(None, a, b).ratio() >= 0.75:
        return False
    return not (subsequence(b, a) or subsequence(a, b))


def drop_swaps(base: pl.DataFrame, s1: pl.DataFrame, targets: pl.DataFrame, country: str) -> pl.DataFrame:
    """Drop accepted pairs of the unlabelled country whose name differs by one common-word swap.
    In training such same-address pairs are true only 45% of the time (test has twice the decoys)."""
    s_norm = read("cache/test_source1.tsv").select("entity_id", "name_core").join(
        s1.select("entity_id", "country"), on="entity_id")
    t_norm = pl.concat([read(f"cache/test_source{i}.tsv").select("entity_id", "name_core") for i in (2, 3)])
    df: Counter = Counter()
    for core in s_norm.filter(pl.col("country") == country)["name_core"]:
        df.update(set(core.split()))
    rows = (base.join(s_norm.rename({"name_core": "s_core"}), left_on="source1_entity_id", right_on="entity_id")
            .filter(pl.col("country") == country)
            .join(t_norm.rename({"name_core": "t_core"}), left_on="matched_entity_ids", right_on="entity_id"))
    flag = [common_swap(a, b, df) for a, b in zip(rows["s_core"].to_list(), rows["t_core"].to_list())]
    dropped = rows.filter(pl.Series(flag, dtype=pl.Boolean))
    print(f"{country}: dropping {dropped.height:,} accepted pairs that swap one word for another common word", flush=True)
    return base.join(dropped.select("source1_entity_id", "matched_entity_ids"),
                     on=["source1_entity_id", "matched_entity_ids"], how="anti")


def initials_pairs(s1: pl.DataFrame, targets: pl.DataFrame, country: str, taken: pl.DataFrame) -> pl.DataFrame:
    """Every scored candidate pair of the country whose target is an acronym or web domain that spells the
    Source 1 initials or words, on the Source 1 street without a house conflict, for targets not yet assigned.
    Training pairs of this kind at the same address are 99.9% true; stage 1 scores the short French
    acronyms low (median p1 0.35) because such names recur across many targets."""
    gen = pl.read_parquet("cache/gen/test.parquet", columns=["si", "ti", "g_acronym", "g_domain", "g_consistent"])
    local = s1.filter(pl.col("country") == country).select("si")
    cand = gen.join(local, on="si", how="semi").filter((pl.col("g_acronym") == 1) | (pl.col("g_domain") == 1))
    s_core = read("cache/test_source1.tsv").select("entity_id", "name_core")
    cand = (cand.join(s1.select("si", "entity_id", pl.col("business_name").alias("s_name")), on="si")
            .join(s_core, on="entity_id").join(targets.select("ti", pl.col("business_name").alias("t_name")), on="ti"))
    cand = cand.filter(pl.Series([consistent("acronym" if a == 1 else "domain", t, s_, c) for a, t, s_, c in zip(
        cand["g_acronym"].to_list(), cand["t_name"].to_list(), cand["s_name"].to_list(), cand["name_core"].to_list())],
        dtype=pl.Boolean)).select("si", "ti", "g_acronym")
    feats = pl.concat([pl.read_parquet(path, columns=["si", "ti", "house_conflict", "address_idf_cos"])
                       .join(cand.select("si", "ti"), on=["si", "ti"], how="semi")
                       for path in sorted(glob.glob("cache/v3/test/part*.parquet"))])
    cand = (cand.join(feats, on=["si", "ti"])
            .filter((pl.col("house_conflict") != 1) & (pl.col("address_idf_cos") >= 0.6))
            .join(targets.select("ti", pl.col("entity_id").alias("matched_entity_ids")), on="ti")
            .join(taken, on="matched_entity_ids", how="anti")
            .sort("address_idf_cos", descending=True).unique("ti", keep="first")
            .join(s1.select("si", pl.col("entity_id").alias("source1_entity_id")), on="si"))
    print(f"{country}: adding {cand.height:,} acronym/domain pairs on the Source 1 street "
          f"({int(cand['g_acronym'].sum()):,} acronyms)", flush=True)
    return cand.select("source1_entity_id", "matched_entity_ids")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated", required=True, help="matching_results.tsv with calibrated decisions")
    parser.add_argument("--raw", required=True, help="matching_results.tsv with raw decisions for the country")
    parser.add_argument("--country", default="France")
    parser.add_argument("--output", required=True)
    parser.add_argument("--typo-jw", type=float, default=0.92)
    parser.add_argument("--drop-conflicts", action="store_true",
                        help="Drop accepted pairs whose house numbers conflict (not typo-close)")
    parser.add_argument("--drop-swaps", action="store_true",
                        help="Drop accepted pairs whose name swaps one word for another common word")
    parser.add_argument("--add-initials", action="store_true",
                        help="Add acronym/domain pairs on the Source 1 street from all scored candidates")
    args = parser.parse_args()

    s1 = read("dataset/test/test_source1.tsv").with_row_index("si")
    targets = pl.concat([read(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti")
    norm_s = read("cache/test_source1.tsv").select("entity_id", pl.col("name_core").alias("s_core"))
    norm_t = pl.concat([read(f"cache/test_source{i}.tsv") for i in (2, 3)]).select(
        "entity_id", pl.col("name_core").alias("t_core"))
    calibrated, raw = pairs(args.calibrated), pairs(args.raw)
    gap = (raw.join(calibrated, on=["source1_entity_id", "matched_entity_ids"], how="anti")
           .join(s1.select("entity_id", "si", "country"), left_on="source1_entity_id", right_on="entity_id")
           .filter(pl.col("country") == args.country)
           .join(targets.select("entity_id", "ti", pl.col("business_name").alias("t_name")),
                 left_on="matched_entity_ids", right_on="entity_id")
           .join(norm_s, left_on="source1_entity_id", right_on="entity_id")
           .join(norm_t, left_on="matched_entity_ids", right_on="entity_id"))
    features = pl.concat([
        pl.read_parquet(p, columns=["si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_compact_jw"])
        .join(gap.select("si", "ti"), on=["si", "ti"], how="semi")
        for p in sorted(glob.glob("cache/v3/test/part*.parquet"))])
    gap = gap.join(features, on=["si", "ti"], how="left").with_columns(
        pl.col("t_name").map_elements(kind, return_dtype=pl.String).alias("kind"),
        ((pl.col("house_conflict") != 1)
         & ((pl.col("house_match") == 1) | (pl.col("address_idf_cos") > 0.8))).alias("same_address"),
        pl.struct("s_core", "t_core").map_elements(
            lambda r: bool(set((r["s_core"] or "").split()) and set((r["s_core"] or "").split())
                           <= set((r["t_core"] or "").split())), return_dtype=pl.Boolean).alias("contains"))
    s_names = s1.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("business_name").alias("s_name"))
    gap = gap.join(s_names, on="source1_entity_id", how="left").with_columns(
        pl.struct("kind", "t_name", "s_name", "s_core").map_elements(
            lambda r: consistent(r["kind"], r["t_name"], r["s_name"], r["s_core"]), return_dtype=pl.Boolean)
        .alias("consistent"))
    keep = gap.filter(pl.col("same_address") & (
        (pl.col("kind").is_in(["acronym", "domain", "alias"]) & pl.col("consistent"))
        | ((pl.col("kind") == "other") & ((pl.col("name_compact_jw") >= args.typo_jw) | pl.col("contains")))))
    taken = calibrated.select("matched_entity_ids").unique()
    keep = keep.join(taken, on="matched_entity_ids", how="anti").unique("matched_entity_ids", keep="first")
    print(f"{args.country} raw-only pairs {gap.height:,}; added back {keep.height:,}", flush=True)
    print(keep.group_by("kind").len().sort("len", descending=True).rows(), flush=True)

    base = calibrated.select("source1_entity_id", "matched_entity_ids")
    if args.drop_conflicts:
        base = drop_house_conflicts(base, s1, targets, args.country)
    if args.drop_swaps:
        base = drop_swaps(base, s1, targets, args.country)
    merged = pl.concat([base, keep.select("source1_entity_id", "matched_entity_ids")])
    if args.add_initials:
        taken = merged.select("matched_entity_ids").unique()
        merged = pl.concat([merged, initials_pairs(s1, targets, args.country, taken)])
    lists = merged.group_by("source1_entity_id").agg(
        pl.col("matched_entity_ids").unique().sort().str.join(","))
    out = s1.select(pl.col("entity_id").alias("source1_entity_id")).join(
        lists, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
    out.write_csv(args.output, separator="\t", quote_style="never")
    print(f"wrote {args.output}: {out.height:,} rows, {merged.height:,} matches", flush=True)


if __name__ == "__main__":
    main()

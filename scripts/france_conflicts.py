"""France house-conflict rule v2, based on how the generator builds decoys.

Training truth for same-street pairs with similar names (name IDF cosine >= 0.9) whose house numbers conflict:
- decoys 'a few doors away' change the name: target adds words, gap 1-50 -> 2.7-2.9% true;
- true records whose number was moved keep the name: identical raw names (accents, case and dotted letters folded)
  are 99.7% true when the gap is over 50 and 91% true when it is 1-50; dropped words far away 97.9%.
France keeps almost none of the identical-name conflicts (its calibration and --drop-conflicts reject them) but lets
through some near conflicts that add words (v7 normalization hides 'groupe'/'developpement' and legal forms).

The rule, applied to France only:
  restore  identical-name conflicts with a gap over 30 whose target no entity holds, one entity per target (highest
           stage-1 p1, then smallest gap). France decoys sit within 30 numbers (the added-word ones: 99.7% within 30)
           and are about 4x denser than in training, so near identical-name conflicts would be only ~77% true and
           far dropped-word ones about half (estimated from the training ratio of identical-name to added-word decoys);
  drop     accepted near (gap 1-50) conflicts whose target adds words.
Usage: python scripts/france_conflicts.py --base A/matching_results.tsv --output B/matching_results.tsv [--min-gap 31]
"""

import argparse
import glob
import re
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def tokens(s: str) -> set[str]:
    out, run = [], ""
    for w in re.findall(r"[a-z0-9]+", fold(s)):
        if len(w) == 1 and w.isalpha():  # merge dotted letters: s.a.r.l -> sarl
            run += w
            continue
        if run:
            out.append(run)
            run = ""
        out.append(w)
    if run:
        out.append(run)
    return set(out)


def diff(sn: str, tn: str) -> str:
    a, b = tokens(sn), tokens(tn)
    return ("add" if b - a else "") + ("drop" if a - b else "") or "same"


def gap(h: str, ta: str):
    try:
        hv = int(h)
    except ValueError:
        return None
    nums = [int(n) for n in re.findall(r"\d+", ta) if len(n) <= 5]
    return min(abs(n - hv) for n in nums) if nums else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-gap", type=int, default=31, help="smallest house-number gap restored")
    ap.add_argument("--no-drop", action="store_true", help="do not drop near added-word conflicts")
    args = ap.parse_args()

    s1 = R("dataset/test/test_source1.tsv").with_row_index("si")
    tg = pl.concat([R(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti")
    fr = s1.filter(pl.col("country") == "France").select("si")
    cols = ["si", "ti", "house_conflict", "address_idf_cos", "name_idf_cos"]
    x = pl.concat([pl.read_parquet(f, columns=cols).join(fr, on="si", how="semi")
                   .filter((pl.col("house_conflict") == 1) & (pl.col("address_idf_cos") >= 0.6) & (pl.col("name_idf_cos") >= 0.9))
                   for f in sorted(glob.glob("cache/v3/test/part*.parquet"))])
    house = R("cache/test_source1.tsv", ["house"])["house"].fill_null("")
    addr = pl.concat([R(f"cache/test_source{i}.tsv", ["address"]) for i in (2, 3)])["address"].fill_null("")
    x = x.with_columns(s1["business_name"].gather(x["si"].to_numpy()).alias("sn"), tg["business_name"].gather(x["ti"].to_numpy()).alias("tn"),
                       house.gather(x["si"].to_numpy()).alias("hs"), addr.gather(x["ti"].to_numpy()).alias("ta"))
    x = x.with_columns(pl.struct("hs", "ta").map_elements(lambda s: gap(s["hs"], s["ta"]), return_dtype=pl.Int64).alias("gap"),
                       pl.struct("sn", "tn").map_elements(lambda s: diff(s["sn"], s["tn"]), return_dtype=pl.String).alias("nd"))
    p1 = pl.concat([pl.read_parquet(f).join(x.select("si", "ti"), on=["si", "ti"], how="semi") for f in sorted(glob.glob("output/v7/test_p1/*.parquet"))])
    x = x.join(p1, on=["si", "ti"], how="left").with_columns(pl.col("p1").fill_null(0.0))

    base = R(args.base)
    links = (base.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids").join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
             .join(tg.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))
    held = links.select("ti").unique()
    accepted = x.join(links, on=["si", "ti"], how="semi")

    restore = x.join(held, on="ti", how="anti").filter((pl.col("nd") == "same") & (pl.col("gap") >= args.min_gap))
    restore = restore.sort(["ti", "p1", "gap"], descending=[False, True, False]).unique("ti", keep="first", maintain_order=True)
    drop = accepted.filter((pl.col("gap") >= 1) & (pl.col("gap") <= 50) & pl.col("nd").is_in(["add", "adddrop"]))
    if args.no_drop:
        drop = drop.head(0)
    print(f"restore {restore.height:,} pairs:", restore.group_by((pl.col("gap") > 50).alias("far"), "nd").len().sort("far", "nd").rows())
    print(f"drop {drop.height:,} pairs")

    new = pl.concat([links.join(drop.select("si", "ti"), on=["si", "ti"], how="anti"), restore.select("si", "ti")])
    new = new.join(s1.select("si", pl.col("entity_id").alias("source1_entity_id")), on="si").join(
        tg.select("ti", pl.col("entity_id").alias("t")), on="ti")
    grouped = new.group_by("source1_entity_id").agg(pl.col("t").sort().str.join(",").alias("matched_entity_ids"))
    out = s1.select(pl.col("entity_id").alias("source1_entity_id")).join(grouped, on="source1_entity_id", how="left").with_columns(
        pl.col("matched_entity_ids").fill_null(""))
    out.write_csv(args.output, separator="\t", quote_style="never")
    print(f"wrote {args.output}: {out.height:,} rows, {new.height:,} matches (base {links.height:,})")
    for r in restore.sample(min(10, restore.height), seed=5).iter_rows(named=True):
        print(f"  + gap={r['gap']} {r['nd']} p1={r['p1']:.2f} | {r['sn'][:30]:30} {r['hs']:>5} || {r['tn'][:30]:30} {r['ta'][:36]}")
    for r in drop.sample(min(6, drop.height), seed=5).iter_rows(named=True):
        print(f"  - gap={r['gap']} {r['nd']} p1={r['p1']:.2f} | {r['sn'][:30]:30} {r['hs']:>5} || {r['tn'][:30]:30} {r['ta'][:36]}")


if __name__ == "__main__":
    main()

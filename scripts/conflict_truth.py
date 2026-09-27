"""Training truth for same-name, same-street, different-house pairs (the France --drop-conflicts class), split by how
the house numbers relate: one number a digit-typo of the other, last digit off by 1-2 or 3-10, or unrelated."""
import glob

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
cols = ["source1_entity_id", "candidate_entity_id", "si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_idf_cos", "name_exact", "name_core_jw"]
x = pl.concat([pl.read_parquet(f, columns=cols).filter((pl.col("house_conflict") == 1) & (pl.col("address_idf_cos") >= 0.6) & (pl.col("name_idf_cos") >= 0.95))
               for f in sorted(glob.glob("cache/v3/train/part*.parquet"))])
gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").select("source1_entity_id", pl.col("matched_entity_ids").alias("candidate_entity_id")).with_columns(pl.lit(1).alias("label")))
x = x.join(gt, on=["source1_entity_id", "candidate_entity_id"], how="left").with_columns(pl.col("label").fill_null(0))
hs = R("cache/train_source1.tsv", ["house"])["house"].fill_null("")
addr = pl.concat([R(f"cache/train_source{i}.tsv", ["address"]) for i in (2, 3)])["address"].fill_null("")
x = x.with_columns(hs.gather(x["si"].to_numpy()).alias("hs"), addr.gather(x["ti"].to_numpy()).alias("ta"))


def relation(h, ta):
    import re
    nums = [n.lstrip("0") or "0" for n in re.findall(r"\d+", ta)]
    h = (h or "").lstrip("0") or "0"
    if not nums or not h:
        return "none"
    best = "unrelated"
    for n in nums:
        if len(n) == len(h) and sum(a != b for a, b in zip(n, h)) == 1:
            if n[:-1] == h[:-1]:
                d = abs(int(n) - int(h))
                r = "last_1_2" if d <= 2 else ("last_3_10" if d <= 10 else "last_gt10")
            else:
                r = "inner_digit"
        elif abs(len(n) - len(h)) == 1 and (n in h or h in n or any(h[:i] + h[i + 1:] == n for i in range(len(h))) or any(n[:i] + n[i + 1:] == h for i in range(len(n)))):
            r = "dropped_digit"
        elif n == "1":
            r = "target_1"
        else:
            continue
        order = ["last_1_2", "inner_digit", "dropped_digit", "last_3_10", "last_gt10", "target_1"]
        if best == "unrelated" or order.index(r) < order.index(best):
            best = r
    return best


x = x.with_columns(pl.struct("hs", "ta").map_elements(lambda s: relation(s["hs"], s["ta"]), return_dtype=pl.String).alias("rel"))
x.select("si", "ti", "label", "rel", "name_exact").write_parquet("cache/conflict_truth_train.parquet")
print(x.group_by("rel", "name_exact").agg(pl.len().alias("n"), pl.col("label").mean().round(3).alias("true_rate")).sort("rel", "name_exact"))

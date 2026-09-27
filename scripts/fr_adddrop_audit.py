"""France same-house pairs whose name both adds and drops words (swaps and rewrites): which added and dropped words the
unaccepted ones carry, against the accepted ones. Reads cache/class_audit_fr.parquet (written by class_audit.py)."""
import re
import sys
import unicodedata
from collections import Counter

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def tokens(s):
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
    return set(out)


h, nd = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("same", "adddrop")
x = pl.read_parquet("cache/class_audit_fr.parquet").filter((pl.col("h") == h) & (pl.col("nd") == nd))
raw1 = R("dataset/test/test_source1.tsv", ["business_name"])["business_name"]
raw2 = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["business_name"]) for i in (2, 3)])["business_name"]
x = x.with_columns(raw1.gather(x["si"].to_numpy()).alias("sn"), raw2.gather(x["ti"].to_numpy()).alias("tn"))
rows = []
for si, ti, acc, sn, tn in x.select("si", "ti", "acc", "sn", "tn").iter_rows():
    a, b = tokens(sn), tokens(tn)
    rows.append((acc, " ".join(sorted(b - a)), " ".join(sorted(a - b))))
d = pl.DataFrame(rows, schema=["acc", "added", "dropped"], orient="row")
print(f"{h}/{nd}: {d.height:,} pairs, accepted {d['acc'].mean():.3f}")
for col in ("added", "dropped"):
    t = d.group_by(col).agg(pl.len().alias("n"), pl.col("acc").mean().alias("acc")).sort("n", descending=True)
    print(f"top {col}:"); print(t.head(int(sys.argv[3]) if len(sys.argv) > 3 else 25))
held = x.filter(pl.col("acc") == 1).select("ti")
free = x.filter(pl.col("acc") == 0).join(held, on="ti", how="anti")
print(f"unaccepted pairs whose target no accepted same-house pair holds: {free.height:,}")

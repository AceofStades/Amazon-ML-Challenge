"""Label-free precision check from Source 2/3 balance. The generator spreads an entity's records over both sources, so
when an entity's other links all sit in one source, a true link is in the other source ~74% of the time (training
truth, other links = 2), while a decoy's source is independent (~50%). DBA/alias records are the exception: true but
generated independently of the balance (fill ~0.50 in training). For each record kind prints the "fill" rate of
links whose remaining links (>= 2) are one-source; pi ~ (fill - 0.5) / (ref - 0.5) estimates the true share.
Usage: python scripts/agent_source_balance.py SUBMISSION"""
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, columns=c)
ALIAS = r"(?i)\b(dba|d/b/a|d\.b\.a\.?|aka|a\.k\.a\.?|a/k/a|fka|f\.k\.a\.?|f/k/a|formerly|nee|née|doing business as|trading as|known as)\b"


def links(path, split):
    tg = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["entity_id", "business_name", "business_address", "country"])
                    for i in (2, 3)]).rename({"entity_id": "ids"})
    m = R(path).rename(lambda c: "s1" if c.startswith("source1") else "ids").filter(pl.col("ids").is_not_null())
    return m.with_columns(pl.col("ids").str.split(",")).explode("ids").join(tg, on="ids")


def fill(m, label):
    m = m.with_columns(pl.when(pl.col("business_name").str.contains(ALIAS)).then(pl.lit("alias"))
                       .when(pl.col("business_name").str.contains(r"^[A-Z]{2,6}$")).then(pl.lit("acronym"))
                       .when(pl.col("business_name").str.contains(r"(?i)(\.(com|net|org|in|fr|co|biz|io)\b|^www|^@|^#[a-z])"))
                       .then(pl.lit("domain")).when(pl.col("business_address").is_null()).then(pl.lit("noaddr"))
                       .otherwise(pl.lit("other")).alias("kind"),
                       pl.col("ids").str.starts_with("S3").cast(pl.Int32).alias("s3"))
    e = m.group_by("s1").agg(pl.len().alias("k"), pl.col("s3").sum().alias("n3"))
    m = m.join(e, on="s1").filter(pl.col("k") >= 3).with_columns((pl.col("k") - 1).alias("kr"), (pl.col("n3") - pl.col("s3")).alias("n3r"))
    one = m.filter((pl.col("n3r") == 0) | (pl.col("n3r") == pl.col("kr"))).filter(pl.col("kr") == 2)
    one = one.with_columns(pl.when(pl.col("n3r") == 0).then(pl.col("s3") == 1).otherwise(pl.col("s3") == 0).alias("fills"))
    t = one.group_by("kind").agg(pl.len().alias("n"), pl.col("fills").mean().round(3).alias("fill")).sort("kind")
    print(f"{label:>14}: all {one['fills'].mean():.3f} (n={one.height}) | " + ", ".join(f"{k} {f} (n={n})" for k, n, f in t.iter_rows()))


if __name__ == "__main__":
    fill(links("dataset/train/train_ground_truth.tsv", "train"), "train truth")
    test = links(sys.argv[1], "test")
    for c in ("US", "India", "France"):
        fill(test.filter(pl.col("country") == c), f"test {c}")

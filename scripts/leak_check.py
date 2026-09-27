"""Leak check on training truth: do row positions or numeric ids of true targets relate to their Source 1 entity?
Spearman correlation over true pairs (all, and address-less only), plus id-difference and row-difference structure."""
import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s1 = R("dataset/train/train_source1.tsv", ["entity_id"]).with_row_index("srow")
t2 = R("dataset/train/train_source2.tsv", ["entity_id", "business_address"]).with_row_index("trow").with_columns(pl.lit(2).alias("src"))
t3 = R("dataset/train/train_source3.tsv", ["entity_id", "business_address"]).with_row_index("trow").with_columns(pl.lit(3).alias("src"))
tg = pl.concat([t2, t3])
gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").select(pl.col("source1_entity_id").alias("s"), pl.col("matched_entity_ids").alias("t")))
x = gt.join(s1, left_on="s", right_on="entity_id").join(tg, left_on="t", right_on="entity_id")
x = x.with_columns(pl.col("s").str.slice(3).cast(pl.Int64).alias("sid"), pl.col("t").str.slice(3).cast(pl.Int64).alias("tid"),
                   (pl.col("business_address") == "").alias("noaddr"))
print("true pairs:", x.height)
for name, f in (("all", x), ("noaddr", x.filter(pl.col("noaddr")))):
    for a, b in (("srow", "trow"), ("sid", "tid")):
        rho = f.select(pl.corr(a, b, method="spearman")).item()
        print(f"{name}: spearman({a},{b}) = {rho:.5f}")
# id arithmetic: same last k digits, id difference concentration
for k in (1, 2, 3):
    m = 10 ** k
    same = (x["sid"] % m == x["tid"] % m).mean()
    print(f"same last {k} digits: {same:.5f} (random {1 / m:.5f})")
d = (x["tid"] - x["sid"]).abs()
print("abs id difference quantiles:", [int(d.quantile(q)) for q in (0.001, 0.01, 0.1, 0.5)])
# per-entity: are an entity's targets closer in id / row to each other than random pairs?
g = x.group_by("s").agg(pl.col("tid").sort().diff().drop_nulls().alias("gaps"), pl.col("trow").sort().diff().drop_nulls().alias("rgaps"))
gaps = g.explode("gaps")["gaps"].drop_nulls()
print("median id gap between targets of one entity:", int(gaps.median()), "min 1% quantile:", int(gaps.quantile(0.01)))

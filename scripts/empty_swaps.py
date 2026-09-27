"""France same-house common-word swaps for entities the submission leaves empty versus all entities.
Decoy swaps land on the Source 1 house 0.42 times as often as on another house of the same street (training), so
same-house swaps beyond 0.42 x other-house swaps are true records. Compares that excess for empty entities with the rest."""
import sys

import polars as pl

sys.argv, argv = sys.argv[:1], sys.argv
exec(open("scripts/fr_adddrop_audit.py").read().split("h, nd = ")[0])
sub = argv[1] if len(argv) > 1 else "output/final_ctxA/matching_results.tsv"
x = pl.read_parquet("cache/class_audit_fr.parquet").filter(pl.col("nd") == "adddrop")
raw1 = R("dataset/test/test_source1.tsv", ["business_name"])["business_name"]
raw2 = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["business_name"]) for i in (2, 3)])["business_name"]
x = x.with_columns(raw1.gather(x["si"].to_numpy()).alias("sn"), raw2.gather(x["ti"].to_numpy()).alias("tn"))
common = {"club", "ecole", "amicale", "comite", "sportive", "centre", "amis", "union", "parents", "college", "primaire", "pharmacie",
          "maison", "fetes", "societe", "loisirs", "federation", "anciens", "sante", "maternelle", "medical", "association", "lycee",
          "elementaire", "cabinet", "boulangerie", "garage", "restaurant", "hotel", "cafe", "atelier", "studio", "agence", "institut"}
keep = []
for si, ti, sn, tn in x.select("si", "ti", "sn", "tn").iter_rows():
    a, b = tokens(sn), tokens(tn)
    add, drop = b - a, a - b
    keep.append(len(add) == 1 and len(drop) == 1 and next(iter(add)) in common and next(iter(drop)) in common)
x = x.filter(pl.Series(keep))
m = R(sub)
s1 = R("dataset/test/test_source1.tsv", ["entity_id"]).with_row_index("si")
empty = m.filter(pl.col("matched_entity_ids") == "").join(s1, left_on="source1_entity_id", right_on="entity_id").select("si").with_columns(pl.lit(1).alias("empty"))
x = x.join(empty, on="si", how="left").with_columns(pl.col("empty").fill_null(0), (pl.col("h") == "same").alias("same_house"))
t = x.group_by("empty", "same_house").agg(pl.len().alias("pairs"), pl.col("si").n_unique().alias("entities"), pl.col("acc").mean().alias("acc"))
print(t.sort("empty", "same_house"))
for e in (0, 1):
    s = x.filter((pl.col("empty") == e) & pl.col("same_house")).height
    o = x.filter((pl.col("empty") == e) & ~pl.col("same_house")).height
    print(f"empty={e}: same-house {s:,}, other-house {o:,}, implied true same-house {max(s - 0.42 * o, 0):,.0f} ({max(s - 0.42 * o, 0) / max(s, 1):.2f})")
x.write_parquet("cache/fr_common_swaps.parquet")

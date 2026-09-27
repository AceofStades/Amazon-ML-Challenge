"""Source-balance precision estimate for France accepted links by name-change subclass (which word was dropped or
added). When an entity's other links all sit in one source, a true link lands in the other source more often (training
reference per number of other links) than a decoy (0.5). pi = (fill - 0.5) / (ref - 0.5).
Usage: python scripts/fr_drop_balance.py SUBMISSION"""
import re
import sys
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
LF = {"sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei", "llc", "inc", "corp", "corporation", "ltd", "limited", "private", "pvt",
      "llp", "lp", "co", "company", "pc", "pllc", "plc", "incorporated"}
FILL = {"groupe", "developpement", "fils", "associes", "cie", "compagnie", "services", "france", "frs", "freres", "et",
        "center", "services", "service", "partners", "group", "holdings"}
TYPE = {"club", "ecole", "comite", "amicale", "association", "sportive", "centre", "amis", "union", "parents", "college", "primaire",
        "pharmacie", "maison", "fetes", "societe", "loisirs", "federation", "anciens", "sante", "maternelle", "medical", "lycee",
        "elementaire", "cabinet", "culturel", "culture", "sport", "sports", "jeunes", "conseil", "institut", "collectif", "foyer"}
STOP = {"de", "du", "des", "la", "le", "les", "d", "l", "s", "a", "the", "of", "and"}


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def toks(s):
    out, run = [], ""
    for w in re.findall(r"[a-z0-9]+", fold(s)):
        if len(w) == 1 and w.isalpha():
            run += w
            continue
        if run:
            out.append(run)
            run = ""
        out.append(w)
    if run:
        out.append(run)
    return set(out)


def subclass(sn, tn, city):
    a, b = toks(sn) - STOP, toks(tn) - STOP
    add, drop = b - a, a - b
    if not add and not drop:
        return "exact"
    def cls(ws):
        if ws <= LF:
            return "lf"
        if ws & TYPE:
            return "type"
        if ws & FILL:
            return "filler"
        if city and ws & city:
            return "city"
        return "other"
    if drop and not add:
        return "drop_" + cls(drop)
    if add and not drop:
        return "add_" + cls(add)
    return "swap_" + cls(drop) + ">" + cls(add)


def table(path, split, countries):
    s1 = R(f"dataset/{split}/{split}_source1.tsv", ["entity_id", "business_name", "business_address", "country"])
    tg = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["entity_id", "business_name"]) for i in (2, 3)])
    col = "matched_entity_ids"
    m = (R(path).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
         .join(s1.rename({"business_name": "sn", "business_address": "sa"}), left_on="source1_entity_id", right_on="entity_id")
         .filter(pl.col("country").is_in(countries))
         .join(tg.rename({"business_name": "tn"}), left_on=col, right_on="entity_id")
         .with_columns(pl.col(col).str.starts_with("S3").cast(pl.Int32).alias("s3")))
    e = m.group_by("source1_entity_id").agg(pl.len().alias("k"), pl.col("s3").sum().alias("n3"))
    m = m.join(e, on="source1_entity_id").filter(pl.col("k").is_in([3, 4]))
    m = m.with_columns((pl.col("k") - 1).alias("kr"), (pl.col("n3") - pl.col("s3")).alias("n3r"))
    m = m.filter((pl.col("n3r") == 0) | (pl.col("n3r") == pl.col("kr")))
    m = m.with_columns(pl.when(pl.col("n3r") == 0).then(pl.col("s3") == 1).otherwise(pl.col("s3") == 0).alias("fills"))
    city = [set(fold(a).replace("-", " ").split(",")[-2].split()) if a.count(",") >= 2 else set() for a in m["sa"].to_list()]
    sc = [subclass(a, b, c) for a, b, c in zip(m["sn"].to_list(), m["tn"].to_list(), city)]
    return m.with_columns(pl.Series("sub", sc))


pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
tr = table("dataset/train/train_ground_truth.tsv", "train", ["US", "India"])
ref = tr.group_by("kr").agg(pl.col("fills").mean().alias("ref"))
print("training reference (true links):", ref.sort("kr").rows())
fr = table(sys.argv[1], "test", ["France"]).join(ref, on="kr")
t = fr.group_by("sub").agg(pl.len().alias("n"), pl.col("fills").mean().round(3).alias("fill"), pl.col("ref").mean().alias("ref"))
t = t.with_columns(((pl.col("fill") - 0.5) / (pl.col("ref") - 0.5)).round(2).alias("pi"),
                   (0.5 / (pl.col("n").sqrt() * (pl.col("ref") - 0.5))).round(2).alias("pi_se"))
print(t.filter(pl.col("n") >= 150).sort("n", descending=True))

if len(sys.argv) > 2:
    ALIAS = r"(?i)\b(dba|d/b/a|d\.b\.a\.?|aka|a\.k\.a\.?|a/k/a|fka|f\.k\.a\.?|f/k/a|formerly|nee|née|doing business as|trading as|known as|anciennement|dit|ex)\b"
    x = fr.filter(pl.col("sub") == sys.argv[2]).with_columns(pl.col("tn").str.contains(ALIAS).alias("alias"))
    print(x.group_by("alias").agg(pl.len(), pl.col("fills").mean().round(3)).rows())
    pl.Config.set_fmt_str_lengths(60)
    print(x.filter(~pl.col("alias")).sample(min(25, x.height), seed=1).select("sn", "tn"))

"""Learn a native-script to Latin token dictionary from training matches.

Some Source 2/3 records spell Indian names and addresses in Devanagari, Bengali,
Tamil, Kannada and other scripts, while Source 1 is always Latin. For matched
training pairs whose names have the same token count, tokens are aligned by
position; address tokens are aligned by co-occurrence (Dice score). Only the
supplied training data is used.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl


NATIVE = re.compile(r"[ऀ-෿]")
SPLIT = re.compile(r"[\s,.;:()\[\]{}&+/\\#\"'_|-]+")


def tokens(text: str | None) -> list[str]:
    if not text:
        return []
    return [t for t in SPLIT.split(text.lower()) if t]


def learn(data_dir: Path, min_count: int = 2) -> dict[str, str]:
    read = lambda name: pl.read_csv(data_dir / name, separator="\t", quote_char=None,
                                    infer_schema=False, missing_utf8_is_empty_string=True)
    s1 = read("train_source1.tsv")
    targets = pl.concat([read("train_source2.tsv"), read("train_source3.tsv")])
    native = r"[ऀ-෿]"
    targets = targets.filter(pl.col("business_name").str.contains(native)
                             | pl.col("business_address").str.contains(native))
    pairs = (
        read("train_ground_truth.tsv")
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .join(targets, left_on="matched_entity_ids", right_on="entity_id")
        .join(s1, left_on="source1_entity_id", right_on="entity_id", suffix="_s1")
    )
    print(f"native-script matched pairs: {pairs.height:,}", flush=True)
    name_counts: dict[str, Counter] = defaultdict(Counter)
    pair_counts: Counter = Counter()
    native_totals: Counter = Counter()
    latin_totals: Counter = Counter()
    for name, s1_name, address, s1_address in zip(
        pairs["business_name"], pairs["business_name_s1"],
        pairs["business_address"], pairs["business_address_s1"],
    ):
        a, b = tokens(name), tokens(s1_name)
        if len(a) == len(b):
            for x, y in zip(a, b):
                if NATIVE.search(x) and not NATIVE.search(y):
                    name_counts[x][y] += 1
        native_address = {t for t in tokens(address) if NATIVE.search(t)}
        if native_address:
            latin = {t for t in tokens(s1_address) if not NATIVE.search(t) and not t.isdigit()}
            native_totals.update(native_address)
            latin_totals.update(latin)
            pair_counts.update((x, y) for x in native_address for y in latin)
    mapping: dict[str, str] = {}
    for token, counter in name_counts.items():
        latin, count = counter.most_common(1)[0]
        if count >= min_count and count >= 0.5 * sum(counter.values()):
            mapping[token] = latin
    best: dict[str, tuple[float, str]] = {}
    for (x, y), count in pair_counts.items():
        if count < min_count:
            continue
        dice = 2 * count / (native_totals[x] + latin_totals[y])
        if dice > best.get(x, (0.3, ""))[0]:
            best[x] = (dice, y)
    for token, (_, latin) in best.items():
        mapping.setdefault(token, latin)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/train"))
    parser.add_argument("--output", type=Path, default=Path("models/translit.json"))
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    args = parser.parse_args()
    mapping = learn(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mapping, ensure_ascii=False, indent=0, sort_keys=True))
    print(f"learned {len(mapping):,} tokens", flush=True)
    # Coverage of native tokens in the test targets (occurrence weighted).
    seen = total = 0
    for source in (2, 3):
        frame = pl.read_csv(args.test_dir / f"test_source{source}.tsv", separator="\t",
                            quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
        for column in ("business_name", "business_address"):
            for text in frame.filter(pl.col(column).str.contains(r"[ऀ-෿]"))[column]:
                for token in tokens(text):
                    if NATIVE.search(token):
                        total += 1
                        seen += token in mapping
    print(f"test native token coverage: {seen / max(total, 1):.4f} of {total:,}", flush=True)
    sample = list(mapping.items())[:: max(1, len(mapping) // 25)]
    print(sample)


if __name__ == "__main__":
    main()

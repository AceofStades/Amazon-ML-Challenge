import unittest
from unittest.mock import patch

import polars as pl

from amzn.france_rule import drop_house_conflicts, drop_swaps, standalone_target_numbers


class FranceHouseNumbersTest(unittest.TestCase):
    def test_explicit_number_prefix_preserves_real_house(self):
        addresses = pl.Series([
            "ndeg210 rue de lepeule batiment a etage 7 appartement 74 roubaix frhdf",
            "no ndeg0012 rue des fleurs appartement 7",
            None,
        ])
        self.assertEqual(standalone_target_numbers(addresses).to_list(),
                         [["210", "7", "74"], ["12", "7"], []])

    def test_street_artifacts_do_not_become_house_numbers(self):
        addresses = pl.Series([
            "4 cite1 de verdun",
            "espl1 des antilles chem1 du moulin",
            "xndeg210 rue du moulin",
        ])
        self.assertEqual(standalone_target_numbers(addresses).to_list(), [["4"], [], []])

    def test_pruning_keeps_number_prefix_match_and_drops_real_conflict(self):
        sources = pl.DataFrame({"entity_id": ["S1-good", "S1-conflict", "S1-us"],
                                "si": [0, 1, 2], "country": ["France", "France", "US"]})
        targets = pl.DataFrame({"entity_id": ["S2-good", "S2-conflict", "S2-us"], "ti": [0, 1, 2]})
        accepted = pl.DataFrame({"source1_entity_id": sources["entity_id"],
                                 "matched_entity_ids": targets["entity_id"]})
        cached_sources = pl.DataFrame({"house": ["210", "210", "210"]})
        cached_targets = pl.DataFrame({"address": [
            "ndeg210 rue de lepeule batiment a etage 7 appartement 74 roubaix frhdf",
            "ndeg340 rue de lepeule appartement 7 roubaix frhdf",
            "ndeg340 rue de lepeule appartement 7 roubaix frhdf",
        ]})

        def cached(path):
            if path == "cache/test_source1.tsv":
                return cached_sources
            if path == "cache/test_source2.tsv":
                return cached_targets
            return cached_targets.head(0)

        with patch("amzn.france_rule.read", side_effect=cached):
            kept = drop_house_conflicts(accepted, sources, targets, "France")
        self.assertEqual(kept["source1_entity_id"].to_list(), ["S1-good", "S1-us"])
        self.assertEqual(cached_sources["house"].to_list(), ["210", "210", "210"])

    def test_number_prefix_repair_keeps_common_word_swap_guard(self):
        ids = ["S1-swap", "S1-good", *[f"S1-vocabulary-{index}" for index in range(20)]]
        sources = pl.DataFrame({"entity_id": ids, "si": range(len(ids)), "country": ["France"] * len(ids)})
        targets = pl.DataFrame({"entity_id": ["S2-swap", "S2-good"], "ti": [0, 1]})
        accepted = pl.DataFrame({"source1_entity_id": ids[:2], "matched_entity_ids": targets["entity_id"]})
        cached_sources = pl.DataFrame({"entity_id": ids, "house": ["210"] * len(ids),
            "name_core": ["lille club", "organisme", *[f"vocabulary{index} ecole" for index in range(20)]]})
        cached_targets = pl.DataFrame({"entity_id": targets["entity_id"], "name_core": ["lille ecole", "organisme"],
            "address": ["ndeg210 rue de lepeule appartement 74 roubaix"] * 2})

        def cached(path):
            if path == "cache/test_source1.tsv":
                return cached_sources
            if path == "cache/test_source2.tsv":
                return cached_targets
            return cached_targets.head(0)

        with patch("amzn.france_rule.read", side_effect=cached):
            kept = drop_house_conflicts(accepted, sources, targets, "France")
            kept = drop_swaps(kept, sources, targets, "France")
        self.assertEqual(kept["source1_entity_id"].to_list(), ["S1-good"])


if __name__ == "__main__":
    unittest.main()

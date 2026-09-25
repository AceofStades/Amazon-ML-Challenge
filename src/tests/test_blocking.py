import csv
import tempfile
import unittest
from pathlib import Path

from amzn.blocking import connect, generate_candidates, import_split, index_targets
from amzn.features import create_features
from amzn.normalize import RECORD_COLUMNS, normalize_record


class BlockingTest(unittest.TestCase):
    def test_transliteration_and_name_variation_reach_model(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            records = {
                1: [
                    ("S1-1", "Raj Investments LLP", "6(29) C.I.T. Colony, Chennai", "India"),
                    ("S1-2", "Payne Enterprises", "3315 Fremont Street, Peoria", "US"),
                ],
                2: [
                    ("S2-1", "राज इन्वेस्टमेंट्स एलएलपी", "6(29) C.I.T. Colony, Chennai", "India"),
                    ("S2-2", "Payne Enterprices", "3315 Fremont St, Peoria", "US"),
                ],
                3: [("S3-1", "Other Business", "91 Elsewhere", "France")],
            }
            for source, rows in records.items():
                with (cache / f"train_source{source}.tsv").open("w", newline="") as handle:
                    writer = csv.writer(handle, delimiter="\t")
                    writer.writerow(RECORD_COLUMNS)
                    for entity_id, name, address, country in rows:
                        writer.writerow(normalize_record({
                            "entity_id": entity_id,
                            "business_name": name,
                            "business_address": address,
                            "country": country,
                        }))

            connection = connect(cache / "test.duckdb", cache, threads=2)
            import_split(connection, cache, "train")
            index_targets(connection, "train")
            generate_candidates(connection, "train", "candidate_pairs")
            create_features(connection, "train", "candidate_pairs", "pair_features")
            pairs = set(connection.execute(
                "SELECT source1_entity_id, candidate_entity_id FROM pair_features"
            ).fetchall())
            self.assertIn(("S1-1", "S2-1"), pairs)
            self.assertIn(("S1-2", "S2-2"), pairs)
            self.assertNotIn(("S1-1", "S3-1"), pairs)


if __name__ == "__main__":
    unittest.main()

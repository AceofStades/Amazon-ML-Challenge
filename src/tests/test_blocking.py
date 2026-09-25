import csv
import tempfile
import unittest
from pathlib import Path

from amzn.blocking import (
    combine_candidates, connect, expand_candidates, generate_candidates,
    import_split, index_targets,
)
from amzn.features import create_features
from amzn.infer import write_grouped_output
from amzn.normalize import RECORD_COLUMNS, normalize_record


class BlockingTest(unittest.TestCase):
    def test_transliteration_and_name_variation_reach_model(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            records = {
                1: [
                    ("S1-1", "Raj Investments LLP", "6(29) C.I.T. Colony, Chennai", "India"),
                    ("S1-2", "Payne Enterprises", "3315 Fremont Street, Peoria", "US"),
                    ("S1-3", "Lumiere", "1 Rue Paris", "France"),
                ],
                2: [
                    ("S2-1", "राज इन्वेस्टमेंट्स एलएलपी", "6(29) C.I.T. Colony, Chennai", "India"),
                    ("S2-2", "Payne Enterprices", "3315 Fremont St, Peoria", "US"),
                ],
                3: [
                    ("S3-1", "Other Business", "91 Elsewhere", "France"),
                    ("S3-2", "Raj Investments Private", "6(29) C.I.T. Colony, Chennai", "India"),
                ],
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

            connection.execute("CREATE TABLE base_only AS SELECT * FROM candidate_pairs WHERE candidate_entity_id='S2-1'")
            connection.execute("CREATE TABLE confident_seeds(source1_entity_id VARCHAR, seed_id VARCHAR)")
            connection.execute("INSERT INTO confident_seeds VALUES ('S1-1', 'S2-1')")
            expand_candidates(connection, "train", "confident_seeds", "base_only", "siblings")
            combine_candidates(connection, "base_only", "siblings", "all_candidates")
            self.assertIn(("S1-1", "S3-2"), connection.execute(
                "SELECT source1_entity_id, candidate_entity_id FROM siblings"
            ).fetchall())
            create_features(connection, "train", "all_candidates", "all_features",
                            include_seed_features=True)
            self.assertEqual(connection.execute("SELECT count(*) FROM all_features").fetchone()[0], 2)

            connection.execute("CREATE TABLE test_source1 AS SELECT * FROM train_source1")
            connection.execute("CREATE TABLE test_features AS SELECT * FROM pair_features")
            output = cache / "candidate_pairs.tsv"
            write_grouped_output(
                connection, "test_features", "candidate_entity_id", output,
                "candidate_entity_ids",
            )
            text = output.read_text()
            self.assertIn("S1-1\tS2-1,S3-2\n", text)
            self.assertIn("S1-3\t\n", text)
            self.assertNotIn('""', text)
            partitioned_output = cache / "candidate_pairs_partitioned.tsv"
            write_grouped_output(
                connection, "test_features", "candidate_entity_id",
                partitioned_output, "candidate_entity_ids", partitions=2,
            )
            self.assertEqual(set(text.splitlines()), set(partitioned_output.read_text().splitlines()))


if __name__ == "__main__":
    unittest.main()

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.fleet_ready import main


class FleetReadyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fleet = self.root / "fleet.tsv"
        self.fleet.write_text("default\taccount/train\tscore\tlarge3\tdescription\n")
        self.cache = self.root / "cache"

    def write_scores(self, name="large3"):
        directory = self.cache / f"ce_{name}"
        directory.mkdir(parents=True, exist_ok=True)
        for split in ("train", "test"):
            pq.write_table(pa.table({"score": [0.5]}), directory / f"{split}_scores.parquet")
        return directory

    def run_cli(self, *arguments):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = main([
                "--fleet", str(self.fleet), "--cache-root", str(self.cache), *arguments,
            ])
        return status, output.getvalue(), errors.getvalue()

    def test_complete_fleet_and_local_models(self):
        self.write_scores()
        self.write_scores("base_hard2")
        self.write_scores("base_bnd")
        self.assertEqual(self.run_cli("--local", "base_hard2", "--local", "base_bnd"),
                         (0, "ready: 3 models\n", ""))

    def test_each_score_file_is_required(self):
        for split in ("train", "test"):
            with self.subTest(split=split):
                directory = self.write_scores()
                (directory / f"{split}_scores.parquet").unlink()
                self.assertEqual(self.run_cli(), (1, "pending: large3\n", ""))

    def test_invalid_and_partial_parquet_remain_pending(self):
        for contents in (b"unfinished download", b"PAR1"):
            with self.subTest(contents=contents):
                directory = self.write_scores()
                (directory / "test_scores.parquet").write_bytes(contents)
                self.assertEqual(self.run_cli(), (1, "pending: large3\n", ""))

    def test_comments_blank_lines_and_fourth_column(self):
        self.fleet.write_text(
            "# account\ttrain\tscore\tname\n\n"
            "   # comment\n\t\t\n"
            "default\taccount/train\tscore\tlarge3\tdescription\n"
        )
        self.write_scores()
        self.assertEqual(self.run_cli(), (0, "ready: 1 models\n", ""))

    def test_pending_local_model_is_reported(self):
        self.write_scores()
        self.assertEqual(self.run_cli("--local", "base_bnd"),
                         (1, "pending: base_bnd\n", ""))

    def test_invalid_fleet_fails_instead_of_reporting_ready(self):
        for contents in (
            "# only comments\n", "default\ttrain\tscore\n",
            "default\ttrain\tscore\t\n", "default\ttrain\tscore\t../outside\n",
            "default\ttrain\tscore\tlarge3\ndefault\ttrain\tscore\tlarge3\n",
        ):
            with self.subTest(contents=contents):
                self.fleet.write_text(contents)
                status, output, errors = self.run_cli()
                self.assertEqual(status, 2)
                self.assertEqual(output, "")
                self.assertIn("invalid fleet:", errors)

    def test_missing_fleet_is_configuration_error(self):
        self.fleet.unlink()
        status, output, errors = self.run_cli()
        self.assertEqual(status, 2)
        self.assertEqual(output, "")
        self.assertIn("invalid fleet:", errors)


if __name__ == "__main__":
    unittest.main()

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.score_ce_checkpointed import recover


class CheckpointedScoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        model = self.root / "model"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"weights")
        data = self.root / "data"
        data.mkdir()
        for number in (1, 2, 3):
            (data / f"test_source{number}.tsv").write_text("record\n")
        self.args = argparse.Namespace(
            split="test", pairs=self.root / "pairs.parquet", output=self.root / "scores.parquet",
            model_dir=model, data_dir=data, checkpoints=self.root / "checkpoints",
            batch_size=128, chunk_rows=2,
        )
        self.expected = pa.table({"si": [4, 1, 3, 2, 0], "ti": [9, 8, 7, 6, 5]})
        pq.write_table(self.expected, self.args.pairs)

    def fake_score(self, command, **kwargs):
        self.assertTrue(kwargs["check"])
        self.assertEqual(command[command.index("--batch-size") + 1], "128")
        self.assertEqual(kwargs["env"]["TOKENIZERS_PARALLELISM"], "false")
        pairs = pq.read_table(command[command.index("--pairs") + 1])
        scores = pairs.append_column("ce_logit", pa.array([0.5] * pairs.num_rows, type=pa.float32()))
        pq.write_table(scores, command[command.index("--output") + 1])

    def run_recovery(self, scorer=None):
        with contextlib.redirect_stdout(io.StringIO()):
            with patch("scripts.score_ce_checkpointed.subprocess.run", side_effect=scorer or self.fake_score) as child:
                recover(self.args)
        return child

    def test_complete_chunks_preserve_order_and_resume_without_gpu_calls(self):
        self.assertEqual(self.run_recovery().call_count, 3)
        actual = pq.read_table(self.args.output)
        self.assertTrue(actual.select(["si", "ti"]).equals(self.expected))
        self.assertEqual(actual.num_rows, 5)
        self.assertEqual(self.run_recovery().call_count, 0)
        self.assertFalse(self.args.output.with_name("scores.parquet.recovering").exists())

    def test_crash_keeps_completed_chunks_and_does_not_publish_partial_output(self):
        calls = 0

        def crash_after_first(command, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise subprocess.CalledProcessError(-6, command)
            self.fake_score(command, **kwargs)

        with self.assertRaises(subprocess.CalledProcessError):
            self.run_recovery(crash_after_first)
        self.assertFalse(self.args.output.exists())
        self.assertTrue((self.args.checkpoints / "part-000000.parquet").exists())
        self.assertEqual(self.run_recovery().call_count, 2)

    def test_corrupt_chunk_is_recomputed(self):
        self.run_recovery()
        (self.args.checkpoints / "part-000001.parquet").write_bytes(b"PAR1")
        self.assertEqual(self.run_recovery().call_count, 1)

    def test_wrong_pair_order_and_nonfinite_scores_are_not_reused(self):
        for failure in ("order", "nonfinite", "null"):
            with self.subTest(failure=failure):
                self.run_recovery()
                chunk = self.args.checkpoints / "part-000000.parquet"
                scores = pq.read_table(chunk)
                if failure == "order":
                    scores = scores.take(pa.array([1, 0]))
                else:
                    bad = None if failure == "null" else float("nan")
                    scores = scores.set_column(2, "ce_logit", pa.array([bad, 0.5], type=pa.float32()))
                pq.write_table(scores, chunk)
                self.assertEqual(self.run_recovery().call_count, 1)

    def test_model_content_change_rejects_stale_manifest_even_with_same_stat(self):
        self.run_recovery()
        weights = self.args.model_dir / "model.safetensors"
        previous = weights.stat()
        weights.write_bytes(b"changed")
        os.utime(weights, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "incompatible checkpoints"):
            self.run_recovery()

    def test_pair_content_change_rejects_stale_manifest(self):
        self.run_recovery()
        pq.write_table(pa.table({"si": [1], "ti": [2]}), self.args.pairs)
        with self.assertRaisesRegex(ValueError, "incompatible checkpoints"):
            self.run_recovery()

    def test_batch_or_chunk_changes_reject_stale_manifest(self):
        self.run_recovery()
        for name, value in (("batch_size", 64), ("chunk_rows", 3)):
            with self.subTest(name=name):
                original = getattr(self.args, name)
                setattr(self.args, name, value)
                with self.assertRaisesRegex(ValueError, "incompatible checkpoints"):
                    self.run_recovery()
                setattr(self.args, name, original)

    def test_manifest_records_split_and_content_identity(self):
        self.run_recovery()
        manifest = json.loads((self.args.checkpoints / "manifest.json").read_text())
        self.assertEqual(manifest["split"], "test")
        self.assertEqual(manifest["rows"], 5)
        self.assertEqual(manifest["batch_size"], 128)
        self.assertIn("sha256", manifest["pairs"])
        self.assertIn("sha256", manifest["model"][0])


class DeferredRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        source = Path(__file__).resolve().parents[2] / "scripts/gpu_recover_hard2.sh"
        self.watcher = scripts / "gpu_recover_hard2.sh"
        self.watcher.write_text(source.read_text())
        self.commands = self.root / "commands"
        self.commands.mkdir()
        self.python = self.root / ".venv/bin/python"
        self.python.parent.mkdir(parents=True)
        self.marker = self.root / "launched"
        self.python.write_text(
            '#!/bin/sh\n[ "$1" = -c ] && exit 1\nprintf "%s\\n" "$*" >> launched\n'
        )
        self.python.chmod(0o755)
        for name in ("base_hard2", "base_bnd"):
            model = self.root / f"models/crossencoder_{name}"
            model.mkdir(parents=True)
            for filename in ("head.pt", "config.json", "model.safetensors"):
                (model / filename).write_bytes(b"complete")
        self.fake_command("sleep", "exit 42")

    def fake_command(self, name, body):
        command = self.commands / name
        command.write_text(f"#!/bin/sh\n{body}\n")
        command.chmod(0o755)

    def run_watcher(self, *, boundary_pid=99_999_999, gpu_memory=350, workers=False):
        self.fake_command("pgrep", f"exit {0 if workers else 1}")
        self.fake_command("nvidia-smi", f"printf '%s\\n' {gpu_memory}")
        environment = os.environ.copy()
        environment["PATH"] = str(self.commands) + os.pathsep + environment["PATH"]
        return subprocess.run(
            ["/bin/sh", str(self.watcher), str(boundary_pid)],
            env=environment, capture_output=True, text=True, timeout=5,
        )

    def test_boundary_chain_prevents_launch(self):
        result = self.run_watcher(boundary_pid=os.getpid())
        self.assertEqual(result.returncode, 42)
        self.assertFalse(self.marker.exists())

    def test_other_crossencoder_worker_prevents_launch(self):
        result = self.run_watcher(workers=True)
        self.assertEqual(result.returncode, 42)
        self.assertFalse(self.marker.exists())

    def test_busy_or_unknown_gpu_prevents_launch(self):
        for memory, status in ((2048, 42), ("unknown", 2)):
            with self.subTest(memory=memory):
                self.assertEqual(self.run_watcher(gpu_memory=memory).returncode, status)
                self.assertFalse(self.marker.exists())

    def test_idle_gpu_launches_only_explicit_bounded_scoring(self):
        result = self.run_watcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        launches = self.marker.read_text().splitlines()
        self.assertEqual(len(launches), 4)
        self.assertTrue(all("--batch-size 128 --chunk-rows 250000" in call for call in launches))
        self.assertIn("--split train", launches[0])
        self.assertIn("--split test", launches[1])
        self.assertTrue(all("crossencoder_base_hard2" in call for call in launches[:2]))
        self.assertTrue(all("crossencoder_base_bnd" in call for call in launches[2:]))

    def test_complete_existing_scores_are_preserved(self):
        self.python.write_text(
            '#!/bin/sh\nif [ "$1" = -c ]; then\n'
            '  [ "$3" = base_hard2 ] && [ "$4" = train ] && exit 0\n'
            '  exit 1\nfi\nprintf "%s\\n" "$*" >> launched\n'
        )
        self.assertEqual(self.run_watcher().returncode, 0)
        launches = self.marker.read_text().splitlines()
        self.assertEqual(len(launches), 3)
        self.assertIn("crossencoder_base_hard2", launches[0])
        self.assertIn("--split test", launches[0])

    def test_unfinished_boundary_model_is_never_scored(self):
        (self.root / "models/crossencoder_base_bnd/head.pt").unlink()
        result = self.run_watcher()
        self.assertEqual(result.returncode, 1)
        self.assertIn("base_bnd has no final model checkpoint", result.stderr)
        launches = self.marker.read_text().splitlines()
        self.assertEqual(len(launches), 2)
        self.assertTrue(all("crossencoder_base_hard2" in call for call in launches))


if __name__ == "__main__":
    unittest.main()

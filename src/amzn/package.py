"""Create the final submission archive without bundling raw challenge data.

Layout:
  output/matching_results.tsv, output/candidate_pairs.tsv   (from --submission-dir)
  code/business_entity_resolution/{src,scripts,kaggle,docs,README.md,requirements.txt,...}
  code/business_entity_resolution/model/...                 (trained artifacts)
  Documentation_template.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def build_package(root: Path, team_name: str, submission_dir: Path, stage1_dir: Path, blend_dir: Path,
                  transformers: bool) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", team_name.lower().replace("'", "").replace("’", "")).strip("_")
    if not slug:
        raise ValueError("Team name must contain letters or numbers")
    required = [
        submission_dir / "matching_results.tsv", submission_dir / "candidate_pairs.tsv",
        root / "README.md", root / "requirements.txt", root / "pyproject.toml", root / "uv.lock",
        root / ".python-version", root / "Documentation_template.md", root / "models" / "translit.json",
        stage1_dir / "stage1.json", blend_dir / "blend.json", blend_dir / "blend_meta.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing package files: " + ", ".join(missing))
    archive = root / "output" / f"{slug}_submission.zip"
    code_root = "code/business_entity_resolution"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            bundle.write(submission_dir / name, f"output/{name}")
        for name in ("README.md", "requirements.txt", "pyproject.toml", "uv.lock", ".python-version"):
            bundle.write(root / name, f"{code_root}/{name}")
        for folder, pattern in (("src", "*.py"), ("scripts", "*.sh"), ("kaggle", "*.py"), ("kaggle", "*.json"),
                                ("docs", "*.md")):
            for path in sorted((root / folder).rglob(pattern)):
                if "data" in path.relative_to(root / folder).parts:
                    continue
                bundle.write(path, f"{code_root}/{path.relative_to(root)}")
        bundle.write(root / "models" / "translit.json", f"{code_root}/model/translit.json")
        for path in sorted(stage1_dir.glob("stage1*.json")):
            bundle.write(path, f"{code_root}/model/stage1/{path.name}")
        for name in ("blend.json", "blend_meta.json"):
            bundle.write(blend_dir / name, f"{code_root}/model/blend/{name}")
        if transformers:
            for folder in ("encoder", "crossencoder"):
                for path in sorted((root / "models" / folder).rglob("*")):
                    if path.is_file():
                        bundle.write(path, f"{code_root}/model/{folder}/{path.relative_to(root / 'models' / folder)}")
        bundle.write(root / "Documentation_template.md", "Documentation_template.md")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--submission-dir", type=Path, default=Path("output"))
    parser.add_argument("--stage1-dir", type=Path, default=Path("output/v4"))
    parser.add_argument("--blend-dir", type=Path, default=Path("output/v4_blend"))
    parser.add_argument("--include-transformers", action="store_true",
                        help="Also bundle the fine-tuned bi-encoder and cross-encoder weights (about 1 GB)")
    args = parser.parse_args()
    root = args.root.resolve()
    archive = build_package(root, args.team_name, (root / args.submission_dir).resolve(),
                            (root / args.stage1_dir).resolve(), (root / args.blend_dir).resolve(),
                            args.include_transformers)
    print(f"Created {archive}")


if __name__ == "__main__":
    main()

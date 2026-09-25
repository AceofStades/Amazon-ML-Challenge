"""Create final submission archive without bundling raw challenge data."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def build_package(root: Path, team_name: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", team_name.lower()).strip("_")
    if not slug:
        raise ValueError("Team name must contain letters or numbers")
    output = root / "output"
    required = [
        output / "matching_results.tsv",
        output / "candidate_pairs.tsv",
        root / "README.md",
        root / "requirements.txt",
        root / "pyproject.toml",
        root / "uv.lock",
        root / ".python-version",
        root / "Documentation_template.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing package files: " + ", ".join(missing))
    archive = output / f"{slug}_submission.zip"
    code_root = "code/business_entity_resolution"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            bundle.write(output / name, f"output/{name}")
        for name in ("README.md", "requirements.txt", "pyproject.toml", "uv.lock", ".python-version"):
            bundle.write(root / name, f"{code_root}/{name}")
        for path in sorted((root / "src").rglob("*.py")):
            bundle.write(path, f"{code_root}/src/{path.relative_to(root / 'src')}")
        for name in ("pair_model.json", "training_metrics.json"):
            path = output / name
            if path.is_file():
                bundle.write(path, f"{code_root}/model/{name}")
        bundle.write(root / "Documentation_template.md", "Documentation_template.md")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--team-name", required=True)
    args = parser.parse_args()
    archive = build_package(args.root.resolve(), args.team_name)
    print(f"Created {archive}")


if __name__ == "__main__":
    main()

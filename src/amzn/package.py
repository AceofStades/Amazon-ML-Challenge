"""Create the final submission archive without bundling raw challenge data.

Layout (all source code under src/, as the challenge requires):
  output/matching_results.tsv, output/candidate_pairs.tsv   (from --submission-dir)
  code/business_entity_resolution/src/{amzn,tests,scripts,kaggle}
  code/business_entity_resolution/{docs,README.md,requirements.txt,pyproject.toml,uv.lock}
  code/business_entity_resolution/model/...                 (trained artifacts)
  Documentation_template.md
The repository keeps scripts/ and kaggle/ at its root; the packaged copies live under src/ and their path references
(and the README's) are rewritten to match, so every command runs from code/business_entity_resolution/.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

PATH_REF = re.compile(r"(?<![A-Za-z0-9_./~-])(?:\./)?(scripts|kaggle)/")


def relocate(text: str) -> str:
    """Rewrite repository-root paths of scripts/ and kaggle/ to their packaged place under src/."""
    text = text.replace('cd "$(dirname "$0")/.."', 'cd "$(dirname "$0")/../.."')
    text = text.replace('parents[1] / "src"', 'parents[2] / "src"')
    text = text.replace('sys.path.insert(0, "scripts")', 'sys.path.insert(0, "src/scripts")')
    return PATH_REF.sub(lambda m: f"src/{m.group(1)}/", text)


def write_text(bundle: ZipFile, text: str, name: str, executable: bool = False) -> None:
    info = ZipInfo(name, date_time=(2026, 9, 28, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = (0o755 if executable else 0o644) << 16
    bundle.writestr(info, text)


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
        readme = (root / "README.md").read_text()
        readme = "\n".join(line for line in readme.splitlines() if "HANDOFF.md" not in line) + "\n"
        write_text(bundle, relocate(readme), f"{code_root}/README.md")
        for name in ("requirements.txt", "pyproject.toml", "uv.lock", ".python-version"):
            bundle.write(root / name, f"{code_root}/{name}")
        for path in sorted((root / "src").rglob("*.py")):
            if "tests" in path.relative_to(root / "src").parts:
                write_text(bundle, relocate(path.read_text()), f"{code_root}/{path.relative_to(root)}")
            else:
                bundle.write(path, f"{code_root}/{path.relative_to(root)}")
        for folder, patterns in (("scripts", ("*.sh", "*.py")), ("kaggle", ("*.py", "*.json", "*.tsv"))):
            for pattern in patterns:
                for path in sorted((root / folder).rglob(pattern)):
                    if "data" in path.relative_to(root / folder).parts or "data_hard" in path.relative_to(root / folder).parts:
                        continue
                    write_text(bundle, relocate(path.read_text()), f"{code_root}/src/{path.relative_to(root)}",
                               executable=path.suffix == ".sh")
        for path in sorted((root / "docs").rglob("*.md")):
            write_text(bundle, relocate(path.read_text()), f"{code_root}/{path.relative_to(root)}")
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

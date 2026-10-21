"""Copy the notebooks of the notebooks repository into the manual's tree.

    python docs/tools/collect_notebooks.py <notebooks checkout> <docs/notebooks>

The notebooks repository holds ``tutorials/`` and ``paper/`` (one folder each,
notebooks with their stored outputs and the figures they wrote). Each folder is
copied whole into its group's folder of the manual (``paper/`` into
``chiang2026d/``), whose ``index.rst`` lists them and is kept. Nothing is
executed here; the build renders the stored outputs.
"""

import shutil
import sys
from pathlib import Path

# the notebooks repository's folder -> the manual's group folder
FOLDERS = {"tutorials": "tutorials", "paper": "chiang2026d"}
SKIP = {".ipynb_checkpoints", "__pycache__", ".git", "README.md"}
KEEP = {"index.rst"}


def main(source: str, target: str) -> int:
    src, dst = Path(source), Path(target)
    if not src.is_dir():
        print(f"no notebooks checkout at {src}; the manual builds without them")
        return 0
    for name, group in FOLDERS.items():
        folder = src / name
        if not folder.is_dir():
            print(f"{folder} is absent; skipped")
            continue
        out = dst / group
        # the group's own page stays; everything copied before goes
        for child in out.iterdir() if out.is_dir() else ():
            if child.name in KEEP:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        shutil.copytree(
            folder,
            out,
            ignore=lambda _d, names: [n for n in names if n in SKIP],
            dirs_exist_ok=True,
        )
        count = sum(1 for _ in out.rglob("*.ipynb"))
        print(f"{name}: {count} notebooks copied to {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))

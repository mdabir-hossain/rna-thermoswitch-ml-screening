"""Generate canonical Phase 2D report figures."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from thermoswitch.plotting import make_all_figures


def main() -> None:
    paths = make_all_figures()
    print("Generated Phase 2D figures:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()

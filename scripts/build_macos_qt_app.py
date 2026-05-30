from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ashare_cross_section_similarity.desktop.packaging import build_macos_app


def main() -> int:
    app_path = build_macos_app()
    print(app_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ashare_cross_section_similarity.benchmark import run_benchmark
from ashare_cross_section_similarity.cli import _parse_args


def _write_bars(root: Path, symbol: str, closes: list[float]) -> None:
    qfq = root / "qfq"
    qfq.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    pd.DataFrame(
        {
            "date": dates,
            "stock_code": symbol,
            "open": closes,
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.98 for value in closes],
            "close": closes,
            "volume": [100 + index for index in range(len(closes))],
            "amount": [1000 + index * 10 for index in range(len(closes))],
        }
    ).to_parquet(qfq / f"{symbol}.parquet", index=False)


def test_parse_benchmark_command() -> None:
    args = _parse_args(
        [
            "benchmark",
            "--cases",
            "docs/research/similarity_benchmark_cases.yaml",
            "--algorithms",
            "baseline_price_feature,return_shape",
            "--output",
            "outputs/research",
        ]
    )

    assert args.command == "benchmark"
    assert args.algorithms == "baseline_price_feature,return_shape"


def test_run_benchmark_writes_csv_and_html_gallery(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    _write_bars(data_root, "000001.SZ", [10, 11, 12, 11, 13, 20, 22, 24, 22, 26])
    _write_bars(data_root, "000002.SZ", [20, 22, 24, 22, 26, 30, 31, 32, 33, 34])
    _write_bars(data_root, "000003.SZ", [8, 7, 6, 5, 4, 10, 9, 8, 7, 6])
    cases_file = tmp_path / "cases.yaml"
    cases_file.write_text(
        """
cases:
  - name: history_fixture
    mode: history
    symbol: 000001.SZ
    start: 2024-01-01
    end: 2024-01-05
    top_n: 2
  - name: cross_fixture
    mode: cross_section
    target_symbol: 000001.SZ
    start: 2024-01-01
    end: 2024-01-05
    universe_symbols:
      - 000002.SZ
      - 000003.SZ
    top_n: 2
""".strip(),
        encoding="utf-8",
    )
    output = tmp_path / "research"

    result = run_benchmark(
        cases_path=cases_file,
        algorithms=("baseline_price_feature", "return_shape"),
        output_dir=output,
        data_root=data_root,
        timeframe="1d",
        adjust="qfq",
    )

    assert len(result.summary) == 4
    assert (output / "benchmark_summary.csv").exists()
    assert (output / "benchmark_results.csv").exists()
    html = (output / "benchmark_gallery.html").read_text(encoding="utf-8")
    assert "history_fixture" in html
    assert "cross_fixture" in html
    assert "baseline_price_feature" in html

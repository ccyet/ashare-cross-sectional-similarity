# A股相似阶段搜集交接说明

更新时间：2026-05-18

## 当前能力

- `历史时序相似`：对同一标的自定义一段窗口，在该标的历史里找相似阶段，输出后验收益、回撤、浮盈统计。
- `横截面相似`：对目标标的一段走势，在指定股票范围内找相似标的；候选窗口可在目标区间前后按交易日整体平移，默认容错 `±5` 根。
- `数据检查 / 下载`：历史页和横截面页共用 `data_check` 与 `_download_symbols_with_progress`，只补缺文件、覆盖不足、区间无数据或读取失败的标的；下载引擎支持 `trend`、`openbb`、`tdx`。
- `自定义价格数据`：支持页面上传或 CLI 导入 `csv / parquet`，写入统一本地 parquet 结构。
- `图表`：页面展示结果表、统计计量、收盘价折线、大小盘价差率、K 线聚合图。
- `Docker`：`docker compose up --build` 可把本机行情目录挂载进容器运行 Streamlit 服务。

## 本地数据约定

默认复用 `trend-backtest` 行情目录：

```text
/Users/a1234/Desktop/trend-backtest/data/market/daily/qfq/<symbol>.parquet
```

页面侧栏的 `本地行情根目录` 应指向：

```text
/Users/a1234/Desktop/trend-backtest/data/market/daily
```

每个 parquet 至少需要：

```text
date, symbol 或 stock_code, open, high, low, close
```

`volume`、`amount` 可缺失，代码会补空值。

## 缓存与覆盖检查

Streamlit 的本地行情缓存包含 parquet 文件指纹：

```text
symbol, st_mtime_ns, st_size
```

所以页面内外更新本地 parquet 后，刷新页面即可读取新数据。不要再用固定参数缓存本地行情，否则会复现“文件已更新但页面仍显示旧日期”的问题。

覆盖检查表区分两组日期：

- `请求开始 / 请求结束`：用户要检查或搜索的区间。
- `本地开始 / 本地结束`：parquet 文件真实覆盖范围。

## 关键命令

运行页面：

```bash
streamlit run streamlit_app.py
```

运行 Docker 服务：

```bash
docker compose up --build
```

检查本地覆盖：

```bash
python -m ashare_cross_section_similarity check \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --timeframe 1d \
  --symbols 399006.SZ,300750.SZ \
  --start 2024-01-01 \
  --end 2024-03-31
```

下载全 A 日线：

```bash
python scripts/download_all_a_daily.py \
  --trend-repo /Users/a1234/Desktop/trend-backtest \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --start 1990-01-01 \
  --adjust qfq \
  --batch-size 100 \
  --provider akshare \
  --output outputs/all_a_daily_download_log.csv
```

直接用本机 TDX 下载：

```bash
python -m ashare_cross_section_similarity download \
  --download-engine tdx \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --timeframe 1d \
  --provider /path/to/TdxInstall/PYPlugins/user \
  --symbols 300750.SZ \
  --start 2024-01-01 \
  --end 2024-03-31
```

## 验证基线

当前验证命令：

```bash
pytest -q
ruff check .
python -m py_compile streamlit_app.py ashare_cross_section_similarity/*.py scripts/download_all_a_daily.py
git diff --check
```

2026-05-18 最新验证结果：`109 passed`，`ruff check .` 通过。

## 已知边界

- 指数成分、行业和概念列表来自 AkShare 当前接口，不是历史时点成分。
- 搜索结果只做研究辅助，不构成交易建议。
- OpenBB 和 TDX 直连链路是可选增强，默认仍走 `trend-backtest`。
- 目前没有自动交易、组合回测或历史成分回溯。

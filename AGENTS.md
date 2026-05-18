# Agent 规则

## 项目定位

这是一个 A 股相似阶段研究工具，保留同一标的历史时序相似搜索，并新增同一时间横截面相似搜索。默认数据源复用：

```text
/Users/a1234/Desktop/trend-backtest/data/market/daily
```

默认页面入口：

```bash
streamlit run streamlit_app.py
```

## 工作原则

- 默认中文回复，直接说明动作、结果、验证和阻塞。
- 优先做最小且正确的修改，不顺手重构无关模块。
- 不要牺牲历史时序相似能力；横截面能力只能增量扩展。
- 不要在源码中硬编码 secret、API key、密码或用户凭证。
- 不要用静默 fallback 掩盖数据、下载或路径错误；失败要显式可定位。

## 数据约定

- 本地行情根目录传到 `data/market/daily` 这一层，代码会按周期和复权映射到 `<timeframe>/<adjust>` 或 `daily/<adjust>`。
- parquet 文件名使用规范代码，如 `399006.SZ.parquet`、`000852.SH.parquet`。
- 价格数据至少包含 `date, symbol 或 stock_code, open, high, low, close`；`volume, amount` 可缺失。
- 历史页和横截面页应复用 `data_check`、`load_local_bars`、`_download_symbols_with_progress` 等同一套本地数据逻辑。
- Streamlit 本地行情缓存必须带 parquet 文件指纹，避免本地文件更新后页面仍显示旧范围。

## 当前页面能力

- 历史时序页支持自定义起止日期、快捷近 `5/10/20/60/120` 根、后验窗口、大小盘价差率和 K 线核验。
- 横截面页支持日期容错；目标区间不扩大，只允许候选窗口整体平移，后验收益从候选命中窗口结束日之后计算。
- 横截面结果表要显示代码和股票名称，百分比字段按百分比展示，距离和斜率等小数字段最多保留两位小数。
- 横截面 K 线聚合图使用桌面 3 列布局，窄屏退到 2 列或 1 列。
- 下载大量数据时必须保留可见进度。

## 验证

改代码后优先跑：

```bash
pytest -q
ruff check .
python -m py_compile streamlit_app.py ashare_cross_section_similarity/*.py scripts/download_all_a_daily.py
git diff --check
```

涉及页面行为时，用本地浏览器检查 `http://localhost:8502/` 或当前 Streamlit 端口。页面刷新验证优先检查标题 `A股相似阶段搜集`、两个 tab、目标功能区是否可见。

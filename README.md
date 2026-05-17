# A股横截面相似搜集

这是一个独立于 `trend-backtest` 的新库，用来做“同一时间窗口内”的横截面相似搜索：

> 先选定某个个股、指数或板块代理的一段区间走势，再在同一段时间里，从指定股票池、指数成分、行业板块或概念板块中寻找走势和环境最相似的标的。

当前版本已经包含完整闭环：

- 数据抓取：通过 AkShare 拉取日线和近端分钟线
- 数据落地：统一写入本地 parquet
- 数据检查：检查目标区间覆盖、缺文件、区间缺失
- 相似搜索：在同一时间窗口内做横截面相似排序
- 页面展示：Streamlit 页面完成检查、下载、搜索、导出

本库不做自动交易。

## 1. 能解决什么问题

典型问题：

- 目标标的：`300750.SZ`
- 目标区间：`2024-01-01` 到 `2024-03-31`
- 搜索范围：沪深300成分、某个行业板块、某个概念板块、或手工指定的一批代码
- 输出：同一段时间内走势最像目标标的的 Top N 个股票/板块代理

系统会比较：

- 归一化后的收盘路径
- 区间收益
- 实现波动率
- 最大回撤
- 趋势斜率
- 下跌放量占比
- 量价相关
- 成交规模

默认综合相似度为：

```text
综合相似度 = 路径相似度 * 70% + 特征相似度 * 30%
```

## 2. 安装

```bash
cd /Users/a1234/Desktop/ashare-cross-sectional-similarity
python -m pip install -r requirements.txt
```

## 3. 数据要求

默认读取本地 parquet；缺数据时可以用本库直接下载。

推荐直接复用 `trend-backtest` 的行情目录：

```text
/Users/a1234/Desktop/trend-backtest/data/market/daily/qfq/000001.SZ.parquet
/Users/a1234/Desktop/trend-backtest/data/market/30m/qfq/000001.SZ.parquet
```

每个 parquet 至少需要这些列：

```text
date, symbol 或 stock_code, open, high, low, close
```

可选列：

```text
volume, amount
```

如果读取 `30m`，`--data-root` 仍可传 `.../data/market/daily`，程序会自动映射到同级 `30m` 目录。

## 4. 数据抓取与检查

### 4.1 下载行情

```bash
python -m ashare_cross_section_similarity download \
  --data-root data/market/daily \
  --timeframe 1d \
  --symbols 300750.SZ,000001.SZ,600519.SH \
  --start 2024-01-01 \
  --end 2024-03-31
```

支持周期：

```text
1d, 30m, 15m, 5m, 1m
```

说明：

- 日线支持股票、指数、ETF。
- 分钟线支持股票、ETF；指数分钟线不同数据源覆盖不稳定，建议用指数 ETF 或本地 parquet。
- 30m 长历史不建议完全依赖 AkShare，最好提前准备本地长历史 parquet。

### 4.2 按指数、行业或概念范围下载

```bash
python -m ashare_cross_section_similarity download \
  --data-root data/market/daily \
  --timeframe 1d \
  --universe-index 000300 \
  --start 2024-01-01 \
  --end 2024-03-31
```

```bash
python -m ashare_cross_section_similarity download \
  --data-root data/market/daily \
  --timeframe 1d \
  --universe-industry 半导体 \
  --start 2024-01-01 \
  --end 2024-03-31
```

### 4.3 检查本地覆盖

```bash
python -m ashare_cross_section_similarity check \
  --data-root data/market/daily \
  --timeframe 1d \
  --symbols 300750.SZ,000001.SZ,600519.SH \
  --start 2024-01-01 \
  --end 2024-03-31
```

检查结果会显示：

- `available`：所选区间有本地行情
- `missing_file`：本地 parquet 文件不存在
- `missing_window`：文件存在，但所选区间没有行情
- `read_error`：文件读取失败

## 5. 横截面搜索命令

### 5.1 手工指定搜索范围

```bash
python -m ashare_cross_section_similarity search \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --timeframe 1d \
  --target-symbol 300750.SZ \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --universe-symbols 000001.SZ,600519.SH,002594.SZ \
  --top-n 20 \
  --output outputs/ningde_2024q1_similarity.csv
```

### 5.2 用文件限定搜索范围

支持 `csv / xlsx / parquet`，文件里有任一列即可：

```text
证券代码, 代码, stock_code, symbol, ts_code
```

示例：

```bash
python -m ashare_cross_section_similarity search \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --target-symbol 300750.SZ \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --universe-file examples/universe_sample.csv
```

### 5.3 用指数成分限定范围

需要安装 AkShare。成分列表取“当前最新成分”，不是严格历史时点成分。

```bash
python -m ashare_cross_section_similarity search \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --target-symbol 300750.SZ \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --universe-index 000300
```

### 5.4 用行业或概念板块限定范围

```bash
python -m ashare_cross_section_similarity search \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --target-symbol 300750.SZ \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --universe-industry 半导体
```

```bash
python -m ashare_cross_section_similarity search \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --target-symbol 300750.SZ \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --universe-concept 融资融券
```

## 6. 页面用法

```bash
streamlit run streamlit_app.py
```

页面操作顺序：

1. 填本地行情目录。
2. 选择周期，默认日线。
3. 填目标代码和区间。
4. 填搜索范围：代码列表、文件、指数成分、行业板块或概念板块。
5. 查看数据检查表。
6. 如缺数据，点击“下载或更新当前目标与搜索范围行情”。
7. 点击“运行横截面搜索”。
8. 查看 Top 相似标的表格、相似度柱状图，并下载 CSV。

## 7. 输出字段解释

| 字段 | 含义 |
| --- | --- |
| `综合相似度` | 路径相似度和特征相似度的加权结果，越高越像 |
| `路径相似度` | 只看归一化收盘路径形状 |
| `特征相似度` | 看收益、波动、回撤、斜率、成交环境 |
| `区间收益` | 该标的在所选区间内的涨跌幅 |
| `波动率` | 区间内日收益或 bar 收益的标准差 |
| `最大回撤` | 区间内从高点到低点的最大下跌 |
| `下跌放量占比` | 下跌 bar 消耗的成交量/成交额占比 |
| `量价相关` | 收益和成交量/成交额的相关性 |

## 8. 当前边界

- 第一版只做“同一时间窗口”的横截面比较，不做历史回溯。
- 指数成分、行业板块、概念板块来自 AkShare 当前接口，不保证历史成分时点准确。
- ETF 成分暂不自动抓取，建议先用 `--universe-file` 输入 ETF 持仓或自定义成分。
- 30m 全市场横截面依赖本地长历史 parquet；AkShare 分钟数据更适合近端补充。
- 结果是研究工具，不是买卖建议。

## 9. 开发验证

```bash
python -m pytest -q
python -m ruff check .
```

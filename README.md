# A股横截面相似搜集

这是一个独立于 `trend-backtest` 的新库，用来做“同一时间窗口内”的横截面相似搜索：

> 先选定某个个股、指数或板块代理的一段区间走势，再在同一段时间里，从指定股票池、指数成分、行业板块或概念板块中寻找走势和环境最相似的标的。

第一版聚焦“横截面搜集”的第一步，不做历史阶段回溯，不做自动交易。

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

如果要通过 AkShare 自动拉取指数、行业或概念成分：

```bash
python -m pip install "akshare>=1.17"
```

## 3. 数据要求

默认读取本地 parquet，不负责下载行情。

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

## 4. 命令行用法

### 4.1 手工指定搜索范围

```bash
python -m ashare_cross_section_similarity \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --timeframe 1d \
  --target-symbol 300750.SZ \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --universe-symbols 000001.SZ,600519.SH,002594.SZ \
  --top-n 20 \
  --output outputs/ningde_2024q1_similarity.csv
```

### 4.2 用文件限定搜索范围

支持 `csv / xlsx / parquet`，文件里有任一列即可：

```text
证券代码, 代码, stock_code, symbol, ts_code
```

示例：

```bash
python -m ashare_cross_section_similarity \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --target-symbol 300750.SZ \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --universe-file examples/universe_sample.csv
```

### 4.3 用指数成分限定范围

需要安装 AkShare。成分列表取“当前最新成分”，不是严格历史时点成分。

```bash
python -m ashare_cross_section_similarity \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --target-symbol 300750.SZ \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --universe-index 000300
```

### 4.4 用行业或概念板块限定范围

```bash
python -m ashare_cross_section_similarity \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --target-symbol 300750.SZ \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --universe-industry 半导体
```

```bash
python -m ashare_cross_section_similarity \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --target-symbol 300750.SZ \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --universe-concept 融资融券
```

## 5. 页面用法

```bash
streamlit run streamlit_app.py
```

页面操作顺序：

1. 填本地行情目录。
2. 选择周期，默认日线。
3. 填目标代码和区间。
4. 填搜索范围：代码列表、文件、指数成分、行业板块或概念板块。
5. 点击“运行横截面搜索”。
6. 查看 Top 相似标的表格、相似度柱状图，并下载 CSV。

## 6. 输出字段解释

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

## 7. 当前边界

- 第一版只做“同一时间窗口”的横截面比较，不做历史回溯。
- 指数成分、行业板块、概念板块来自 AkShare 当前接口，不保证历史成分时点准确。
- ETF 成分暂不自动抓取，建议先用 `--universe-file` 输入 ETF 持仓或自定义成分。
- 30m 全市场横截面依赖本地长历史 parquet；AkShare 分钟数据通常不足以支撑长区间研究。
- 结果是研究工具，不是买卖建议。

## 8. 开发验证

```bash
python -m pytest -q
python -m ruff check .
```

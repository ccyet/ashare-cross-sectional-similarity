# A股相似阶段搜集

这是一个独立新库，但能力设计上继承原 `trend-backtest` 的“同一标的历史时序相似阶段”研究方向，并在此基础上新增“同一时间横截面相似标的”搜索。

当前版本包含完整闭环：

- 数据抓取：默认委托原 `trend-backtest/scripts/update_data.py`，也可选用 OpenBB 直接写入本地 parquet
- 数据落地：统一写入本地 parquet
- 数据检查：检查目标区间覆盖、缺文件、区间缺失
- 历史时序搜索：同一标的自己的历史阶段相似度回溯
- 横截面搜索：同一时间窗口内，不同标的之间的相似度比较
- 页面展示：Streamlit 页面完成检查、下载、搜索、导出

本库不做自动交易。

## 1. 两类搜索

### 1.1 历史时序相似

问题形式：

- 目标标的：`399006.SZ`
- 当前窗口结束：`2024-03-31`
- 主走势窗口：`5 / 10 / 20 / 60 / 120` 根 K 线
- 输出：该标的历史上最像当前窗口的 Top N 个阶段，以及这些阶段之后的收益、回撤、最大浮盈

用途：

- 回答“现在这段走势，在历史上像哪几段？”
- 看历史样本之后的后验表现
- 避免把同一标的的历史回溯能力丢掉

### 1.2 横截面相似

问题形式：

- 目标标的：`300750.SZ`
- 目标区间：`2024-01-01` 到 `2024-03-31`
- 搜索范围：沪深300成分、某个行业板块、某个概念板块、或手工指定的一批代码
- 输出：同一段时间内走势最像目标标的的 Top N 个股票/板块代理

用途：

- 回答“同一时间里，还有哪些标的走得像它？”
- 可限定成分指数、板块 ETF 成分、行业或概念范围

## 2. 相似度口径

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

历史时序搜索会额外输出：

- 后 N 根收益
- 后 N 根最大回撤
- 后 N 根最大浮盈

## 3. 安装

```bash
cd /Users/a1234/Desktop/ashare-cross-sectional-similarity
python -m pip install -r requirements.txt
```

OpenBB 抓取链路是可选增强，不作为默认依赖。需要使用时再安装：

```bash
python -m pip install openbb openbb_akshare
python -c "import openbb; openbb.build()"
```

## 4. 数据要求

默认读取本地 parquet；缺数据时可以在本库页面或 CLI 中触发下载。下载动作默认调用原 `trend-backtest` 的 `scripts/update_data.py`，数据源、TDX/AkShare 路由和落地目录仍以原库配置为准。

如果使用 `--download-engine openbb`，程序会通过 OpenBB 的统一接口抓取行情，并按本库目录结构直接写入 parquet。A 股默认使用 `openbb_akshare` 扩展。

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

## 5. 数据抓取与检查

### 5.1 下载行情

```bash
python -m ashare_cross_section_similarity download \
  --trend-repo /Users/a1234/Desktop/trend-backtest \
  --timeframe 1d \
  --symbols 399006.SZ,300750.SZ,000001.SZ \
  --start 2024-01-01 \
  --end 2024-03-31
```

支持周期：

```text
1d, 30m, 15m, 5m, 1m
```

说明：

- 本命令不会在新库内重写行情抓取逻辑。
- 它只把参数转发给原库 `scripts/update_data.py`。
- 具体支持哪些周期、使用 AkShare 还是 TDX、落到哪个目录，以原 `trend-backtest/config/data_source.yaml` 和原脚本实现为准。
- 如需指定原脚本 provider，可加 `--provider akshare` 或 `--provider tdx`。

也可以用 OpenBB/AKShare 直接写入本地 parquet：

```bash
python -m ashare_cross_section_similarity download \
  --download-engine openbb \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --timeframe 1d \
  --provider akshare \
  --symbols 300750.SZ,000001.SZ,600519.SH \
  --start 2024-01-01 \
  --end 2024-03-31
```

### 5.2 按指数、行业或概念范围下载

```bash
python -m ashare_cross_section_similarity download \
  --trend-repo /Users/a1234/Desktop/trend-backtest \
  --timeframe 1d \
  --universe-index 000300 \
  --start 2024-01-01 \
  --end 2024-03-31
```

```bash
python -m ashare_cross_section_similarity download \
  --trend-repo /Users/a1234/Desktop/trend-backtest \
  --timeframe 1d \
  --universe-industry 半导体 \
  --start 2024-01-01 \
  --end 2024-03-31
```

### 5.3 检查本地覆盖

```bash
python -m ashare_cross_section_similarity check \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --timeframe 1d \
  --symbols 399006.SZ,300750.SZ,000001.SZ \
  --start 2024-01-01 \
  --end 2024-03-31
```

### 5.4 下载全 A 股票历史日线

脚本会用 AkShare 获取当前全 A 股票列表，按批次委托本库下载器抓取 `1d/qfq` 日线，并在每批结束后复查本地 parquet 覆盖情况，日志写入 CSV。

先做 dry-run：

```bash
python scripts/download_all_a_daily.py \
  --dry-run \
  --symbols-output outputs/all_a_symbols.csv
```

正式下载全历史：

```bash
python scripts/download_all_a_daily.py \
  --trend-repo /Users/a1234/Desktop/trend-backtest \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --start 1990-01-01 \
  --end 2026-05-17 \
  --adjust qfq \
  --batch-size 100 \
  --provider akshare \
  --output outputs/all_a_daily_download_log.csv
```

常用选项：

| 参数 | 含义 |
| --- | --- |
| `--skip-available` | 下载前跳过本地已覆盖区间的股票 |
| `--limit` | 调试时只下载前 N 个股票 |
| `--sleep` | 批次之间暂停秒数，避免数据源限流 |
| `--download-engine openbb` | 改用 OpenBB 直接写入本地 parquet |

## 6. 历史时序搜索命令

```bash
python -m ashare_cross_section_similarity history \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --timeframe 1d \
  --symbol 399006.SZ \
  --as-of 2024-03-31 \
  --window-size 20 \
  --forward-windows 5,20,60 \
  --top-n 10 \
  --output outputs/399006_history_similarity.csv
```

关键参数：

| 参数 | 含义 |
| --- | --- |
| `--symbol` | 要回溯的同一标的 |
| `--as-of` | 当前窗口结束日；匹配时不会使用此日期之后的数据 |
| `--window-size` | 当前窗口长度，支持 5、10、20、60、120 等 |
| `--forward-windows` | 历史样本结束后继续观察多少根 K 线 |
| `--exclusion-bars` | 排除当前窗口附近样本，避免“刚刚挨着”的阶段被误判为相似 |
| `--nearby-gap-days` | 历史样本之间最小间隔，默认 20 天；邻近重叠阶段只保留最像的一个 |

## 7. 横截面搜索命令

### 7.1 手工指定搜索范围

```bash
python -m ashare_cross_section_similarity search \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --timeframe 1d \
  --target-symbol 300750.SZ \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --universe-symbols 000001.SZ,600519.SH,002594.SZ \
  --top-n 20 \
  --output outputs/ningde_2024q1_cross_section.csv
```

### 7.2 用文件限定搜索范围

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

### 7.3 用指数、行业或概念限定范围

需要安装 AkShare。成分列表取“当前最新成分”，不是严格历史时点成分。

```bash
python -m ashare_cross_section_similarity search \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --target-symbol 300750.SZ \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --universe-index 000300
```

```bash
python -m ashare_cross_section_similarity search \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --target-symbol 300750.SZ \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --universe-industry 半导体
```

## 8. 页面用法

```bash
streamlit run streamlit_app.py
```

页面有两个工作台：

1. `历史时序相似`：同一标的自己的历史相似阶段回溯。
2. `横截面相似`：同一时间窗口内从指定范围找相似标的。

通用操作顺序：

1. 填本地行情目录。
2. 填原 `trend-backtest` 仓库路径。
3. 选择周期，默认日线。
4. 选择下载引擎：默认调用原库 `update_data.py`，也可选择 OpenBB。
5. 在对应工作台填写目标代码、窗口或区间；横截面工作台还需填写搜索范围。
6. 查看或点击数据检查。
7. 如缺数据，点击下载或更新。
8. 运行搜索。
9. 查看结果表、图表；横截面工作台会展示 TradingView lightweight-charts 交互走势对比，并可下载 CSV。

## 9. 输出字段解释

| 字段 | 含义 |
| --- | --- |
| `综合相似度` | 路径相似度和特征相似度的加权结果，越高越像 |
| `路径相似度` | 只看归一化收盘路径形状 |
| `特征相似度` | 看收益、波动、回撤、斜率、成交环境 |
| `区间收益` | 该窗口内的涨跌幅 |
| `波动率` | 区间内日收益或 bar 收益的标准差 |
| `最大回撤` | 区间内从高点到低点的最大下跌 |
| `下跌放量占比` | 下跌 bar 消耗的成交量/成交额占比 |
| `量价相关` | 收益和成交量/成交额的相关性 |
| `后N根收益` | 历史样本结束后 N 根 K 线的最终收益 |
| `后N根最大回撤` | 历史样本结束后 N 根 K 线内的最大回撤 |
| `后N根最大浮盈` | 历史样本结束后 N 根 K 线内的最大浮盈 |

## 10. 当前边界

- 历史时序搜索和横截面搜索分开运行，不自动合成总评分。
- 指数成分、行业板块、概念板块来自 AkShare 当前接口，不保证历史成分时点准确。
- ETF 成分暂不自动抓取，建议先用 `--universe-file` 输入 ETF 持仓或自定义成分。
- 默认数据抓取仍按原 `trend-backtest` 处理；OpenBB 链路为可选增强，依赖本机 OpenBB 与对应 provider 扩展是否可用。
- 结果是研究工具，不是买卖建议。

## 11. 开发验证

```bash
python -m pytest -q
python -m ruff check .
```

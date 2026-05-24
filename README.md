# A股相似阶段搜集

这是一个独立新库，但能力设计上继承原 `trend-backtest` 的“同一标的历史时序相似阶段”研究方向，并在此基础上新增“同一时间横截面相似标的”搜索。

当前版本包含完整闭环：

- 数据抓取：默认委托原 `trend-backtest/scripts/update_data.py`，也可选用原生 AkShare、OpenBB 或 TDX 直接写入本地 parquet
- 数据落地：统一写入本地 parquet
- 数据检查：历史时序和横截面共用同一套本地覆盖检查、下载修复和 parquet 缓存失效逻辑
- 历史时序搜索：同一标的自己的历史阶段相似度回溯，支持自定义起止区间和快捷近 N 根
- 横截面搜索：同一时间窗口内，不同标的之间的相似度比较，支持候选窗口日期容错平移
- 下载任务：Streamlit 下载支持任务摘要、进度条、当前标的提示、暂停和继续；暂停在批次边界生效，避免中断正在写入的批次
- 路径选择：目录和文件选择统一显示当前选择、选择按钮和清除按钮，路径不再只靠手填文本
- 页面展示：Streamlit 页面完成检查、下载、搜索、计量统计、走势图、K 线聚合和 CSV 导出

本库不做自动交易。

## 1. 两类搜索

### 1.1 历史时序相似

问题形式：

- 目标标的：`399006.SZ`
- 当前窗口：自定义 `区间开始` 到 `区间结束`
- 快捷窗口：最新收盘、近 `5 / 10 / 20 / 60 / 120` 根 K 线
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
- 日期容错：默认允许候选窗口在目标区间前后 `±5` 个交易日内整体平移，目标窗口本身不扩大
- 输出：同一段时间附近走势最像目标标的的 Top N 个股票/板块代理，并记录命中区间和日期偏移

用途：

- 回答“同一时间里，还有哪些标的走得像它？”
- 可限定成分指数、板块 ETF 成分、行业或概念范围
- 避免精确日期错位导致误判，同时不把后验收益窗口混入相似度计算

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

横截面搜索会额外输出：

- 候选命中区间开始、结束和日期偏移
- 后 3 / 5 / 10 根收益，收益从候选命中窗口结束日之后开始计算
- Top 结果的收盘价折线和单票 K 线聚合图

横截面日期容错计算已做向量化优化。`1000` 个标的、约 `120K` 行、`±5` 根容错的合成 profile 从约 `7.38s` 降到约 `1.81s`；后续修改不要退回逐偏移窗口逐个调用 `z_normalize` 的实现。

## 3. 安装

```bash
cd /Users/a1234/Desktop/ashare-cross-sectional-similarity
python -m pip install -r requirements.txt
```

Qt 桌面壳和小范围数据 API 是可选增强：

```bash
python -m pip install ".[desktop,server]"
```

如果要生成 macOS 双击启动的 `.app`，再安装打包依赖：

```bash
python -m pip install ".[desktop,server,packaging]"
```

OpenBB 抓取链路是可选增强，不作为默认依赖。需要使用时再安装：

```bash
python -m pip install openbb openbb_akshare
python -c "import openbb; openbb.build()"
```

TDX 抓取链路复用本机通达信量化终端的 `tqcenter`，不新增 pip 依赖。使用前需安装并登录通达信终端，并让程序能找到 `PYPlugins/user` 目录：

```bash
export TDX_TQCENTER_PATH=/path/to/TdxInstall/PYPlugins/user
```

## 4. 数据要求

默认读取本地 parquet；缺数据时可以在本库页面或 CLI 中触发下载。下载动作默认调用原 `trend-backtest` 的 `scripts/update_data.py`，数据源、TDX/AkShare 路由和落地目录仍以原库配置为准。

如果使用 `--download-engine akshare`，程序会直接通过 AkShare 抓取 `1d` 日线，并按本库目录结构直接写入 parquet。分钟线请使用 TDX。

如果使用 `--download-engine openbb`，程序会通过 OpenBB 的统一接口抓取行情，并按本库目录结构直接写入 parquet。A 股默认使用 `openbb_akshare` 扩展。

如果使用 `--download-engine tdx`，程序会直接导入本机通达信 `tqcenter`，调用 `tq.get_market_data` 抓取 `1d/30m/15m/5m/1m` K 线，并写入同一套本地 parquet。

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

Streamlit 页面会缓存本地行情读取结果。缓存键包含目标 parquet 文件的修改时间和大小，因此页面内外更新 parquet 后，再刷新页面会自动读取新文件，不需要手动重启服务。

## 4.1 小范围数据 API 与 Qt 桌面壳

推荐把 TDX / AkShare 抓取统一收口到本地或局域网内的数据 API，再让 Qt 应用消费同一套本地 parquet。API 默认只做小范围请求，避免误触发全市场下载。

启动数据 API：

```bash
ashare-xsec-data-api \
  --host 127.0.0.1 \
  --port 8765 \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --tdx-path /path/to/TdxInstall/PYPlugins/user
```

读取本地数据：

```bash
curl "http://127.0.0.1:8765/api/v1/bars?symbols=000001.SZ&start=2024-01-01&end=2024-01-31&source=local"
```

用 AkShare 自动补齐后再返回：

```bash
curl "http://127.0.0.1:8765/api/v1/bars?symbols=000001.SZ&start=2024-01-01&end=2024-01-31&source=akshare"
```

用本机 TDX 补齐后再返回：

```bash
curl "http://127.0.0.1:8765/api/v1/bars?symbols=000001.SZ&start=2024-01-01&end=2024-01-31&source=tdx"
```

启动 Qt 桌面壳：

```bash
ashare-xsec-sim-qt
```

桌面壳顶部可以选择两种模式：

- `本地 parquet`：不经过 HTTP，直接读取本机行情目录，适合离线研究。
- `数据 API`：调用 `ashare-xsec-data-api`，再选择 `local / akshare / tdx`。其中 `akshare` 会自动补数据，`tdx` 要求已配置本机通达信 `PYPlugins/user` 路径。

生成 macOS 应用包：

```bash
python scripts/build_macos_qt_app.py
```

产物位置：

```text
dist/A股相似阶段.app
```

打包脚本会生成应用图标，并使用 `packaging/macos/ashare_xsec_sim_qt.spec` 固定 Qt 入口和 bundle 信息。

生成 Win11 桌面版 `.exe` 需要在 Windows 侧执行。Parallels 共享目录下的已验证基线：

```powershell
cd C:\Mac\Home\Desktop\ashare-cross-sectional-similarity
C:\Users\Public\venvs\ashare-xsec-sim\Scripts\python.exe -m pip install ".[desktop,server,packaging]"
C:\Users\Public\venvs\ashare-xsec-sim\Scripts\python.exe scripts\build_windows_qt_app.py
```

产物位置：

```text
dist\windows\A股相似阶段\A股相似阶段.exe
```

Win11 侧使用 TDX 时，先确认通达信已启动并登录，再把 `TDX_TQCENTER_PATH` 指向实际 `PYPlugins\user` 目录：

```powershell
$env:TDX_TQCENTER_PATH="C:\path\to\TdxInstall\PYPlugins\user"
C:\Users\Public\venvs\ashare-xsec-sim\Scripts\python.exe -m pytest tests\test_tdx_source.py tests\test_download_all_a_daily.py
```

如果 `numba` 或 `stumpy` 在 Win11 导入时报 `DLL load failed while importing _typeconv`，先安装 Microsoft Visual C++ x64 Runtime，再重跑测试，不要先改业务代码。

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

- 默认 `--download-engine trend` 只把参数转发给原库 `scripts/update_data.py`。
- 具体支持哪些周期、使用 AkShare 还是 TDX、落到哪个目录，以原 `trend-backtest/config/data_source.yaml` 和原脚本实现为准。
- 如需指定原脚本 provider，可加 `--provider akshare` 或 `--provider tdx`。

也可以用原生 AkShare 直接写入本地 parquet：

```bash
python -m ashare_cross_section_similarity download \
  --download-engine akshare \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --timeframe 1d \
  --symbols 300750.SZ,000001.SZ,600519.SH \
  --start 2024-01-01 \
  --end 2024-03-31
```

也可以直接用本机 TDX 写入本地 parquet：

```bash
python -m ashare_cross_section_similarity download \
  --download-engine tdx \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --timeframe 1d \
  --provider /path/to/TdxInstall/PYPlugins/user \
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

脚本会用 AkShare 获取当前全 A 股票列表，并默认补入常用指数代理（上证、深成指、创业板、中证1000、沪深300、中证500），按批次委托本库下载器抓取 `1d/qfq` 日线，并在每批结束后复查本地 parquet 覆盖情况，日志写入 CSV。Streamlit 页面也提供 `TDX 全量日 K 线更新` 入口，默认跳过已覆盖区间，只补缺文件、覆盖不足、区间无数据或读取失败的标的。

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

使用本机 TDX 更新全 A 日线：

```bash
python scripts/download_all_a_daily.py \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --start 1990-01-01 \
  --end 2026-05-19 \
  --adjust qfq \
  --download-engine tdx \
  --provider /path/to/TdxInstall/PYPlugins/user \
  --batch-size 100 \
  --skip-available \
  --output outputs/all_a_daily_tdx_update_log.csv
```

常用选项：

| 参数 | 含义 |
| --- | --- |
| `--skip-available` | 下载前跳过本地已覆盖区间的股票 |
| `--no-indexes` | 只下载股票，不额外补常用指数代理 |
| `--extra-symbols` | 额外下载指数、ETF 或代理标的，逗号或换行分隔 |
| `--limit` | 调试时只下载前 N 个股票 |
| `--sleep` | 批次之间暂停秒数，避免数据源限流 |
| `--download-engine akshare` | 改用原生 AkShare 直接写入本地日线 parquet |
| `--download-engine openbb` | 改用 OpenBB 直接写入本地 parquet |
| `--download-engine tdx` | 改用本机 TDX 直连写入本地 parquet，`--provider` 填通达信安装目录、`PYPlugins` 或 `PYPlugins/user` |

### 5.5 导入自定义价格数据

也可以不走下载器，直接导入用户准备好的 `csv / parquet` 价格数据。文件需要符合本库价格数据规范：

```text
date, symbol 或 stock_code, open, high, low, close
```

可选列：

```text
volume, amount
```

如果文件只包含单一标的且没有 `symbol / stock_code` 列，可以用 `--fallback-symbol` 指定代码：

```bash
python -m ashare_cross_section_similarity import-data \
  --input examples/my_prices.csv \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --timeframe 1d \
  --adjust qfq \
  --fallback-symbol 000001.SZ
```

导入后会按本库目录结构写入：

```text
<data-root>/<adjust>/<symbol>.parquet
```

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
| `--exclusion-bars` | 排除当前窗口附近样本，避免“紧邻窗口”的阶段被误判为相似 |
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

页面有三个工作台：

1. `历史时序相似`：同一标的自己的历史相似阶段回溯。
2. `横截面相似`：同一时间窗口内从指定范围找相似标的。
3. `走势复盘`：基于本地 K 线生成排序复盘、锐评和视频脚本卡片；生成前可选择 `默认复盘` 或 `AI 复盘`，结果区只展示一个来源。

通用操作顺序：

1. 填本地行情目录。
2. 填原 `trend-backtest` 仓库路径。
3. 选择周期，默认日线。
4. 选择下载引擎：默认调用原库 `update_data.py`，也可选择 AkShare、OpenBB 或 TDX 本地。
5. 在对应工作台填写目标代码、窗口或区间；横截面工作台还需填写搜索范围。
6. 如需使用自有行情，在左侧 `上传自定义价格数据` 中导入 `csv / parquet`。
7. 查看或点击数据检查。
8. 如缺数据，点击下载或更新；历史时序和横截面使用同一套覆盖检查与下载修复逻辑。
9. 下载过程中可看任务状态、总数、已完成、失败、剩余、当前批次和当前标的；如需暂停，点击 `暂停下载`，系统会在当前批次结束后停住，点击 `继续下载` 从下一批接着跑。
10. 运行搜索。
11. 查看结果表、统计计量、走势图和 K 线图；横截面工作台会展示目标与相似标的收盘价折线图、单票 K 线聚合图，并可下载 CSV。

## 走势复盘来源与 DeepSeek V4

页面里的 `走势复盘` 先选择来源：

- `默认复盘`：使用本地 K 线和对标统计，研究端按“市场总环境、排序总表、逐个锐评、关键转折点复盘、明日验证”输出；视频端按 `夯爆了 > 人上人 > 立棍单打 > 刷子 > 混子 > NPC > 拉完了` 打标签，每个标的三句话封顶，不写“明天”。
- `AI 复盘`：使用同一份本地证据包调用 DeepSeek V4，按同一框架输出 `复盘 / 分析 / 锐评 / 视频脚本卡片`。页面不会同时展示默认复盘和 AI 复盘，AI 视频卡片也使用同一套标签和颜色。

```bash
export DEEPSEEK_API_KEY=your_api_key
streamlit run streamlit_app.py
```

CLI 可只生成证据包，也可调用 DeepSeek：

```bash
python -m ashare_cross_section_similarity review \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --target-symbol 000852.SH \
  --start 2026-01-01 \
  --end 2026-05-15 \
  --evidence-only \
  --output outputs/review_evidence.json
```

```bash
python -m ashare_cross_section_similarity review \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --target-symbol 000852.SH \
  --start 2026-01-01 \
  --end 2026-05-15 \
  --model deepseek-v4-flash \
  --output outputs/review.json
```

调用 DeepSeek 且未通过环境变量、CLI 参数或页面输入提供 API Key 时，会明确报错；使用 `--evidence-only` 不会调用 DeepSeek。

## 9. Docker 服务

本分支提供 Docker 服务化运行方式，默认把本机 `/Users/a1234/Desktop/trend-backtest/data` 挂载到容器内 `/data`，并把行情根目录设为 `/data/market/daily`。

启动：

```bash
docker compose up --build
```

浏览器打开：

```text
http://localhost:8502
```

如需改端口：

```bash
ASHARE_PORT=8510 docker compose up --build
```

如需挂载其他行情目录，例如自建数据目录：

```bash
ASHARE_HOST_DATA_DIR=/path/to/trend-backtest/data docker compose up --build
```

在 Docker 服务里有两种使用自定义价格数据的方式：

1. 页面左侧 `上传自定义价格数据`，上传符合规范的 `csv / parquet`。
2. 把已有 parquet 放入挂载目录的 `market/daily/qfq/`，文件名使用规范化代码，如 `000001.SZ.parquet`。

## 10. 输出字段解释

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

## 11. 当前边界

- 历史时序搜索和横截面搜索分开运行，不自动合成总评分。
- 指数成分、行业板块、概念板块来自 AkShare 当前接口，不保证历史成分时点准确。
- ETF 成分暂不自动抓取，建议先用 `--universe-file` 输入 ETF 持仓或自定义成分。
- 默认数据抓取仍可按原 `trend-backtest` 处理；小范围 API 和 Qt 桌面壳优先使用原生 AkShare 或本机 TDX。OpenBB 仍保留为可选增强。
- 结果是研究工具，不是买卖建议。

## 12. 开发验证

```bash
python -m pytest -q
python -m ruff check .
python -m py_compile streamlit_app.py ashare_cross_section_similarity/*.py scripts/download_all_a_daily.py
git diff --check
```

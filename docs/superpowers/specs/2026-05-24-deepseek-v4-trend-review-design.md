# DeepSeek V4 走势复盘设计

日期：2026-05-24

## 背景

当前工具已经具备两条核心研究链路：

- 历史时序相似：同一标的在自身历史里寻找相似走势，并输出后验收益、回撤和浮盈。
- 横截面相似：同一时间窗口内，在指定股票范围中寻找走势相似标的，并输出命中区间、日期偏移和后验收益。

下一阶段目标是接入 DeepSeek V4，对时序、横截面和走势复盘做第一步识别与复盘，让工具从“自用搜索脚本”升级为“可交付投研工作台”。

## 目标

第一版新增一个统一的 `走势复盘` 能力：

1. 复用现有历史时序相似和横截面相似逻辑，不替换、不削弱原有能力。
2. 将搜索结果、K 线窗口、特征指标、后验统计和大小盘价差率整理为结构化证据包。
3. 调用 DeepSeek V4 生成固定结构的复盘结论。
4. 同时支持 Streamlit 页面和 CLI，后续可平滑封装成 API 服务。
5. 页面呈现从工具脚本感改为专业投研工作台风格。

## 非目标

第一版不做这些事情：

- 不做自动交易、调仓建议或买卖信号下单。
- 不接多供应商模型路由，先固定 DeepSeek V4。
- 不展示完整思考链，只展示最终复盘结论和证据引用。
- 不重写前端技术栈，不为了 UI 升级牺牲当前 Streamlit 可用性。
- 不把 API key、secret 或用户凭证写进源码或配置样例。

## DeepSeek V4 接入

默认配置：

- Base URL：`https://api.deepseek.com`
- 默认模型：`deepseek-v4-flash`
- 高质量可选：`deepseek-v4-pro`
- 认证：优先读取 `DEEPSEEK_API_KEY`，页面可临时输入密钥但只保存在会话状态。
- 接口格式：OpenAI-compatible Chat Completions。
- 思考模式：默认启用模型能力，但不展示 `reasoning_content`；页面只显示 `content` 中的复盘结果。

依据 DeepSeek 官方文档，V4 支持 OpenAI-compatible 调用、JSON Output、Tool Calls、1M 上下文，`deepseek-chat` 和 `deepseek-reasoner` 属于兼容旧名，后续会废弃。因此第一版直接使用 `deepseek-v4-flash` / `deepseek-v4-pro`，不使用旧别名。

参考：

- https://api-docs.deepseek.com/
- https://api-docs.deepseek.com/quick_start/pricing
- https://api-docs.deepseek.com/guides/thinking_mode
- https://api-docs.deepseek.com/guides/json_mode

## 模块设计

新增模块建议：

- `ashare_cross_section_similarity/trend_review.py`
  - 组织复盘请求参数。
  - 调用历史时序搜索与横截面搜索。
  - 生成结构化证据包。
  - 解析和校验复盘输出。

- `ashare_cross_section_similarity/deepseek_client.py`
  - 封装 DeepSeek V4 调用。
  - 处理 API key、base URL、模型名、timeout、thinking 参数。
  - 失败时保留原始错误摘要，不静默降级。

- `ashare_cross_section_similarity/review_prompt.py`
  - 管理系统提示词和输出 JSON schema。
  - 固定输出字段，减少模型自由发挥。

现有模块继续保持职责：

- `history.py` 只负责历史相似搜索。
- `similarity.py` 只负责横截面相似搜索。
- `data.py` 和 `downloader.py` 继续负责本地行情读取、检查和下载。
- `streamlit_app.py` 只做页面编排，不承载复盘核心逻辑。

## 证据包结构

证据包必须可追溯，至少包含：

- 任务参数：目标代码、区间、周期、复权、搜索范围、算法、走势权重、日期容错。
- 当前窗口：起止日期、K 线数量、区间收益、波动率、最大回撤、趋势斜率、成交环境。
- 历史相似样本：Top N 样本的起止日期、相似度、路径距离、后验收益、最大回撤、最大浮盈。
- 横截面相似样本：Top N 标的、股票名称、命中区间、日期偏移、相似度、后 3/5/10 根收益。
- 分层统计：相似度分层后的后验收益分布和胜率。
- 大小盘背景：若本地有 `000852.SH` 和 `000300.SH`，附带目标窗口内大小盘价差率变化；缺失时显式标注未纳入。
- 数据限制：缺失数据、跳过标的、当前成分非历史成分等已知边界。

证据包优先 JSON 化，避免把大表全文塞给模型。默认只传 Top 10 历史样本和 Top 20 横截面样本，页面可调整。

## 复盘输出合同

DeepSeek V4 的输出固定为 JSON，再由页面渲染为可读文本。字段如下：

- `stage_identification`：阶段识别，判断是趋势延续、修复、加速、筑顶、筑底、震荡还是异常段。
- `history_readthrough`：历史相似样本解释，说明最关键的相似样本和后验分布。
- `cross_section_readthrough`：横截面共振解释，说明同时间相似标的集中在哪里。
- `forward_outcome`：后验收益与风险分布，不只写平均数。
- `counter_evidence`：反证与风险，列出与主判断相冲突的数据。
- `review_conclusion`：复盘结论，必须区分事实、推断和待验证事项。
- `evidence_refs`：引用证据包中的字段或样本编号，便于用户回查。
- `disclaimer`：固定研究辅助声明。

如果模型返回非 JSON 或字段缺失，系统应显示“模型输出格式不符合预期”，并保留原始返回文本供排查，不伪造成功结果。

## 页面设计

Streamlit 新增第三个 tab：`走势复盘`。

布局：

- 顶部：目标代码、区间、周期、复权、模型选择、运行按钮。
- 左侧参数区：复用历史和横截面的核心参数，避免重复填写过多细节。
- 中间证据区：展示当前窗口 K 线、历史相似结果、横截面相似结果、统计指标。
- 右侧或下方复盘区：展示 DeepSeek V4 结论，固定 6 块内容。
- 证据抽屉：展开后查看传给模型的证据包摘要。

视觉风格：

- 底色使用浅灰或白色。
- 主文字深灰，辅助文字中灰。
- 主操作使用蓝色。
- 行情涨跌只用红绿，不把红绿用于普通按钮。
- 卡片只用于结果块和复盘块，不做层层嵌套。
- 图表和表格保持高密度、可扫描，避免营销页式大标题和装饰渐变。

## CLI 设计

新增命令：

```bash
python -m ashare_cross_section_similarity review \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --timeframe 1d \
  --target-symbol 300750.SZ \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --universe-symbols 000001.SZ,600519.SH,002594.SZ \
  --model deepseek-v4-flash \
  --output outputs/trend_review.json
```

输出：

- JSON 复盘结果。
- 可选 Markdown 报告。
- 证据包摘要，便于复现。

CLI 失败时返回非 0 状态码，错误信息必须明确区分：数据问题、模型配置问题、API 调用失败、模型输出格式错误。

## 封装路径

第一阶段使用当前性价比最高的封装方式：

1. 核心能力沉淀为 Python 包。
2. Streamlit 作为本地工作台和第一版产品界面。
3. Docker Compose 继续服务 Mac mini 全天运行。
4. CLI 支持定时任务和批量报告。

第二阶段再考虑：

- FastAPI 封装 `/review` 接口。
- 域名和反向代理。
- 用户鉴权和任务队列。
- 更正式的前端。

这个顺序能最快把 DeepSeek V4 复盘跑通，也不会把当前可用的历史/横截面工作台推倒重来。

## 错误处理与安全

- 缺少 `DEEPSEEK_API_KEY` 时，页面和 CLI 都明确提示配置方式。
- API key 不写入源码、README 示例或输出文件。
- DeepSeek API 失败时不自动换模型，不静默 fallback。
- 模型输出格式错误时不伪造结构化结果。
- 证据包中不加入用户本地路径以外的敏感信息。
- 输出中固定声明：结果仅用于研究复盘，不构成投资建议。

## 验证

最小验证范围：

- 单测：
  - 证据包构建。
  - DeepSeek 客户端参数组织，使用 fake client，不真实消耗 API。
  - JSON 输出解析和格式错误处理。
  - CLI `review` 参数解析。
  - Streamlit 复盘区辅助格式化函数。

- 命令：

```bash
pytest -q
ruff check .
python -m py_compile streamlit_app.py ashare_cross_section_similarity/*.py scripts/download_all_a_daily.py
git diff --check
```

- 页面验证：
  - 启动 `streamlit run streamlit_app.py`。
  - 打开本地页面。
  - 检查标题 `A股相似阶段搜集`、三个 tab、`走势复盘` tab、DeepSeek V4 配置区和复盘输出区可见。

## 验收标准

第一版完成必须同时满足：

1. 现有历史时序和横截面能力仍可运行。
2. `走势复盘` tab 可以构建证据包。
3. 配置 DeepSeek API key 后可以调用 `deepseek-v4-flash` 生成复盘。
4. 未配置 API key 时有明确错误，不伪装成功。
5. CLI `review` 可以输出 JSON 结果。
6. 复盘结论引用证据包样本或字段，用户能回查来源。
7. 测试、lint、py_compile 和 diff check 通过。
8. 页面视觉符合投研工作台风格，避免花哨装饰和过度单色主题。

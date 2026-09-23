# 图书检索排序 Agent 实现计划

> **面向 AI 代理的工作者：** 在当前会话中按任务顺序内联执行本计划；每个任务先写失败测试，再实现并验证。由于仓库有用户未提交文档，所有 Git 操作只暂存本任务明确列出的文件。

**目标：** 做成可复现的图书候选检索、用户偏好排序、详情核验和 LLM 有据回答链路，并分别评测推荐基线与 20–30 条真实模型 Agent 案例。

**架构：** 新建 `book_agent` 本地应用包，隔离现有历史实验模型。目录检索、按用户历史/画像排序、详情读取作为三个只读工具；Agent 通过 OpenAI Responses API 的 function tools 执行显式工具调用循环。离线评测用固定时间切分和热门度/ItemCF 基线；真实模型评测使用 JSONL 案例并保存脱敏 trace。

**技术栈：** Python 3.12+、pandas、scikit-learn TF-IDF、OpenAI Python SDK Responses API、pytest。

---

## 文件职责

- 创建 `book_agent/data.py`：加载目录、用户、交互 CSV；规范日期/ID/类别字段；生成按时间排序的训练与留出交互。
- 创建 `book_agent/catalog.py`：离线字符 n-gram TF-IDF 检索、元数据过滤和图书详情查询。
- 创建 `book_agent/ranking.py`：热门度与 ItemCF 基线、用户候选重排及可解释分项。
- 创建 `book_agent/metrics.py`：Recall@K、NDCG@K、MRR@K 及共同评测集聚合。
- 创建 `book_agent/tools.py`：三个只读工具、参数验证、JSON Schema 和结构化结果。
- 创建 `book_agent/providers.py`：Provider 协议、可控 FakeProvider、OpenAI Responses API 适配器。
- 创建 `book_agent/agent.py`：工具调用循环、轮数限制、证据约束和运行 trace。
- 创建 `book_agent/agent_eval.py`：案例加载、trace 检查、约束/事实/异常评分与结果持久化。
- 创建 `book_agent/cli.py`：本地 demo、推荐评测和真实模型 Agent 评测命令。
- 创建 `book_agent/__init__.py`：包初始化。
- 创建 `tests/book_agent/` 下的单元和集成测试：数据切分、检索、排序、指标、工具、调用循环、评测评分。
- 创建 `data/agent_eval_cases.jsonl`：25 条中文案例，覆盖约束、多步个性化、简单检索、澄清、无结果和异常。
- 创建 `requirements-agent.txt`：本地 demo、测试和真实模型评测的轻量依赖。
- 创建 `.env.example` 和 `.gitignore`：列出模型配置变量并忽略真实凭证、缓存与评测输出。
- 创建 `AGENT_DEMO.md`：安装、运行、评测、案例报告与简历表述说明。
- 修改 `README.md`：保留现有未提交内容，在项目入口处链接 Agent 演示文档。该文件已有用户未提交更改，因此修改后保持未暂存、不纳入本轮提交。

## 任务 1：数据加载、时间切分与排序指标

**测试：** 创建 `tests/book_agent/test_data.py` 与 `tests/book_agent/test_metrics.py`。用小型临时 CSV 验证 UTF-8 BOM、分号类别、重复用户交互及同时间戳稳定排序；用手算列表验证 Recall/NDCG/MRR、空候选与无相关物品行为。

**实现：** 创建 `book_agent/__init__.py`、`book_agent/data.py`、`book_agent/metrics.py`。`load_demo_data(data_dir)` 返回按 ID 索引的图书/用户记录和规范化交互；`temporal_leave_last_out(interactions)` 对每位至少 2 条交互的用户留出最后一条并返回训练集、测试目标和排除用户数。所有基线只能从训练集拟合。

**验证：** `python -m pytest -q tests/book_agent/test_data.py tests/book_agent/test_metrics.py`；预期所有用例通过。

## 任务 2：目录检索、推荐基线与用户候选排序

**测试：** 创建 `tests/book_agent/test_catalog.py` 和 `tests/book_agent/test_ranking.py`。验证标题/关键词中文查询、类别/关键词/价格过滤、未知图书、已交互候选排除、热门度稳定并列顺序、ItemCF 排序、无历史/未知用户回退，以及分项分数能解释排序。

**实现：** 创建 `book_agent/catalog.py` 与 `book_agent/ranking.py`。检索索引使用字符 n-gram TF-IDF 建立，避免逐本下载图片或访问网络。`rank_candidates_for_user(user_id, query, candidate_ids, limit)` 仅能重排传入候选；返回 query 相关度、训练历史 ItemCF 亲和度、用户画像类别/关键词匹配分和最终分。未知用户回退到 query 相关度，候选少时返回可用数量。

**验证：** 先确认两个测试文件在实现前因模块/行为缺失而失败，再运行 `python -m pytest -q tests/book_agent/test_catalog.py tests/book_agent/test_ranking.py`。

## 任务 3：只读工具契约与证据型结果

**测试：** 创建 `tests/book_agent/test_tools.py`。检查三个工具的 JSON Schema、缺失/非法参数、无效价格范围、未知 ID、详情批量查询、详情字段来源，以及排序工具不能返回未传入候选。

**实现：** 创建 `book_agent/tools.py`。只注册 `search_catalog`、`rank_candidates_for_user`、`get_book_details`；每个响应包含 `item_id`，检索返回分数，排序返回分数分解，详情只返回 CSV 实际存在的字段。目录不含星级评分，工具 schema 不提供评分字段。

**验证：** `python -m pytest -q tests/book_agent/test_tools.py`。

## 任务 4：Agent 编排、真实工具调用协议与 trace

**测试：** 创建 `tests/book_agent/test_agent.py`，用假 Provider 排队返回工具调用，验证完整序列 `search_catalog → rank_candidates_for_user → get_book_details → final`；另测单步搜索、未知工具、坏 JSON 参数、工具异常、无候选、最大轮数、调用 ID 配对和 answer 中书籍 ID/证据必须来自详情输出。

**实现：** 创建 `book_agent/providers.py` 与 `book_agent/agent.py`。先定义可注入 Provider 接口和 FakeProvider；实现 allowlist 分发、严格参数解析、最大工具轮数、工具输出回传、答案证据 ID 检查与 trace（时间、模型名、工具/参数、结果 ID、错误、token/耗时若有）。日志/trace 不包含 API key。

**验证：** `python -m pytest -q tests/book_agent/test_agent.py`，FakeProvider 全程离线。

## 任务 5：OpenAI Responses API 适配器与配置

**测试：** 创建 `tests/book_agent/test_provider.py`，对 SDK 响应对象构造工具调用项、最终文本、拒绝/空输出和 API 错误的标准化行为做测试；Mock transport 不发网络请求。

**实现：** 实现 `OpenAIResponsesProvider`，使用严格 JSON Schema 函数工具并按 Responses API 的 `call_id` 回传函数输出；支持环境变量 `OPENAI_API_KEY`、`OPENAI_MODEL`、可选 `OPENAI_BASE_URL`。创建 `requirements-agent.txt` 和 `.env.example`，并从 `AGENT_DEMO.md` 指明真实凭证不入库。使用现有 `requests`/OpenAI SDK 依赖的选择在实现前按官方文档核对。

**验证：** `python -m pytest -q tests/book_agent/test_provider.py tests/book_agent/test_agent.py`；缺少 API key 时 Provider 初始化给出明确错误，测试不读取真实凭证。

## 任务 6：推荐离线评测与 Agent 案例评分

**测试：** 创建 `tests/book_agent/test_agent_eval.py`，用人工构造 trace 覆盖工具顺序、约束满足、详情依据、空结果/澄清、未知用户及轨迹缺失的正确计分和错误计分。

**实现：** 创建 `book_agent/agent_eval.py` 与 `data/agent_eval_cases.jsonl`。案例固定 25 条，字段包含 case ID、用户请求、预期阶段、过滤条件、预期结果类型和必需证据类别。包括要求按用户历史个性化的多步案例；不得引用目录不存在的评分属性。实现真实模型运行命令，逐条写入脱敏 JSONL trace，并生成工具选择正确率、约束满足率、字段依据率、异常处理率、轨迹完整率及端到端成功率。

**验证：** `python -m pytest -q tests/book_agent/test_agent_eval.py`；使用 FakeProvider 跑完整 25 条案例，验证报告汇总数与逐例结果一致。

## 任务 7：命令行、端到端验证和失败改进

**实现：** 创建 `book_agent/cli.py`。提供 `demo`、`evaluate-recommender`、`evaluate-agent --provider openai --cases data/agent_eval_cases.jsonl` 子命令。生成固定种子、按用户时间留一的热门度/ItemCF 报告；真实模型评测对 25 条案例逐条运行并保存每条 trace、失败摘要、总耗时和模型配置（不存密钥）。

**真实模型执行：** 先运行完整 25 条初始评测，分类失败（工具顺序、限制条件、证据引用、空结果/澄清）；只针对失败改进系统提示或工具说明，再用相同案例完整重跑一次。保留两轮数据和简短误差分析。若当前环境未配置 `OPENAI_API_KEY`，停止于真实调用前并请求用户配置，不以 FakeProvider 结果冒充真实模型结果。

**验证：** `python -m pytest -q tests/book_agent`；之后运行推荐评测、离线 Agent 全案例测试，并在有凭证时运行两轮真实评测。

## 任务 8：文档、简历证据与最终回归

**实现：** 创建 `AGENT_DEMO.md`，修改 `README.md` 增加链接，说明安装、数据格式、精确命令、Agent 工具流程、两个分开的评测、模拟数据限制、配置/API 成本和真实评测报告路径。只填实际运行结果；未运行真实模型时保留明确的待跑状态。

**验证：** 按文档在当前可用 Python 环境执行安装检查、`python -m pytest -q tests/book_agent`、推荐评测和 FakeProvider 端到端 demo；若有 key 再执行完整真实评测并核对 JSONL 条数为 25。

## 实施纪律

- 每项行为先写测试并运行至预期失败，再实现最小改动并运行通过。
- 每项任务提交时只包含该任务列出的代码/测试文件；README 的用户改动及本轮 README 链接都保持未暂存，其他用户未提交文档不暂存、不清理用户已有缓存或文件。
- 实际用户行为数据是模拟数据；报告、README 和简历示例不得把内部指标写成业务收益。
- 真实模型评测需要可用的 OpenAI API 凭证和模型名。当前工作环境未发现相关环境变量；代码完成后再请求配置方式，不读取或记录密钥值。

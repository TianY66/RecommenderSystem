# 图书检索与推荐 Agent

这个本地演示把目录检索、用户偏好排序和详情核验连成一个工具调用流程。LLM 负责解析自然语言并选择工具；候选召回、排序和图书事实来自仓库 CSV 与 Python 代码。工具只读，不连接 MySQL，也不会下载图书图片。

## 安装

在仓库根目录运行：

```powershell
python -m pip install -r requirements-agent.txt
Copy-Item .env.example .env
```

真实模型评测需要在本机 `.env` 中设置 `OPENAI_API_KEY` 和 `OPENAI_MODEL`。`OPENAI_BASE_URL` 可选。请勿把 `.env` 提交到 Git，也不要把 API key 粘贴到聊天或评测报告中。命令会读取本机环境变量或仓库根目录 `.env`；报告只记录模型名、token 用量、耗时、工具参数和工具结果。

## 本地检索与排序演示

```powershell
python -m book_agent.cli demo `
  --query 算法 `
  --user-id U00001 `
  --category 科技 `
  --keyword 算法 `
  --max-price 150 `
  --limit 2
```

演示先用字符 n-gram TF-IDF 召回目录候选，再按该用户的交互历史和画像排序，最后读取详情。详情工具仅返回书名、作者、类别、关键词、简介和价格等回答所需字段。未知用户会回传用户上下文不可用的状态，Agent 不应声称结果经过了个性化。

## 推荐离线评测

```powershell
python -m book_agent.cli evaluate-recommender --ks 5 10 20
```

评测按每位用户的时间顺序留出最后一条不同图书交互，只在其余历史上拟合热门度与 ItemCF。默认报告写到被 Git 忽略的 `reports/recommender_eval.json`。

本地 94,934 条交互的实测切分包含 84,375 条训练交互、9,989 位测试用户和 10 位不足两本不同图书而跳过的用户。测得结果如下：

| 方法 | Recall@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: |
| 热门度 | 0.0061 | 0.0021 | 0.0010 |
| ItemCF | 0.0123 | 0.0045 | 0.0024 |

这是仓库现有交互数据上的离线留一结果，不代表线上转化或真实用户收益；这批交互数据主要是模拟数据。指标用于建立可复现的推荐基线，不应当写成业务效果提升。

## 真实模型 Agent 案例评测

25 条 JSONL 案例覆盖多步个性化、类别/关键词/价格筛选、双本推荐、未知用户、需要澄清、无结果和目录不含星级评分等情况。运行：

```powershell
python -m book_agent.cli evaluate-agent `
  --provider openai `
  --cases data/agent_eval_cases.jsonl `
  --output reports/agent_eval_initial.json
```

评测会为每条请求保存脱敏后的工具轨迹和回答，并汇总：

- 工具序列准确率
- 筛选条件满足率
- 书名/目录标签/明确价格陈述的一致率
- 详情依据率与未知用户个性化声明准确率
- 澄清和无结果处理率
- 轨迹完整率与端到端成功率

工具调用上限为每轮 6 次；API 错误、参数错误和未核验书目引用都会被记录。目录没有星级评分，所以案例要求对这类请求澄清，不会编造评分。

真实模型案例报告应由你配置的模型实际运行生成。当前仓库只完成了离线工具协议和评测框架的模拟器全案例回归；它不等于真实模型成绩。真实评测完成后，可先检查失败类别和 trace，再调整 Agent 提示词，并用相同的 25 条案例完整重跑一次：

```powershell
python -m book_agent.cli evaluate-agent `
  --provider openai `
  --cases data/agent_eval_cases.jsonl `
  --output reports/agent_eval_refined.json
```

模型 API 的实际费用和可用模型取决于本机账号配置；运行前请在服务侧设置支出限制。不要把尚未真实运行的模拟器分数写成真实模型效果。

## 简历表述草稿

完成真实模型评测后，可将实际结果填入下面草稿；未跑之前不要保留占位数字或声称通过了真实模型验证：

> 构建图书检索与推荐 Agent，编排目录召回、用户偏好排序和详情核验 3 个只读工具；实现严格 JSON Schema 参数校验、候选来源约束、引用书名核验和可追踪调用日志，并用 25 条真实模型对话案例评测工具选择、筛选条件、事实一致性及失败处理。实测端到端成功率 **[填写实际报告值]**，对失败轨迹进行分类后迭代提示词并复测。

推荐系统离线指标可以单独陈述为本地模拟交互数据的留一评测结果，不要把它和 Agent 案例成功率混为一谈。

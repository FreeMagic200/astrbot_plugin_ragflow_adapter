# Changelog

## [Unreleased / v0.4] - 2026-03-16

### 新增
- `main.py` 新增 `@filter.regex(r"(?i)^/ask\S")` handler，支持 `/ask后无空格` 直接跟问题内容的模糊匹配（如 `/ask什么是DNA`）
- `src/helpers.py` 新增 `RETRIEVAL_NO_RESULTS` 哨兵常量，用于区分"API 成功但知识库为空"与"配置错误/网络错误"两种失败场景
- 检索结果为空时，触发兜底 system prompt，强制 LLM 在回答开头明确告知用户未检索到相关信息，避免"隐性幻觉"

### 改进
- `src/rewriter.py` 将查询重写策略从"重写为完整问句"改为"提取核心实体/关键词"
  - 自动去除"判断正误"、"请解释"、"是否正确"等对向量检索（Embedding）和关键词检索（BM25）无意义的指令词
  - 输出空格分隔的关键词串，而非完整疑问句，显著降低检索噪声
  - 增加三组 few-shot examples 强化提示稳定性
- `src/helpers.py` RAG 注入模板加入输出结构约束
  - 要求 LLM 先在 `<thinking>` 标签内完成推理，再输出最终结论，防止"开头说错误、结尾说正确"的前后矛盾
  - 参考资料未覆盖的知识点必须标注【未在知识库中检索到相关信息，以下基于通用知识库作答】
- `main.py` 将 `kb_query` 核心逻辑抽取为 `_run_kb_query()` 异步生成器，命令 handler 与 regex handler 共享同一套处理流程

### 修复
- 同步远程 3 个提交（将命令从 `/kb` 改为 `/ask`，扩充别名 KB/ASK/Ask/知识库/提问/疑问）

---

## [v0.3] - 2026-03-14

- `4c63814` 修复：将命令组模式改回单一命令模式，正确处理 `/ask <query>` 参数解析
- `e6781b1` 重构为命令组模式，使用 AstrBot 内置 handler 机制替代全局 LLM 钩子
- `a654649` 添加 `trigger_prefix` 配置，支持按消息前缀触发 RAGFlow 检索
- `ebcf5bd` 添加 `ragflow_request_timeout` 配置项
- `7961fb6` 添加知识库来源标注功能（chunk 附带文档名称）

---

## [v0.2] - 2025-11-27

- `a8d17c6` Merge PR #1 from BiDuang/main
- `86cc33a` 添加重排序模型（rerank）和跨语言检索支持；重构查询重写管理器
- `7f7456e` 添加 UMO 白名单配置，限制特定群组使用 RAGFlow 功能

---

## [v0.1] - 2025-10-20 ~ 2025-10-31

- `1403d34` 实装会话消息计数器
- `6f437e4` 规划归档功能 TODO
- `0fe1a4b` 为查询重写添加错误重试与 fallback 机制
- `e47c49b` 添加查询重写管理器（UnifiedQueryRewriter）
- `5c6a518` 将辅助方法抽取到 `src/helpers.py`
- `88d816b` 添加归档相关配置项
- `82cfe2b` 优化查询 prompt，编写 README
- `32c9bb8` 基本功能可用
- `1ccc9db` 初始架子搭建
- `e89c9ed` 初步 API 探索

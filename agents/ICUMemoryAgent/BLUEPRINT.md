### ICUMemoryAgent 自然语言检索与规划执行蓝图

本蓝图规划将 `ICUMemoryAgent` 从“仅存储”升级为“可接收自然语言、进行意图识别与规划、调用工具/MCP、分步执行并多轮检索汇总答案”的智能检索体。

---

### 1) 背景与现状
- **数据入口**: `ICUDataIngestionAgent` 负责加载 ICU 原始事件，调用 `UMLSClinicalTranslator` 将中文事件专业化为英文并产出 Envelope（包含 `patient_id/event_id/timestamp/event_type/sub_type/event_content/raw`）。
- **存储**: `ICUMemoryAgent` 当前使用 `ScalableMemory` 将事件入库，支持：
  - 为每位患者维护独立向量库（以 `patient_id` 为键）
  - 维护全局向量缓存（按 `event_id` 复用向量，避免重复嵌入）
  - 基础读取：`get_by_id/get_all`、基于时间窗口和最近事件的遍历
- **不足**: 仅支持被动读取或简单遍历；无自然语言理解与规划；检索主要依赖单次向量相似度。

---

### 2) 目标与非目标
- **目标**
  - 接收自然语言问题（中文或英文），自动进行意图识别、槽位填充（时间窗/对象/条件）、查询规划与执行。
  - 通过一次或多次调用患者 `ScalableMemory` 的向量检索与元数据过滤，实现“多阶段候选召回 → 证据聚合 → 最终答复”。
  - 可选集成 MCP 与工具（药物标准化、时间窗解析、单位归一、UMLS 映射、验证器）以提升准确性与可解释性。
  - 产出结构化答案：结论 + 证据列表（相关事件/时间/来源）。
- **非目标**
  - 不在本阶段实现完整 EHR 读写或外部系统写操作；仅检索与回答。

---

### 3) 设计原则与约束
- **以患者为作用域**: 所有检索必须指定 `patient_id` 并在此作用域内执行。
- **检索优先秩序**: 先精确过滤（时间窗/类型/子类型/结构化条件），再向量相似度，必要时重复细化（多轮检索）。
- **可解释性**: 输出包含证据条目与推理要点，便于临床人员复核。
- **一致性与幂等**: 相同输入（问题+患者+时间）在无外部变化时尽量得到一致结果；对外提供可复现的 plan/trace。
- **日志与可观测性**: 使用 `agent_engine.agent_logger` 记录英文日志（组件、步骤、耗时、top_k、threshold、过滤条件、召回量）。
- **持久化规则**: 传入 `ScalableMemory` 的 `persist_dir` 视为最终存储目录，不再基于 `name` 追加子目录 [[memory:8183017]]。建议：为每名患者提供独立 `persist_dir` 目录（见 §8）。

---

### 4) 总体架构
- **NL Interface**: 入口方法接收自然语言问题与 `patient_id`。
- **Intent Classifier**: 识别意图类别与必要槽位（时间窗、对象、操作、限定条件）。
- **Planner**: 生成可执行计划（Plan），包含若干步骤（结构化过滤、候选召回、重排、证据聚合、回答生成）。
- **Tool Router**: 选择性调用工具/MCP（药物标准化、时间解析、UMLS 概念映射、单位归一等）。
- **Retriever**: 调用患者 `ScalableMemory` 的 `search`/`get_all`，支持元数据过滤与向量检索的多阶段组合。
- **Evidence Aggregator**: 对候选事件去重、归并（同一医嘱/同一时间点）、打分（相关性/时序/置信度）。
- **Answer Synthesizer**: 结构化总结与自然语言答案生成（包含证据编号和简述）。

---

### 5) 关键意图与策略
- **相关事件检索（about X）**
  - 将自然语言中 X 提炼为查询短语与同义词/关键词；必要时从中文翻译为英文以贴合已存事件语言。
  - 初步元数据过滤：`event_type/sub_type`（若话语中明确）、时间窗（若出现“最近/24小时/今天”等）。
  - 向量检索 top_k（如 20~50），阈值动态（如 0.3~0.5），结合关键词匹配加权重排。
  - 证据聚合：相同 `event_id`/相同时间点合并，保留最强证据与上下文。

- **是否存在类问题（Has/Did）如“24小时内是否服用xxx药物”**
  - 药物名标准化（MCP 或本地工具）映射至 RxNorm/通用名/商品名集合。
  - 时间窗解析（24h、过去一天、入科以来等）。
  - 元数据过滤优先：`event_type in {medication, order, administration}`；随后向量检索补充召回。
  - 设立证据充分性判据：若出现“给药/执行/签署医嘱”事件且时间在窗内，判定为“是”；若仅有“停止/拒绝”则判为“否”；证据不足标记为“不确定”。

- **统计/趋势类（labs/vitals）**
  - 解析目标指标（如 Lactate）、单位归一（mmol/L vs mg/dL）、时间窗与聚合函数（最大/最小/均值/最后一次）。
  - 以结构化过滤为主（`sub_type`/字段名/单位），向量检索作为兜底。

---

### 6) 执行流程（单次问答）
1. 接收输入：`patient_id`、`question`（可中文）。
2. 预处理：必要时将 `question` 翻译为英文（与已存内容语言一致），提取时间表达式与关键实体。
3. 意图识别与槽位填充：意图类型、时间窗、对象（药物/检验/手术/护理）、限定条件（频次、剂量等）。
4. 计划生成：拆解为步骤（Filter → Vector Search → Rerank → Aggregate → Answer）。
5. 候选召回：
   - 结构化过滤：`metadata_filter = {patient_id, event_type/sub_type, ts in window}`。
   - 向量检索：对重写后的短查询执行 `search(text, top_k, threshold, metadata_filter)` 多次迭代（必要时缩放阈值/扩容 top_k）。
6. 证据聚合与验证：去重、按时间排序、合并同源事件、冲突消解（停止医嘱 vs 执行医嘱）。
7. 生成答案：给出结论（Yes/No/Uncertain 或 列表/摘要），并附带证据（事件时间、类型、内容摘要、event_id）。
8. 返回结构：`{"answer": str, "confidence": float, "evidence": [..], "trace_id": str, "plan": [...]}`。

---

### 7) 检索与重排策略细化
- **多阶段召回**
  - Stage A：严格过滤（按 `patient_id`、时间窗、类型）；
  - Stage B：向量检索（多个查询子句：同义词/改写/关键短语）；
  - Stage C：关键字/规则命中（如“q8h/PRN/IV/停止/拒绝/给药”）。
- **重排与打分**
  - 综合得分 = 相似度分 + 规则命中分 + 时间接近度分 + 类型优先级分；
  - 输出前去重（同 event_id/同时间点）与聚合（同一医嘱的多条操作记录）。
- **空召回自愈**
  - 降低阈值、扩大时间窗、改写查询；
  - 若仍为空，返回“不确定”并附自愈轨迹。

---

### 8) 存储布局与 ScalableMemory 约定
- 为避免目录歧义，遵循：`persist_dir` 即为该库的最终目录，不再基于 `name` 追加子目录 [[memory:8183017]]。
- 建议：
  - 全局向量缓存：`agents/ICUMemoryAgent/database/vector_cache/` 作为 `persist_dir`；`name` 固定常量（例如 `icu_vector_cache`）。
  - 患者向量库：`agents/ICUMemoryAgent/database/patient/{patient_id}/` 作为 `persist_dir`；`name` 可固定常量或简短别名（不参与路径构成）。
- 元数据最小集合：`{"id": event_id, "patient_id": ..., "timestamp": ..., "event_type": ..., "sub_type": ..., "raw_content": ...}`。

---

### 9) 工具与 MCP 集成
- **内置工具（建议实现）**
  - TimeWindowResolver：解析“最近/24h/过去一周”等时间表达式，输出 UTC 时间窗。
  - MedicationNormalizer：同义词与 RxNorm 正规化（可先本地规则，后接 MCP）。
  - UnitNormalizer：单位统一与换算，避免错配。
  - EventFilter：按 `event_type/sub_type/metadata` 的布尔过滤与管道化组合。
  - MemorySearch：对接 `ScalableMemory.search/get_by_id/get_all` 的统一门面，支持批量查询与并行。
  - EvidenceAggregator：去重、合并、打分与裁剪。
  - AnswerFormatter：将结构化证据转为人类可读答案（保留证据引用）。
- **MCP 路由（可选）**
  - 通过 MCP 连接外部工具/服务：药物本体、单位/术语映射、下游检索代理等；
  - 统一工具注册/发现/调用协议（远端 schema 刷新、异常隔离、超时控制）。

---

### 10) Prompt 工程（关键模板）
- Intent Classification（few-shot）：输出 `intent_type` 与 `slots`（time_window, target_object, constraints）。
- Query Rewrite：将自然语言问题改写为 1~3 个检索短语（英文），并生成候选关键词。
- Plan Generation：给出步骤化计划（Filter → Search → Rerank → Aggregate → Answer）。
- Evidence-Grounded Answer：严格基于证据生成答案，不臆断；若证据不足必须输出“不确定”。

提示：保持温度低（如 0.0~0.2），强约束输出 JSON，易于解析与执行；中文输入优先翻译为英文短语以贴合存储语言。

---

### 11) 对外 API（建议）
- `async def natural_language_query(patient_id: str, question: str, now: Optional[datetime]=None, max_iterations: int = 3) -> Dict[str, Any]`
  - 输入：患者、问题、参考时间、最大迭代数。
  - 输出：见 §6 第 8 步结构。
- 内部拆分：`parse_intent` → `plan_actions` → `execute_plan`（在其中多次调用 `MemorySearch`）。
- 领域便捷接口：
  - `has_medication_within(patient_id, medication: str, hours: int) -> Dict[str, Any]`
  - `find_events_about(patient_id, topic: str, hours: Optional[int]) -> Dict[str, Any]`

---

### 12) 失败处理与降级策略
- 无 LLM/超时：仅执行结构化过滤 + 关键词近似匹配（简单规则），并显式标注置信度低。
- 空结果：逐步放宽阈值/扩大时间窗/改写查询（最多 N 次），否则返回“不确定”。
- 冲突证据：规则裁决（执行 > 开立 > 取消/拒绝），同时在答案里保留冲突说明。

---

### 13) 观测、日志与复现
- 日志统一使用 `agent_engine.agent_logger`，日志内容使用英文；记录 trace_id、每步输入输出摘要、耗时与召回量。
- 支持 `dry_run=True` 返回 plan 而不实际执行；支持 `debug=True` 返回详细 trace。
- 关键超参（top_k、threshold、迭代次数）可通过环境变量/构造参数配置，便于线下评估。

---

### 14) 里程碑
- M0 基线：
  - `natural_language_query` 主流程（意图识别 → 结构化过滤 → 向量检索 → 聚合 → 作答）；
  - 两个意图模板：相关事件检索、24h 用药判断；
  - 证据与 trace 输出。
- M1 增强：
  - 工具化：时间窗解析、药物标准化、单位归一；
  - 多子查询与重排；错误处理与空召回自愈。
- M2 集成：
  - MCP 路由与外部本体/工具接入；
  - 评估框架（精确率/召回率/覆盖率/平均响应时间）。
- M3 体验：
  - 更丰富的便捷接口与问题模板；
  - Top-N 证据可视化与溯源链接。

---

### 15) 示例用法（语义层面）
- “我需要查看一下可能与{event_content}有关的事件”：
  1) 意图：相关事件检索；2) 计划：过滤（患者+可选时间窗）→ 改写为英文短语 + 同义词 → 向量检索多次 → 聚合；3) 输出：事件列表 + 摘要。
- “我想知道病人24小时内有没有服用xxx药物”：
  1) 意图：用药存在性；2) 计划：时间窗解析（24h）→ 药物标准化（通用名/RxNorm）→ 过滤（medication/order/administration）→ 向量检索兜底 → 证据裁决；3) 输出：Yes/No/Uncertain + 证据。

---

此蓝图为实现提供结构化落地路径：先最小可用（M0），再逐步引入工具与 MCP，最终达到稳健、可解释、可扩展的临床检索体验。



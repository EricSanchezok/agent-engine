## ICUMemoryAgent 设计与检索算法说明

本文件描述 ICUMemoryAgent 的职责、数据结构、嵌入策略，以及“以事件为输入，检索最相关事件”的两段式检索算法（召回 + 重排）。本文为实现蓝图，便于后续工程化落地与迭代评估。

### 1. 目标与范围
- 目标：
  - 为每个 `patient_id` 构建独立的可持久化记忆库（向量检索 + 元数据），支持快速、可解释的相关事件检索。
  - 提供一个全局向量缓存，按 `event_id` 复用 embeddings，避免重复 token 成本。
  - 以“事件 → 相关事件”为核心检索类型，融合语义相似、时间邻近与临床规则，兼顾召回与精度。
- 非目标（本篇暂不展开）：
  - 自然语言问题（如“患者最近服过什么药？”）的 Planner/Tool 使用流程；CBR（Case-Based Reasoning）系统级记忆与跨患者检索将在后续文档中讨论。

### 2. 数据与存储
- 患者级数据库：
  - 名称即为 `patient_id`，底层为 `ScalableMemory`（DuckDB + HNSWlib 优先，自动回退）。
  - `item_id` 使用事件的 `event_id`，便于 O(1) 定位与跨组件引用。
  - `metadata` 至少包含：`patient_id`、`timestamp`、`event_type`、`sub_type`。后续会加入概念抽取产物（药物/疾病/检查/异常）与解析标记。
- 全局向量缓存（只按 `event_id` 存向量）：
  - 名称固定为 `icu_vector_cache`，用于缓存所有事件 embeddings。
  - 写入策略：首次遇到 `event_id` 嵌入并入库；再次使用时直接读取，避免再次嵌入。

### 3. 事件文本与嵌入策略
- 嵌入服务：Azure OpenAI `text-embedding-3-large`（可通过 `AGENT_ENGINE_EMBED_MODEL` 配置）。
- 嵌入文本推荐：默认仅使用 `event_content`，可选在前置加入简短类型标签（如 `[order]` 或 `[order/lab]`）。
  - 不建议将 `event_id` 与 `timestamp` 写入嵌入文本（不含语义、引入噪声、破坏稳定性）。
  - 类型标签（如 `[event_type/sub_type] content`）可在内容过短/结构化程度低时提高可分性。
  - 同时在 `metadata` 中保留 `event_type/sub_type`，以支持后续精确筛选。

### 4. 以“事件”为输入的检索：两段式（召回 + 重排）

给定 `patient_id` 与某个 `event_id`/事件对象，输出 TopK 最相关事件。核心是把“语义相似 + 时间关系 + 临床路径/规则”融合到一个统一得分中。

#### 4.1 召回层（高召回，低精度）
- 向量召回（Vector Recall）
  - 使用该事件的向量在患者级向量库进行相似度检索，取 `TopN_vec`（建议 N≈100）。
- 时间邻近召回（Temporal Recall）
  - 以事件时间为中心的窗口（如 ±24h）内的所有事件加入候选，强化时序相关。
- 类型优先召回（Type-Prior Recall，可选）
  - 基于类型映射关系，先加入潜在相关类型：
    - history → order、lab、exam
    - lab → order（复查/矫治/监测）、history
    - surgery → order/exam/nursing
    - order ↔ lab（用药监测/效果评估）
- 合并去重，得到候选集 C。

#### 4.2 重排层（高精度，带解释）
为候选集中每个事件计算总体得分：

  s = w1 · sim_vec + w2 · time_score + w3 · concept_overlap + w4 · rule_boost

各特征定义：
- sim_vec（语义相似度）
  - 余弦相似度；可直接由 `ScalableMemory.search` 返回的 similarity 使用。
- time_score（时间近邻分数）
  - 指数衰减：`time_score = exp(-|Δt|/τ)`，τ 建议 6–12 小时。
- concept_overlap（概念重合度）
  - 事件级概念集合的重合度（Jaccard/加权Jaccard）：
    - 概念包括：药物（含类别）、疾病/诊断、手术/操作、检查/化验项目、异常标记（↑/↓/阳性/阴性）。
    - 可对类别加权（如药物/异常更高权）。
- rule_boost（临床规则奖励）
  - 命中“临床语义关系规则”时累加奖励；方向性与权重可配置。
  - 类型先验矩阵 R[type_i, type_j] 作为基础加分（例如 history→order=0.6，lab→order=0.5，…）。

推荐初始权重：`w1=0.5, w2=0.2, w3=0.15, w4=0.15`；TopK=20；`TopN_vec=100`；时间窗口 ±24h。

### 5. 概念抽取（轻量规则/词典法）
为降低延迟与成本，优先使用本地规则/词典解析 `event_content`：
- 药物与类别：抗生素、PPI/护胃、镇痛/镇静、抗凝、利尿、电解质补充（补钾/钙/镁）、血制品等（中文商品名/通用名与同义词）。
- 疾病/诊断：从 `history` 段落的“入院诊断/拟诊/计划”中抽取关键疾病词（例如：脓毒症、SAH/蛛网膜下腔出血、肺挫伤等）。
- 手术/操作：surgery 事件与各类介入/穿刺/置管关键词。
- 检查/化验：项目名与异常标记（↑/↓/阳性/阴性），形成“异常概念”（如 `PCT↑`、`BNP↑`、`K+↓`）。

抽取时机：
- 在 `add_event/add_events` 阶段同步解析并写入患者级库的 `metadata`。搜索时直接使用，避免重复解析。

### 6. 临床规则模板（示例）
规则用于 `rule_boost`，命中即加分；可通过 YAML/JSON 配置，便于扩展与调参。

示例（伪 YAML）：
```yaml
rules:
  # 诊断/症状 → 对应医嘱
  - name: infection_to_antibiotics
    if_any_disease: ["感染", "脓毒症", "发热"]
    then_any_drug_class: ["抗生素"]
    window_hours: 48
    weight: 0.25

  - name: ugib_to_ppi
    if_any_disease: ["上消化道出血", "消化道出血", "应激性溃疡风险"]
    then_any_drug_class: ["PPI", "护胃"]
    window_hours: 24
    weight: 0.25

  # 异常化验 → 矫治/监测
  - name: hypokalemia_to_kcl
    if_any_abnormal: ["K+↓"]
    then_any_drug: ["氯化钾"]
    window_hours: 24
    weight: 0.3

  - name: high_bnp_to_diuretics
    if_any_abnormal: ["BNP↑"]
    then_any_drug_class: ["利尿剂"]
    window_hours: 24
    weight: 0.2

  # 手术/操作 → 围术期护理与医嘱
  - name: surgery_to_postop_orders
    if_event_type: ["surgery"]
    then_any_type: ["order", "exam", "nursing"]
    window_hours: 24
    weight: 0.2

type_prior:
  history:
    order: 0.6
    lab: 0.4
    exam: 0.4
  lab:
    order: 0.5
  surgery:
    order: 0.5
    exam: 0.5
  order:
    lab: 0.4
```

说明：
- 规则与先验矩阵仅作起始版本，需依据数据分布与医师反馈迭代。
- 规则评估时同时考虑方向与时间窗（近时序加权更高）。

### 7. 接口设计（草案）
- 事件相关检索：
  - `search_related_events(patient_id: str, event_id: str, top_k: int = 20, window_hours: int = 24, weights: Optional[dict] = None, tau_hours: int = 6) -> List[Dict]`
  - 返回字段（建议）：`[{"event_id", "score", "reasons": {"sim_vec", "time_score", "concept_overlap", "rule_hits": [...]}}]`
- 说明：
  - 召回合并去重后重排；`reasons` 用于可解释输出与线上调参。
  - 如 `event_id` 不存在，返回空列表或抛出异常由上层处理。

### 8. 性能与可用性
- 嵌入与缓存：
  - 首次写入时写全局缓存，后续读取向量零开销。
- 概念抽取：
  - 在写入阶段完成并存储到 `metadata`，检索阶段仅进行规则匹配与打分，低延迟。
- 可选索引：
  - 保持 `ScalableMemory` 的 ANN 索引；时间窗口遍历可按 `timestamp` 先读入内存并缓存排序索引以加速。

### 9. 评估建议
- 指标：
  - Evidence Recall@K（基于风险/诊疗代理输出中的证据事件）、TopK 点击一致性（与医生标注）
  - 解释质量（规则命中是否合理）、时间一致性（近窗优先）。
- 数据划分：
  - 按病人 ID 划分 dev/val/test，避免泄漏。

### 10. 未来扩展
- CBR（系统级记忆）：
  - 保存跨患者的“病例-决策-评估-医师点评”，形成可复用案例库。事件检索可作为 CBR 检索的子模块（先找到证据事件，再比对临床路径）。
- 自然语言检索（Planner + Tool）：
  - 当输入是自由文本问题时：先规划所需证据类型 → 调用事件检索/汇总器 → 结构化回答。
- 字典与规则的自动扩充：
  - 通过离线统计与少量 LLM 辅助，扩充同义词与药物/检查映射，逐步减少人工维护成本。

---

以上方案在真实 ICU 数据中具备较强的可解释性与实用性：向量语义覆盖“描述相似”关系，时间与规则覆盖“临床路径/因果/监测”关系；二者融合可兼顾召回与精度。下一步建议：
1) 整理一版轻量中文词典与规则模板；
2) 在 ICUMemoryAgent 中实现 `search_related_events`（两段式 + 解释输出）；
3) 在 2–3 名医师参与下快速迭代权重与规则，收敛初版参数。



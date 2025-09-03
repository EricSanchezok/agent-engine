## ICU 多智能体统一方案（Unified Multi-Agent System for ICU）

本文件汇总并收敛我们关于 ICU 风险预测与诊疗建议的讨论，形成一个不依赖外部重型基础设施、以 LLM+RAG 为核心的科研验证级统一方案。与 `ICU_BLUEPRINT.md` 协同：该文档偏“落地实现与接口细节”，蓝图偏“愿景与模块说明”。

### 1. 总体目标

- 实时/回放环境下：
  - 自动化风险预测：对关键 ICU 风险给出 p_1h/p_3h/p_6h 概率与趋势，维持稳定与可解释；
  - 交互式诊疗建议：对医生问题给出结构化、有证据引用、可追溯的建议。
- 科研友好：不训练专用模型；无需 Kafka/Postgres；可用本地数据完成评估与展示。

### 2. 关键组件与职责

- ICUDataIngestionAgent（数据接入/回放）
  - 从 `database/icu_patients/*.json` 加载 `sequence`；
  - 提供 `load/reset/update(n)` 等接口；
  - 输出 “最小封装事件（envelope）”：`patient_id/event_id/timestamp/event_type/sub_type/event_content/raw`。

- MemoryAgent（记忆）
  - 短期记忆：近 N 条或近 6h/24h 事件窗口（内存环形缓冲）。
  - 向量记忆：仅索引文本类片段（history/护理自由文本/exam 结论），用 `agent_engine.memory.Memory`（SQLite + sentence-transformers: all-MiniLM-L6-v2）。
  - 检索：6h 优先、24h 补充，TopK 固定、去重合并、确定性排序（先时间后相关度）。

- ICURiskPredictionAgent（风险预测，见 `docs/ICU_RISK_AGENT.md`）
  - “风险组”状态机：`monitoring/active/resolved/snoozed`；
  - LLM 两段式推断：证据表（A）→ 概率与趋势（B）；概率取锚点集合；
  - 稳定化：单调性修正 + EMA 平滑（α≈0.7）+ 单步限幅（≤0.15）+ 迟滞/冷却；
  - 不回喂历史概率给 LLM，仅在数值层保存与平滑；
  - 输出：含 `top_evidence_event_ids` 与英文 rationale 的结构化 JSON。

- ClinicalSuggestionAgent（诊疗建议，见 `docs/ICU_CLINICAL_SUGGESTION_AGENT.md`）
  - 多步 LLM：问题理解（Q）→ 证据表（A）→ 结构化建议（B）→ 自检（C）；
  - 依赖 MemoryAgent 检索与风险组 `meta`（仅名称/notes/top evidence ids）；
  - 输出：结构化 JSON（工作诊断/鉴别/管理计划/用药/监测/禁忌/随访/引用/限制）。

- SummarizationAgent（可选）
  - 周期/触发生成阶段摘要，压缩长程上下文；
  - 摘要进入向量记忆，RAG 时固定纳入。

- Evaluator（线下评估）
  - 弱金标准自动构造（量表阈值、lab 阈值/变化、干预时间锚、文本关键词）；
  - 风险：AUPRC/ROC、Brier、提前量；证据：Evidence Recall@K；
  - 问答：事实 EM/容差、证据召回、医师评分/一致性。

### 3. 数据与上下文

- 数据源：`database/icu_patients/*.json`；文件名为 `patient_id`；`sequence` 时间有序。
- 事件类型（常见）：`history`、`nursing`（含量表与护理记录）、`exam`、`lab`、`order`。
- 记忆策略：文本类入向量库；近窗事件进入短期窗口；检索 TopK 确定性排序，固定包含阶段摘要与关键 labs/干预小结（可选）。

### 4. 统一时序流程（每次 update）

1) Ingestion：读取下一事件，封装为标准 envelope。
2) Memory：写入短期窗口；文本类（及摘要）入向量库（如开启）。
3) Risk：
   - 从 Memory 检索近窗证据；
   - LLM A（证据表）→ LLM B（锚点概率与趋势、新候选）；
   - 数值后处理：单调性、EMA、限幅、迟滞/冷却、TTL、容量限制；
   - 产出/存储：风险组快照与 delta。
4) Clinical（按需/交互触发）：
   - 读取医生问题 + 风险组 meta + Memory 检索；
   - LLM Q/A/B/C；
   - 产出结构化建议 JSON（附 event_id/知识引用）。
5) 记录：英文日志与 JSON 结果；可选写入向量库（不污染文档检索）。

### 5. 概率与稳定（统一规范）

- 概率锚点：{0.01, 0.05, 0.15, 0.35, 0.60}（可在 dev 集微调）。
- 单调性：强制 p_1h ≤ p_3h ≤ p_6h。
- 多采样：N=3（可配），取中位数/众数为 p_raw。
- 平滑：EMA α≈0.7；单步限幅 ≤0.15；趋势基于 p_smooth 差分。
- 迟滞/冷却：θ_up=0.35 连续≥2 次触发；θ_down=0.15 连续≥3 次解除；冷却 30min。

### 6. 提示词与 RAG（统一约束）

- 证据先、结论后：先证据表，后结论，减少幻觉与抖动。
- 引用规范：所有硬结论附 `event_id`；知识条目标注 `guideline:slug`。
- 检索确定性：TopK 固定，去重合并，先时间后相关度排序；固定包含阶段摘要与关键 labs/干预小结。
- 不回喂历史概率与完整旧答案正文到 LLM；仅提供风险名称与简要 notes。

### 7. 接口与产物（统一示例）

- 风险输出（节选）
```json
{
  "patient_id": "...",
  "timestamp": "...",
  "risks": [
    {
      "name": "Sepsis",
      "state": "active",
      "p_raw": {"1h": 0.15, "3h": 0.35, "6h": 0.60},
      "p_smooth": {"1h": 0.14, "3h": 0.32, "6h": 0.55},
      "trend": "rising",
      "top_evidence_event_ids": ["e1", "e2", "e3"],
      "rationale": "..."
    }
  ],
  "delta": {
    "added": ["Sepsis"],
    "removed": [],
    "state_changed": [],
    "prob_changed": [{"name": "Sepsis", "from": {"1h": 0.05}, "to": {"1h": 0.15}}]
  }
}
```

- 诊疗建议输出（节选）
```json
{
  "query": "...",
  "working_diagnoses": [{"name": "Sepsis", "likelihood": "high", "evidence_event_ids": ["e1"]}],
  "management_plan": [
    {"priority": 1, "action": "Early broad-spectrum antibiotics", "evidence_event_ids": ["e2"], "references": ["guideline:surviving_sepsis_2021"]}
  ]
}
```

### 8. 评估与数据使用

- 数据划分：按病人 ID 分 dev/val/test（建议 60/20/20），避免泄漏。
- 弱金标准：量表（Braden/MEWS/Caprini）、lab 阈值与变化、干预时间锚、文本关键词合并判定。
- 指标：
  - 风险：AUPRC/ROC、Brier、提前量、Evidence Recall@K；
  - 问答：事实 EM/容差、证据召回、医师评分、重复性一致性。

### 9. 默认参数（统一起点）

- 检索：TopK=20；窗口 6h 优先、24h 补充。
- 多采样：temperature=0.2, top_p=0.9；预测 N=3；问答 N=1–3。
- 平滑：EMA α=0.7；单步限幅 0.15；迟滞 θ_up=0.35/θ_down=0.15；冷却 30min。
- 风险组容量：Top-K=8；新候选 M=3；ttl=3。

### 10. 最小演示与路线

- 最小演示：
  1) Ingestion 回放单病人；
  2) Risk 每次 update 产出概率与风险组；
  3) Clinical 交互问答给出建议；
  4) 英文日志 + JSON 存档用于离线评估。
- 路线：
  - Phase 0：无外部知识，仅内部 RAG；
  - Phase 1：加入阶段摘要与关键 labs/干预小结；
  - Phase 2：本地化指南片段与更多风险本体；
  - Phase 3：评估体系完善，形成可重复实验报告。



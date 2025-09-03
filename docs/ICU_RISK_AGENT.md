## ICU 风险预测 Agent（ICURiskPredictionAgent）设计文档

### 1. 目标与原则

- **核心目标**: 在不训练专用预测模型的前提下，基于 LLM 与 RAG 的证据推理，输出多时间视角（1h/3h/6h）的风险概率与趋势，并保持输出稳定、可解释、可复现。
- **关键能力**:
  - 在线：每次有新事件（event）到达时，动态更新“风险组”（Risk Group），对既有风险做概率更新、对新风险做发现与纳入、对过期风险做移除。
  - 概率：输出 p_1h、p_3h、p_6h，来自离散锚点集合，保证 p_1h ≤ p_3h ≤ p_6h。
  - 稳定：通过提示词约束、确定性检索、数值后处理（EMA/迟滞/限幅）降低抖动。
  - 可解释：每个风险绑定可引用的 event_id 证据与英文简短理由（rationale）。
- **不做的事**: 不训练传统或专用机器学习模型；不引入外部基础设施（Kafka/Postgres 等）。

### 2. 数据与上下文来源

- **事件数据**: 来自 `database/icu_patients/*.json`。每个文件名为 `patient_id`，内部 `sequence` 为时间有序事件。
- **典型字段**: `id`（事件UUID）、`timestamp`（ISO8601）、`event_type`、`sub_type`、`event_content`、`raw`（原始对象）。
- **事件范畴**（已抽样确认）：
  - 文本/半结构化：`history`（病程、首次病程录）、`nursing`（护理记录/量表）、`exam`（检查结论）。
  - 结构化倾向：`lab`（检验）、`order`（医嘱/用药）。
- **内存与检索**（MemoryAgent 提供）：
  - 短期记忆：近 N 条或近 6h/24h 的原始事件窗口（环形缓冲）。
  - 向量记忆：仅索引文本类片段（history/护理自由文本/exam 结论），使用 `agent_engine.memory.Memory`（SQLite + sentence-transformers: all-MiniLM-L6-v2）。
  - 检索策略：6h 优先、24h 补充，TopK 固定、去重合并、确定性排序（先时间后相关度）。

### 3. 风险组（Risk Group）

- **定义**: 针对每个病人，维护一个由若干“风险项”构成的集合。每个风险项携带状态、概率、趋势、证据与维护字段。
- **状态**：`monitoring | active | resolved | snoozed`。
- **概率**：
  - p_raw：当前时刻 LLM 直接输出的锚点概率（1h/3h/6h）。
  - p_smooth：对 p_raw 做数值稳定化后的概率，用于展示与告警。
- **趋势**：`rising | falling | flat`，基于 p_smooth 的短期差分判断。
- **证据**：与该风险关联的关键 `event_id` 列表，以及英文简短 `rationale`。

```json
{
  "patient_id": "1125203145",
  "updated_at": "2025-01-06T10:15:00",
  "risks": [
    {
      "name": "Sepsis",
      "state": "monitoring",
      "p_raw": {"1h": 0.15, "3h": 0.35, "6h": 0.60},
      "p_smooth": {"1h": 0.14, "3h": 0.32, "6h": 0.55},
      "trend": "rising",
      "top_evidence_event_ids": ["e1", "e2", "e3"],
      "rationale": "Concise English rationale citing key events.",
      "notes": "Last evidence: ...",
      "first_detected_ts": "2025-01-06T09:40:00",
      "last_update_ts": "2025-01-06T10:15:00",
      "up_count": 2,
      "down_count": 0,
      "ttl": 3
    }
  ]
}
```

### 4. 触发与总体流程（每次 update）

1) 事件触发：ICUDataIngestionAgent 推送一个新事件（或步进若干事件）。
2) 检索上下文：MemoryAgent 按窗口与 TopK 返回证据片段，附 `event_id`。
3) LLM 步骤A（更新已有风险）：
   - 输入：
     - 新证据片段（history/nursing/exam 为主，带 event_id）。
     - 风险组“元信息”：仅包含风险名称 `name` 与上一轮证据的 1-2 行摘要 `notes`（不包含历史概率数字）。
   - 输出：对每个既有风险生成“证据表（支持/反驳/不确定，含 event_id 与影响力）”。
4) LLM 步骤B（概率与新候选）：
   - 输入：步骤A 的证据表。
   - 输出：
     - 既有风险：选择锚点概率 p_raw（p_1h/p_3h/p_6h），并给出 `trend`（rising/falling/flat）、`top_evidence_event_ids`（≤3）与英文 `rationale`。
     - 新候选风险（≤M）：`name`、p_raw、证据 event_id 与理由，避免与既有重复。
5) 合并与稳定化（仅数值层，不回喂 LLM）：
   - 单调性修正：强制 p_1h ≤ p_3h ≤ p_6h。
   - EMA 平滑：p_smooth(t) = α·p_raw(t) + (1-α)·p_smooth(t-1)，每个提前量独立平滑。
   - 限幅：|p_smooth(t) - p_smooth(t-1)| ≤ Δ_max。
   - 迟滞/冷却：抬升阈值 θ_up 连续≥2 次才入组/升级；回落阈值 θ_down 连续≥3 次才降级/出组；告警冷却窗口内不重复告警。
   - TTL：若连续 T 次无证据，ttl 递减至 0→`resolved` 并从风险组移除。
   - 容量限制：仅保留 Top-K 高 p_smooth 的 `monitoring/active` 风险，控制上下文规模。
6) 产出：返回最新风险组快照与变更 delta（新增/删除/状态变更/概率变更），记录英文日志。

### 5. LLM 概率锚点与 p_raw 生成

- **概率锚点集合**（可调）：{0.01, 0.05, 0.15, 0.35, 0.60}。
- **单调约束**：强制 p_1h ≤ p_3h ≤ p_6h（违反则请求一次“自我修正”或在数值层纠正）。
- **两段式提示词**：
  - 步骤A：严格生成“证据表”（支持/反驳/不确定，带 `event_id` 与影响力）。不下结论、不给概率。
  - 步骤B：依据证据表，选择锚点概率，给出 `trend`、`top_evidence_event_ids` 与英文 `rationale`，并列出新候选风险（≤M）。
- **多采样集成**：对同一 `t` 生成 N 次（如 3 或 5），对每个提前量取“锚点中位数/众数”作为 p_raw，减少偶然抖动与随机性。
- **禁止自我强化**：不向 LLM 提供历史概率数字；仅提供风险名称与上一轮证据 1-2 行摘要（帮助“查什么”）。

### 6. p_smooth 稳定化（数值层）

- **EMA 平滑**（每个风险、每个提前量）：
  - 初始化：若无历史，p_smooth(t0) = p_raw(t0)。
  - 递推：p_smooth(t) = α·p_raw(t) + (1-α)·p_smooth(t-1)。
  - 建议：α = 0.7（可在 dev 集微调）。
- **单步限幅**：|Δp_smooth| ≤ Δ_max，建议 Δ_max=0.15。
- **单调修正**：若 p_smooth_1h > p_smooth_3h 或 p_smooth_3h > p_smooth_6h，则就地拉回为非降序。
- **迟滞与冷却**（驱动告警而非评估）：
  - 抬升阈值 θ_up=0.35 连续≥2 次才触发监控/升级；
  - 回落阈值 θ_down=0.15 连续≥3 次才解除/移除；
  - 冷却窗口 30 分钟：窗口内维持概率更新但不重复告警。
- **趋势计算**：基于 p_smooth 的短期差分（如过去 N 次均上升→rising）。
- **节流**：每 5 分钟或遇到“实质证据”才触发一次 LLM 推断，降低计算与抖动。

### 7. 提示词设计（摘要）

- **System（一次性）**：
  - 角色：ICU 风险预测专家；仅使用给定患者证据与指南摘要。
  - 约束：概率必须取锚点；p_1h ≤ p_3h ≤ p_6h；引用证据必须标注 `event_id`；输出严格 JSON；所有说明使用英文短句。
  - 校准守则：单一轻弱证据→下探一档；多强证据一致且持续→上探一档；证据冲突→下调一档；不得使用外部信息。
- **Step A（证据表）**：输入（风险名称+上一轮证据摘要），新证据片段；输出结构化证据表（支持/反驳/不确定，带 `event_id` 与 strength）。
- **Step B（概率与候选）**：输入证据表；输出既有风险的 p_raw/趋势/证据 id/英文理由，以及新候选风险（≤M）。
- **Step C（自检）**：若不单调或格式异常，一次自我修正。

### 8. 接口契约（建议）

- **输入**（每次 update 调用）
  - `patient_id`: string
  - `current_event`: {id, timestamp, event_type, sub_type, event_content, raw}
  - `evidence_snippets`: [{event_id, timestamp, event_type, sub_type, text}]
  - `risk_group_meta`: 上一轮风险组“元信息”（仅 name + 1-2 行 notes，不含概率）

- **输出**
```json
{
  "patient_id": "...",
  "timestamp": "...",
  "risk_group": { /* 见第3节 JSON 结构 */ },
  "delta": {
    "added": ["Sepsis"],
    "removed": ["VAP"],
    "state_changed": [{"name": "AKI", "from": "monitoring", "to": "active"}],
    "prob_changed": [{"name": "Sepsis", "from": {"1h": 0.05}, "to": {"1h": 0.15}}]
  }
}
```

### 9. 配置项（默认值）

- 概率锚点：{0.01, 0.05, 0.15, 0.35, 0.60}
- 多采样：N=3，temperature=0.2，top_p=0.9
- 检索：TopK=20；窗口 6h 优先、24h 补充；去重与确定性排序
- EMA：α=0.7；单步限幅 Δ_max=0.15
- 阈值与迟滞：θ_up=0.35 连续 2 次入组；θ_down=0.15 连续 3 次出组；冷却=30min
- 风险组容量：Top-K 风险=8；新候选上限 M=3；ttl=3（连续 3 次无证据自动移除）
- 更新节流：每 5 分钟或有“实质证据”才触发 LLM 推断

### 10. 评估与数据使用（科研模式）

- **数据划分**：按病人 ID 分为 dev/val/test（如 60/20/20），避免泄漏。
- **弱金标准构造**（自动化，无需人工）：
  - 量表：`nursing` 中 Braden/MEWS/Caprini 打阈标注风险时段。
  - 检验：`lab` 中乳酸/WBC/肌酐/CRP/PCT 等阈值与变化（如 AKI 近似规则）。
  - 干预：`order/nursing` 中升压药/广谱抗生素/CRRT 等作为强证据时间锚。
  - 文本：`history/exam` 中肺炎/渗出/ARDS/心衰等关键词；与上面交叉验证作“合并判定”。
- **指标**：
  - 判别：p@1h/3h/6h 的 ROC/AUPRC；
  - 校准：Brier score、Reliability 曲线；
  - 早预警：首次告警提前量分布；
  - 证据：Evidence Recall@K（引用 `event_id` 命中率）。
- **流程**：dev 调提示词/锚点/采样参数；val 选版本；test 冻结汇报。

### 11. 故障与回退

- LLM 超时/异常：沿用上次 p_smooth 并标记 degraded；下一次成功后恢复。
- 证据稀少：允许输出最低锚点并标注不确定；保留 `notes` 引导下次检索关注点。
- 不回喂概率：历史概率不写入向量库，仅在 Agent 内部状态保存以做平滑与迟滞。

### 12. 隐私与日志

- 日志输出使用英文（print、log）并避免输出患者敏感信息全文，只打印必要摘要与 `event_id`。
- 设计用于本地科研验证，不将数据外发；如需联网 LLM，请遵循合规策略与脱敏要求。

### 13. 路线图（可选）

- 增加阶段摘要（SummarizationAgent）以压缩长程上下文。
- 风险词表/本体精炼，映射到 `risks_table.json` 的分类树，便于拓展。
- 概率锚点与提示词的系统化校准（用 dev 的等分箱/可靠性曲线调参）。
- 增加“问答/诊疗建议 Agent”对接，直接消费风险组与证据引用。

### 14. 最小演示建议

- 使用 `ICUDataIngestionAgent` 回放单个 `patient_id` 的 `*.json`；每次 `update()` 触发本 Agent：
  - 从 `MemoryAgent` 检索近 6h/24h 片段；
  - 按第 4 节流程执行 LLM 推断与数值稳定化；
  - 打印最新风险组快照（英文日志），并保存到本地 JSON 便于离线评估。



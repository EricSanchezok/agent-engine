## ICU 风险预测 Agent（ICURiskPredictionAgent）设计文档

### 1. 目标与原则

- **核心目标**: 在不训练专用预测模型的前提下，基于 LLM 与 RAG 的证据推理，输出多时间视角（1h/3h/6h）的风险概率与趋势，并保持输出稳定、可解释、可复现。
- **关键能力**:
  - 在线：每次有新事件（event）到达时，动态更新“风险组”（Risk Group），对既有风险做概率更新、对新风险做发现与纳入、对过期风险做移除。
  - 概率：输出 p_1h、p_3h、p_6h，来自离散锚点集合，保证 p_1h ≤ p_3h ≤ p_6h。
  - 稳定：通过提示词约束、确定性检索、数值后处理（EMA/迟滞/限幅）降低抖动。
  - 可解释：每个风险绑定可引用的 event_id 证据与英文简短理由（rationale）。

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

0) 事件重要性 Gating（先决条件）
   - 输入：
     - `current_event` 的基础字段：`event_id/timestamp/event_type/sub_type/event_content`（必要时截断/摘要）。
     - 检索摘要：从 MemoryAgent 检索近 6h（必要时 24h 补充）TopK 片段，仅传递 `event_id+一句话摘要`，避免长上下文成本。
     - 风险组“元信息”：仅 `name` 与上一轮 1-2 行 `notes`（不含历史概率）。
   - 策略：
     - 规则硬触发（始终触发）：
       - `surgery`/插管/CRRT/透析/气管切开/转入ICU/转出ICU/心跳骤停相关；
       - `order` 中启动或升级：升压药/抗休克/CRRT/气道管理/镇静镇痛持续泵入/血制品；
       - `lab` 严重异常：乳酸↑↑、PCT/CRP 急升、K+ 严重异常、肌酐/尿量提示 AKI 分期、ABG 重度酸碱紊乱等；
       - `nursing` 量表阈值：MEWS ≥ θ_MEWS_critical，或 Braden 急降（≥Δ_Braden）。
     - 规则软触发（可与节流结合）：
       - 新发“感染/出血/休克/ARDS/急性心衰/AKI”相关诊断词出现在 `history/exam` 摘要；
       - 广谱抗生素起始/阶梯升级；电解质/利尿/胰岛素等矫治用药显著调整；
       - 中度实验室异常、生命体征边缘波动。
     - LLM 轻量判别（可选）：
       - 对上述输入执行 2/3 分类：`critical | signal | noise`，输出 `is_important` 与英文理由、引用 `event_id`；
       - 多采样 N=2-3 取众数，降低偶然误差；阈值以 Recall 优先（保证不漏重大事件）。
     - 小模型判别（可选，详见第16节）：
       - 使用 icu_raw 弱标注训练轻量分类器，输出 `p(important)`；校准后与规则/LLM 融合。
   - 决策融合：`hard_trigger → 必触发`；否则 `LLM/小模型 ≥ θ_importance` 或命中软触发累计器（节流桶）时触发；默认 5 分钟最小触发间隔。
   - 结果：若 `is_important=True`，进入步骤 1-6；否则跳过推断，仅做 `ttl/冷却/限速` 更新并记录英文日志。

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
6) 产出：返回最新风险组快照与变更 delta（新增/删除/状态变更/概率变更），记录英文日志；同步记录本次 `gating` 决策与理由。

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

- Gating：
  - `gating_mode`: `rule_only | llm | hybrid`（默认 `hybrid`）。
  - 规则硬触发：事件类型/子类型白名单 + 实验室阈值集（乳酸/肌酐/电解质/ABG/炎症标志物等）。
  - 规则软触发：诊断/关键词/药物类目匹配（广谱抗生素/升压/CRRT 等），配合节流桶（默认 2 tokens/10min）。
  - LLM 判别：N=2，多数投票；温度=0.2；最大输入 1.2k tokens；近 6h TopK=10 的一句话摘要。
  - 小模型阈值：θ_importance=0.35（针对 Recall 优先可下调至 0.25）。
  - 最小触发间隔：5min；批处理窗口：2–5min 可聚合多条 `order`/`nursing`。

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
- **Gating 指标**：
  - 重要事件检出：Precision/Recall/F1（以弱金标准 + 专家抽样复核）。
  - 触发成本：平均每 1000 事件触发次数；LLM Token 消耗/延迟分布。
  - 漏检分析：对“硬触发”回溯应为 0；“软触发/LLM/小模型”漏检病例复盘。
- **流程**：dev 调提示词/锚点/采样参数；val 选版本；test 冻结汇报。

### 11. 故障与回退

- LLM 超时/异常：沿用上次 p_smooth 并标记 degraded；下一次成功后恢复。
- 证据稀少：允许输出最低锚点并标注不确定；保留 `notes` 引导下次检索关注点。
- 不回喂概率：历史概率不写入向量库，仅在 Agent 内部状态保存以做平滑与迟滞。
- Gating 回退：LLM 失败或小模型不可用时，自动切换到 `rule_only`；若连续多次失败，提升仅“硬触发”模式并延长最小触发间隔。

### 12. 隐私与日志

- 日志输出使用英文（print、log）并避免输出患者敏感信息全文，只打印必要摘要与 `event_id`。
- 设计用于本地科研验证，不将数据外发；如需联网 LLM，请遵循合规策略与脱敏要求。
  - 记录 `gating` 决策（event_id、决策类别、硬/软触发命中、LLM/模型分数、触发原因摘要），避免暴露事件全文。

### 13. 路线图（可选）

- 增加阶段摘要（SummarizationAgent）以压缩长程上下文。
- 风险词表/本体精炼，映射到 `risks_table.json` 的分类树，便于拓展。
- 概率锚点与提示词的系统化校准（用 dev 的等分箱/可靠性曲线调参）。
- 增加“问答/诊疗建议 Agent”对接，直接消费风险组与证据引用。

### 14. 最小演示建议

- 使用 `ICUDataIngestionAgent` 回放单个 `patient_id` 的 `*.json`；每次 `update()` 触发本 Agent：
  - 先执行 `gating`：打印本次决策、理由与触发类型（硬/软/LLM/模型）。
  - 从 `MemoryAgent` 检索近 6h/24h 片段；
  - 按第 4 节流程执行 LLM 推断与数值稳定化；
  - 打印最新风险组快照（英文日志），并保存到本地 JSON 便于离线评估。

### 15. 事件重要性 Gating（详细设计）

- 目标：用最低成本捕捉“会改变风向”的关键事件，减少对所有事件的推断与抖动。
- 三层策略：
  1) 规则硬触发（高召回、高确定性，0 成本）：
     - 类型：`surgery/插管/CRRT/透析/转入ICU/心肺复苏相关`；
     - 用药：`升压药/去甲肾上腺素/肾上腺素/多巴胺` 开始或剂量显著增加；`血制品`/`CRRT`/`镇静持续泵入`；
     - 实验室：`乳酸≥2 mmol/L 且上升`、`PCT/CRP 急升`、`肌酐急升或尿量骤降（AKI 分期）`、`K+<3.0 或 >5.5`、`ABG 重酸/重碱`；
     - 量表：`MEWS≥4`（或院感/跌倒等量表达高危转变）。
  2) 规则软触发（中召回，中确定性）：
     - 新发/强化诊断词：感染/出血/休克/ARDS/心衰/AKI/急腹症等；
     - 广谱抗生素起始或从窄谱升级；
     - 电解质/利尿/胰岛素/抗凝等矫治用药显著调整；
     - 中度实验室异常或连续边缘异常（以滑动窗口检测）。
  3) 统计/模型触发：
     - LLM 轻量判别：输入为事件摘要 + 近 6h TopK 一句话摘要 + 风险组元信息；输出 `critical|signal|noise`、英文理由与 `is_important`；N=2-3 多数投票。
     - 小模型：见第16节（弱标注训练 + 概率阈值 + 校准）。

- 决策融合：
  - `hard_trigger=TRUE → 触发`；
  - 否则 `LLM_is_important=TRUE` 或 `p_model ≥ θ_importance` 或 `软触发累计器超阈` → 触发；
  - 加节流：同 5 分钟最小触发间隔；可对 `order/nursing` 聚合 2–5 分钟批处理。

### 16. 小模型训练（重要事件分类器）

- 数据来源：`database/icu_raw/*.json` 的 `sequence`。
- 弱标注构造：
  - 正类（重要）：命中“硬触发”规则的事件；`lab` 严重异常；`order` 启动升压药/CRRT/广谱抗生素；`surgery`；`nursing` MEWS≥阈；
  - 负类（非重要）：长期常规护理医嘱、重复量表无变化、稳定生命体征/正常化验；
  - 难例（可选标注/半监督）：软触发/边缘异常/新诊断但未见干预。
- 特征工程：
  - 类别特征：`event_type/sub_type`；
  - 词袋/短语特征：`event_content`（停用词、n-gram）+ 关键词字典（药物类目/异常符号↑↓/诊断词）；
  - 结构特征：是否含“↑/↓/阳性/阴性/剂量/频次/停用/启动”等；
  - 时序特征：距上次触发的时间、同类事件的计数/速率；
  - 预训练向量：MiniLM 等句向量（冻结）作为稠密特征。
- 模型建议：
  - 轻量基线：Logistic Regression / Linear SVM / XGBoost；
  - 稠密融合：将句向量拼接到稀疏特征；
  - 校准：Platt/Isotonic 校准输出 `p(important)`，以 Recall 优先设置阈值（如 0.25–0.35）。
- 评估：
  - Dev/Val/Test 按病人划分；
  - 指标：Recall@阈、PR曲线、触发数量/千事件；
  - 误差分析：漏检/误检的子类型、关键词。

### 17. CBR/Memory 增强（案例库驱动的早期预警）

- 动机：相似病例的“事件→干预→结局/风险演化”轨迹可作为强先验，帮助早识别“真正会改变风向”的事件。
- 案例定义：
  - 索引键：`proto_event`（触发事件的向量 + 结构特征）+ 近 6–24h 证据集合（TopK 摘要）、患者简要表型（年龄/主要诊断/器官支持）；
  - 结果：随后的风险组变化（p_raw/p_smooth 的抬升/下降）、触发的干预（抗生素/升压/CRRT/手术等）、时间差分；
  - 解释：Top 证据 `event_id` 与英文 rationale 摘要。
- 构建：
  - 从 icu_raw 回放并在规则/弱标注下提取“重要事件”与其后 6–24h 的干预与风险线索（可由 MemoryAgent + 规则抽取完成）；
  - 去重与聚类（句向量 + 类型/关键词约束）形成原型；
  - 保存为可检索的案例库（DuckDB/HNSW + 元数据）。
- 推断时使用：
  - Gating 阶段检索相似案例，若命中“高影响原型”，直接加权提高 `is_important`；
  - 风险预测阶段将案例的“概率锚点建议/趋势先验”作为软约束，仍由数值层（EMA/限幅/迟滞）稳定输出；
  - 输出解释中列出命中的案例 `case_id` 与对预测的影响（英文简短理由）。



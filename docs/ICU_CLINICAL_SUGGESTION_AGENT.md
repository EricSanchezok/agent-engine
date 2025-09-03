## ICU 诊疗建议 Agent（Clinical Suggestion Agent）设计文档

### 1. 目标与定位

- **目标**：面向 ICU 场景的问答与决策支持，在患者当前证据与内置知识的基础上，给出结构化、可解释、可追溯的诊疗建议。
- **定位**：纯 LLM + RAG，不训练模型；与 `ICURiskPredictionAgent`、`MemoryAgent` 无缝衔接；所有输出附证据 `event_id` 引用与知识引用。

### 2. 输入与依赖

- **输入**
  - `query`：医生自然语言问题（诊断/鉴别/治疗/用药/监测/围术期等）。
  - `patient_id`：用于检索患者上下文。
  - `risk_group_meta`：来自风险 Agent 的 Top-K 风险（仅名称、p_smooth、top evidence ids、notes）。
  - `evidence_snippets`：来自 MemoryAgent 的近窗文本片段（6h/24h，TopK，附 `event_id`）。
- **依赖**
  - `MemoryAgent`：向量检索（history/护理自由文本/exam 结论）+ 短期窗口。
  - 可选外部知识：本地化指南片段/静态知识库（无网条件下可缺省）。

### 3. 高层流水线

1) 问题理解（LLM-步骤Q）
   - 解析任务类型：诊断/鉴别/治疗/用药/监测/并发症/围手术等。
   - 抽取关键实体：疾病/症状/器官/药物/目标指标/时间窗与安全约束（禁忌、肾/肝功能）。
2) 上下文检索（RAG）
   - 从 `MemoryAgent` 获取近 6h/24h 片段 TopK，确定性排序（去重；先时间后相关度）。
   - 合并风险组摘要（不含历史概率数字），保留 `top evidence ids` 与关注点。
   - 可加入“关键 labs/干预”小结（若可从结构化/文本轻解析得到）。
3) 证据表（LLM-步骤A）
   - 输出“支持/反驳/不确定”三类证据表，每条证据包含 `event_id`、影响力、方向，不下最终结论。
4) 方案推理（LLM-步骤B）
   - 生成结构化建议：
     - 工作诊断与鉴别（带证据 event_id 引用）；
     - 管理方案（优先级化步骤）、用药（剂量/频次/途径/肾肝调整/相互作用）、监测项与复评节律；
     - 禁忌与风险提示；
     - 未确证项的下一步检查/试验治疗建议。
   - 所有硬结论均附 `event_id` 与知识引用（若有）。
5) 自检与输出（LLM-步骤C）
   - 一致性/安全自检：与证据冲突、禁忌/相互作用冲突则自修正；
   - 输出严格 JSON（见第 5 节），同时给英文简短叙述摘要。

### 4. 提示词分层（摘要）

- **System**
  - 角色：ICU 诊疗建议专家；仅依据提供的患者证据片段与知识条目。
  - 规则：关键结论必须引用 `event_id` 与知识条目；输出严格 JSON；所有说明使用英文短句。
  - 安全：禁忌/相互作用优先；不确定时提出“下一步检查/保守方案”。
- **Step Q（问题理解）**：识别任务类型与实体，产出结构化“问题解析对象”。
- **Step A（证据表）**：生成支持/反驳/不确定证据表（带 `event_id` 与影响力），不下结论。
- **Step B（建议）**：给出结构化建议（诊断/管理/监测/用药/禁忌/随访），逐项附引证 ID 与知识引用。
- **Step C（自检）**：一致性/安全校验与 JSON 验证，必要时自修正一次。

### 5. 输出 JSON（建议结构）

```json
{
  "query": "Doctor's question",
  "patient_snapshot": {
    "time_window": "last_6h/24h",
    "salient_findings_event_ids": ["e1", "e2"]
  },
  "working_diagnoses": [
    {"name": "Sepsis", "likelihood": "high", "evidence_event_ids": ["e1", "e5"]}
  ],
  "differentials": [
    {"name": "Cardiogenic shock", "likelihood": "medium", "rationale": "..."}
  ],
  "management_plan": [
    {"priority": 1, "action": "Early broad-spectrum antibiotics", "rationale": "...", "references": ["guideline:surviving_sepsis"], "evidence_event_ids": ["e2"]},
    {"priority": 2, "action": "Fluid resuscitation 30 ml/kg", "monitoring": ["MAP", "Lactate"], "caveats": ["CKD stage 4"], "references": ["guideline:..."]}
  ],
  "medications": [
    {"drug": "Norepinephrine", "dose": "0.05–0.3 µg/kg/min", "route": "IV", "adjustment": "Titrate to MAP ≥65", "interactions": [], "contraindications": []}
  ],
  "monitoring": [
    {"item": "Lactate", "frequency": "q4h", "target": "<2 mmol/L"}
  ],
  "contraindications": ["Recent GI bleed (if considering anticoagulation)"],
  "followups": ["Blood cultures before antibiotics if not delayed"],
  "citations": [
    {"type": "event", "id": "e1"},
    {"type": "guideline", "slug": "surviving_sepsis_2021"}
  ],
  "limitations": ["No recent creatinine available; dosing conservative"]
}
```

### 6. 稳定性与可控性

- 检索确定性：TopK 固定、去重、先时间后相关度排序；固定包含“阶段摘要+关键 labs/干预小结”。
- 多步推理：Q → A → B → C，避免一步到位的幻觉与遗漏。
- 多采样（可选）：N=2–3 合并要点；建议强度锚点化（low/medium/high）。
- 不污染记忆：不把上次答案正文写入向量库，仅存 Q&A 在单独通道并标注类型。

### 7. 工程接口（与现有组件对接）

- **输入**：`query`、`patient_id`、`risk_group_meta`、`evidence_snippets`（MemoryAgent 检索的片段）。
- **输出**：第 5 节 JSON + 英文摘要；缺证/冲突时返回“所需信息清单”。
- **LLM 参数**：temperature=0.2–0.4，top_p=0.9；max_tokens 与 TopK 可配置；复用 `agent_engine.llm_client.AzureClient`。

### 8. 评估（线下）

- 证据召回：引用 `event_id` 的 Recall@K。
- 事实准确：实验室数值/时间点的 EM/容差匹配。
- 医师评分：相关性/可操作性/安全性；冲突检测率（禁忌/相互作用）。
- 一致性：同语义问题在相近窗口的一致性。

### 9. 默认参数（起点）

- 检索窗口：6h 优先、24h 补充；TopK=20。
- 多步：Q → A → B → C。
- 采样：temperature=0.2, top_p=0.9；N=1（可升至 2–3）。
- 输出：严格 JSON + 英文摘要；所有 prints/logs 用英文。

### 10. 路线图（可选）

- 本地知识库扩展：常见 ICU 指南/药典片段（可离线），统一 `guideline:slug` 标识。
- 结构化小结增强：为 labs/干预生成稳定的“近窗小结”片段供 RAG 使用。
- 与风险 Agent 协同：根据风险组高关注项，动态调整检索查询与建议优先级。



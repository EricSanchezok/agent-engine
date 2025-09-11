# ICUAgent Memory 检索与工具方案 v0.2

> 目标：为 ICU 单体代理提供“可被大模型稳健调用”的记忆检索与证据构建能力，支持治疗决策类问答（抗生素、液体与血流动力学、CRRT 抗凝/禁忌、药物安全等）。

---

## 1. 设计原则

- 语义 + 结构双引擎：语义相似度用于召回“相关”，结构化过滤（类型/标签/时间窗/风险）保证“精准”。
- 时间感知优先：必须支持锚点时间窗、最近值、趋势、阈值突破与事件前后链。
- 可解释与可追溯：所有聚合/计算工具返回原始 `event_id` 溯源与计算细节。
- 统一返回 Schema：固定字段，便于 LLM 进行二次推理与引用。
- 可扩展：在工具层新增专题能力不影响底层存储与基础检索接口。

> 存储与索引：向量检索 + 结构化过滤 + 时间衰减打分。遵循 ScalableMemory 目录约定：`persist_dir` 视为最终存储目录本身（不要再基于 name 创建子目录）。日志使用 `agent_engine.agent_logger`。

---

## 1.5 数据画像与字段映射（基于 1125112810.json 抽样）

- 事件类型与分布：存在 `history`、`order`、`nursing`、`lab`、`exam`、`surgery` 等；其中大量关键信息分散在 `history` 叙述文本中。
- 感染学证据：
  - “培养/药敏”多以检验“医嘱”（`order`）形式下达，但“结果”常回写到 `history` 文本（如“血培养…找到革兰阴性杆菌”“胸水培养：鲍曼不动杆菌”）。
  - 炎症指标（CRP、PCT、IL-6、WBC/NEUT% 等）主要在 `lab` 聚合文本中。
- 呼吸与通气：
  - 呼吸机参数（模式 SIMV、VT、f、FiO2）既见于围术期 `nursing`，也见于后续 `history` 文本。
  - ABG（血气）结果常出现在 `history` 文本段，未必独立成 `blood_gas` 子类型。
- 液体与出入量：
  - “昨日入量/出量/尿量/胸引”在 `history` 文本中以中文自然语言出现（例：“入量3440ml…出量2800ml…”、“左侧胸引1700ml”）。
- 影像与结构化检查：
  - `exam` 事件 `event_content` 为结构化对象（如 `exams_name`/`findings`/`result`）。
- 医嘱时序：
  - `order` 事件含 `metadata.end_timestamp` 可作为长/短期医嘱的时段边界。

数据画像结论：需强化“叙述文本结构化抽取”与“跨类型证据聚合”。专题工具要能从 `history` 中抽取 ABG、出入量、培养结果与呼吸机参数；感染工具需同时利用 `order`（取样/送检）与 `history`（结果/药敏）。

---

## 2. 工具分层

### 2.1 基础检索（通用入口）
- search_events_semantic(query, top_k, time_range, event_types, sub_types, risk_tags, must_include, must_exclude)
- search_events_structured(event_types, sub_types, risk_tags, metadata_filters, time_range, limit)
- get_timeline(start_time, end_time, group_by=["event_type","sub_type"], include_content)
- get_anchor_window(anchor_event_id|anchor_time, before, after, filters)
- get_latest_by_type(event_type, sub_type, n, with_same_type_window_hours)
- regex_keyword_search(patterns, time_range, event_types)
- expand_related_events(seed_event_id, relation=["same_order","same_session","same_day","same_tag"], depth)

### 2.2 解析与抽取（面向中文叙述文本与归一化）
- extract_structured_from_notes(patterns, time_range, scopes=["history","nursing"])：从叙述文本中抽取 ABG（pH/pCO2/pO2/HCO3-/BE/sO2/Lac 等）、呼吸机参数（模式/VT/f/FiO2/PEEP）、出入量（入量/出量/尿量/胸引）。
- normalize_labs(lab_events, unit_map, alias_map)：将 `lab` 聚合文本解析为结构化键值，归一单位与别名（如 CRP/PCT/IL-6 等）。
- map_drug_aliases(text, dictionaries)：药品通用名/商品名/类别统一到标准药理类别，便于抗菌药史与剂量调整。

### 2.3 临床专题检索/计算（面向决策）
- 感染与抗感染：
  - get_infection_bundle(time_range, latest_only)
  - get_antibiotic_history(drug_names_or_classes, time_range, include_dose)
  - get_source_control_events(time_range)
  - get_culture_and_susceptibility(sites=["blood","sputum","urine","csf","catheter","pleural"], time_range)
- 肾功能/CRRT：
  - get_renal_status(time_range)
  - get_rrt_sessions(modality=["CRRT","HD","SLED"], time_range, anticoag=["heparin","citrate","none"])
  - get_renal_dose_adjustment_candidates(time_range)
- 出血/凝血与输血：
  - get_bleeding_and_coag_profile(time_range)
  - find_anticoagulation_contraindications(time_range)
- 血流动力学与液体管理：
  - compute_fluid_balance(window_hours, include_inputs, include_outputs)
  - get_hemodynamic_support(time_range)
- 呼吸通气：
  - get_respiratory_status(time_range)
- 肝功能/代谢/其他：
  - get_hepatic_status(time_range)
  - get_endocrine_glycemic_status(time_range)
- 药物安全：
  - find_drug_allergies_and_adrs(time_range)
  - get_high_risk_medications(time_range, classes)
- 关键趋势/阈值：
  - get_trend(metric, window_hours, summarize)
  - get_threshold_crossings(metric, threshold, direction, window_hours)
- 矛盾/冲突扫描：
  - find_order_contraindication_conflicts(topic, time_range)

### 2.4 证据合成与查询理解
- summarize_events_for_question(question, events, style)
- build_treatment_context(topic, time_range)
- parse_question_to_filters(question)

### 2.5 数据可用性与质量
- list_available_channels()
- check_data_completeness(required, time_range)
- deduplicate_and_rank(events, by, top_k)
 - build_synonyms_catalog(categories=["labs","drugs","vent_modes","fluid_terms"])：基于样本数据构建中文同义词/别名词典（如“入量/摄入量/输入量”，“出量/输出量/出液/引流量”，“FiO2/吸入氧浓度”）。

---

## 3. 统一返回 Schema（建议）

```json
{
  "events": [
    {
      "id": "string",
      "timestamp": "ISO-8601",
      "event_type": "string",
      "sub_type": "string",
      "content": "string",
      "attributes": { "risks": [], "metadata": {} },
      "score": 0.0
    }
  ],
  "metrics": { "net_fluid_24h": -1200, "lactate_trend": "down" },
  "extractions": {
    "abg": { "pH": 7.47, "pCO2": 44, "pO2": 124, "Lac": 0.9 },
    "vent": { "mode": "SIMV", "VT_ml": 500, "RR": 15, "FiO2": 0.4 },
    "fluid": { "in_ml_24h": 3440, "out_ml_24h": 2800, "urine_ml_24h": 2800, "pleural_drain_ml_24h": 1700 }
  },
  "units": { "FiO2": "fraction", "VT_ml": "mL", "urine_ml_24h": "mL" },
  "time_window": { "start": "...", "end": "..." },
  "summary": "string",
  "trace": ["event_id_1","event_id_2"]
}
```

说明：
- events：返回原子事件列表，必须可追溯到存储中的 `event_id`。
- metrics：聚合或计算指标（如净入出量、趋势方向等）。
- time_window：统一标注本次检索/计算的时间窗。
- summary：可选的简要结论，便于 LLM 直接引用。
- trace：计算或筛选使用过的关键事件引用。

---

## 4. 排序与打分建议

- 基础分：语义相似度 \(s_{sem}\)
- 加权项：
  - 时间衰减：\( w_t \cdot \exp(-\Delta t / \tau) \)
  - 严重度/风险标签：\( w_r \)
  - 事件类型优先级（如 labs > orders > notes）：\( w_c \)
- 总分：\( score = s_{sem} + w_t + w_r + w_c \)

---

## 5. 存储/索引与实现要点

- 向量化：对 `Event.event_content` 进行嵌入，当前实现维度 3072；常用字段入 `attributes`（如 event_type、sub_type、drug_name/class、lab_name/value/unit、risks、session_id 等）。
- 结构化过滤优先，其次再做向量重排；大查询可先语义召回，再分桶聚合。
- 目录与日志：
  - `persist_dir` 直接作为最终存储目录，不再基于 `name` 生成子目录。
  - 日志统一使用 `agent_engine.agent_logger`，记录入参、命中计数、时间窗与用时（英文日志）。
- 缓存：
  - 可维护事件到向量的只读缓存（避免重复嵌入），最终写入主 Memory 存储以参与检索。

### 5.1 文本抽取策略（针对中文叙述）
- 规则优先 + 轻量词典：正则抽取 ABG、出入量、呼吸机参数、培养结果等，结合同义词词典做归一。
- 去重与容错：同一事件内字段重复记录需去重；对单位/小数/中英文混写做鲁棒解析。
- 位置与溯源：在 `extractions` 中标注来源事件 `id`，必要时记录文本切片位置，便于审计。

---

## 6. MVP 工具清单（优先实现的 10 个）

1. search_events_semantic
2. search_events_structured
3. get_timeline
4. get_anchor_window
5. get_latest_by_type
6. compute_fluid_balance（支持从 history 抽取入出量/胸引）
7. get_infection_bundle（整合 lab + order + history 培养/药敏）
8. get_antibiotic_history（含药名→药类映射与剂量解析）
9. get_bleeding_and_coag_profile
10. find_anticoagulation_contraindications

附加推荐（如需扩展为 12 项）
11. get_culture_and_susceptibility
12. get_abg_results（从 history/nursing 抽取）

---

## 7. 典型问答到工具序列映射

- 下一步抗生素怎么用？
  1) parse_question_to_filters → 感染主题
  2) get_infection_bundle + get_antibiotic_history
  3) get_renal_status → 判断是否需肾功剂量调整
  4) summarize_events_for_question("antibiotic escalation", …)

- CRRT 的病人是否要抗凝？
  1) get_rrt_sessions + get_bleeding_and_coag_profile
  2) find_anticoagulation_contraindications
  3) get_anchor_window(首个 CRRT 会话, ±24h)
  4) summarize_events_for_question("CRRT anticoagulation decision", …)

---

## 8. 示例接口（签名示例）

```python
from typing import List, Optional, Dict, Any
from agent_engine.agent_logger import AgentLogger

logger = AgentLogger(__name__)

def search_events_semantic(
    query: str,
    top_k: int = 50,
    time_range: Optional[Dict[str, str]] = None,  # {"start": "...", "end": "..."}
    event_types: Optional[List[str]] = None,
    sub_types: Optional[List[str]] = None,
    risk_tags: Optional[List[str]] = None,
    must_include: Optional[List[str]] = None,
    must_exclude: Optional[List[str]] = None,
) -> Dict[str, Any]:
    logger.info("search_events_semantic called")
    # 1) embed query → 2) vector search → 3) apply filters → 4) recency re-rank → 5) return schema
    ...
```

---

## 9. 待办/下一步

- 按 MVP 清单落地工具骨架与统一返回结构、日志与参数校验。
- 基于真实 `icu_patients` 数据字段，细化关键词/同义词与 `parse_question_to_filters`，并产出 `build_synonyms_catalog` 初版。
- 在 `patient_memory` 中梳理缓存与主存的写入策略，确保可检索的一致性。
- 血流动力学与呼吸通气的专题工具补充高级指标（如 ScvO2、静脉血气/肺泡-动脉氧梯度等）。

---

## 10. 变更历史

- v0.2：新增“数据画像与字段映射”，补充“解析与抽取”分层、文化/药敏与 ABG 工具，扩展返回 Schema（extractions/units），细化 MVP 对应真实数据来源。
- v0.1：首次方案整理，定义分层、统一返回 Schema、MVP 工具与映射示例。



# ICUAgent Memory 检索与工具方案 v0.3

> 目标：为 ICU 单体代理提供"可被大模型稳健调用"的记忆检索与证据构建能力，支持治疗决策类问答（抗生素、液体与血流动力学、CRRT 抗凝/禁忌、药物安全等）。

---

## 1. 设计原则

- 语义 + 结构双引擎：语义相似度用于召回"相关"，结构化过滤（类型/标签/时间窗/风险）保证"精准"。
- 时间感知优先：必须支持锚点时间窗、最近值、趋势、阈值突破与事件前后链。
- 可解释与可追溯：所有聚合/计算工具返回原始 `event_id` 溯源与计算细节。
- 统一返回 Schema：固定字段，便于 LLM 进行二次推理与引用。
- 可扩展：在工具层新增专题能力不影响底层存储与基础检索接口。
- **工具收敛与高效调用**：减少功能重叠，将高频串联操作合并为复合工具，降低上下文消耗。
- **命名空间隔离**：使用 `icu_mem.*` 前缀避免与其他服务工具混淆。

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

## 2. 优化后的工具分层（v0.3）

### 2.1 基础原语（3个核心工具）
- **icu_mem.search_events**：统一语义+结构化检索
  - 参数：`query?`, `filters?`, `relations?`, `ranking?`, `response_format`, `fields?`, `pagination?`
  - 功能：合并原 `search_events_semantic` + `search_events_structured` + `regex_keyword_search`
- **icu_mem.time_context**：统一时间相关操作
  - 参数：`mode ∈ {timeline, anchor_window, latest}`, `time_range`, `anchor?`, `before?`, `after?`, `group_by?`, `filters?`
  - 功能：合并原 `get_timeline` + `get_anchor_window` + `get_latest_by_type`
- **icu_mem.expand_related**：关系扩展
  - 参数：`seed_event_id`, `relation_types`, `depth`, `filters?`
  - 功能：原 `expand_related_events`，支持 `same_order/same_session/same_day/same_tag`

### 2.2 解析与抽取（2个工具）
- **icu_mem.extract_from_notes**：中文叙述文本结构化抽取
  - 参数：`patterns`, `time_range`, `scopes=["history","nursing"]`, `response_format`
  - 功能：抽取 ABG、呼吸机参数、出入量、培养结果等，统一返回结构与溯源
- **icu_mem.normalize_labs**：实验室数据归一化
  - 参数：`lab_events`, `unit_map`, `alias_map`, `response_format`
  - 功能：将 `lab` 聚合文本解析为结构化键值，归一单位与别名

### 2.3 复合专题（3个高价值工具）
- **icu_mem.topic_context**：一站式决策上下文构建
  - 参数：`topic ∈ {infection, renal, hemodynamic, respiratory, hepatic, endocrine, drug_safety}`, `time_range`, `filters?`, `response_format`
  - 功能：内部组合调用基础工具，输出统一 Schema，替代以下分散工具：
    - `get_infection_bundle` + `get_culture_and_susceptibility` + `get_source_control_events`
    - `get_renal_status` + `get_rrt_sessions` + `get_renal_dose_adjustment_candidates`
    - `get_bleeding_and_coag_profile` + `find_anticoagulation_contraindications`
    - `get_hemodynamic_support` + `get_respiratory_status` + `get_hepatic_status` + `get_endocrine_glycemic_status`
    - `find_drug_allergies_and_adrs` + `get_high_risk_medications`
    - `get_trend` + `get_threshold_crossings` + `find_order_contraindication_conflicts`
- **icu_mem.antibiotic_history**：抗菌药物史（高频独立调用）
  - 参数：`drug_names_or_classes`, `time_range`, `include_dose`, `response_format`
  - 功能：药名→药类映射与剂量解析，含肾功剂量调整建议
- **icu_mem.fluid_balance**：液体平衡计算（高频独立调用）
  - 参数：`window_hours`, `include_inputs`, `include_outputs`, `response_format`
  - 功能：净入出量计算，含从 notes 抽取胸引/尿量等

### 2.4 实用工具（1个）
- **icu_mem.synonyms_catalog**：同义词与别名词典
  - 参数：`categories=["labs","drugs","vent_modes","fluid_terms"]`, `version?`
  - 功能：提供只读词典视图，支持中文同义词映射（如"入量/摄入量/输入量"）

### 2.5 兼容性策略
- **废弃工具映射**：保留旧名为薄封装，参数转发到新工具，记录 deprecation 日志
- **迁移时间线**：v0.3 并行支持，v0.4 标记废弃，v0.5 完全移除

---

## 3. 统一输入/输出协议（v0.3）

### 3.1 通用输入参数（所有工具共享）
```python
# 通用参数
time_range: Optional[Dict[str, str]] = None  # {"start": "...", "end": "..."}
filters: Optional[Dict[str, Any]] = None     # {event_types, sub_types, risk_tags, metadata, must_include, must_exclude}
relations: Optional[Dict[str, Any]] = None  # {types: ["same_order","same_session"], depth: int}
ranking: Optional[Dict[str, Any]] = None     # {time_decay_tau: float, type_priorities: dict, risk_weights: dict}

# 响应控制
response_format: str = "simple"              # "simple" | "detailed" | "trace"
fields: Optional[List[str]] = None           # ["id","timestamp","event_type","summary","score","attributes"]
pagination: Optional[Dict[str, Any]] = None  # {page_size: int=25, page_token: str}
result_limit: Optional[int] = None           # 最大返回事件数
char_limit_per_event: Optional[int] = None   # 单事件内容字符限制（默认280）
```

### 3.2 统一输出 Schema（增强版）
```json
{
  "events": [
    {
      "id": "string",
      "timestamp": "ISO-8601",
      "event_type": "string",
      "sub_type": "string",
      "summary": "string",                    // 高信号摘要，替代大段原文
      "content_preview": "string",             // 仅在 detailed 模式提供
      "attributes": { "risks": [], "metadata": {} },
      "score": 0.0,
      "score_components": {                    // 可解释性打分
        "semantic": 0.8,
        "time_decay": 0.2,
        "risk": 0.1,
        "type_priority": 0.05
      }
    }
  ],
  "metrics": { 
    "net_fluid_24h": -1200, 
    "lactate_trend": "down",
    "infection_score": 0.7,
    "renal_function": "moderate_impairment"
  },
  "extractions": {
    "abg": { "pH": 7.47, "pCO2": 44, "pO2": 124, "Lac": 0.9, "source_event_id": "evt_123" },
    "vent": { "mode": "SIMV", "VT_ml": 500, "RR": 15, "FiO2": 0.4, "source_event_id": "evt_124" },
    "fluid": { "in_ml_24h": 3440, "out_ml_24h": 2800, "urine_ml_24h": 2800, "pleural_drain_ml_24h": 1700 }
  },
  "units": { "FiO2": "fraction", "VT_ml": "mL", "urine_ml_24h": "mL" },
  "time_window": { "start": "...", "end": "..." },
  "summary": "string",                         // 自然语言结论，便于 LLM 直接引用
  "trace": ["event_id_1","event_id_2"],       // 计算或筛选使用过的关键事件引用
  "next_page_token": "string",                // 分页游标
  "ranking_params": {                         // 本次排序参数，便于回溯
    "time_decay_tau": 24.0,
    "type_priorities": {"lab": 1.0, "order": 0.8},
    "risk_weights": {"high": 1.5, "medium": 1.0}
  }
}
```

### 3.3 响应格式说明
- **simple**（默认）：仅返回高信号字段，`summary` 替代 `content`，无 `content_preview`
- **detailed**：附加 `content_preview`、完整 `score_components`、`ranking_params`
- **trace**：专用于调试，包含完整计算链路与中间结果

### 3.4 Token 效率策略
- 默认 `response_format="simple"`，仅返回高信号字段
- 强制分页 + `result_limit` 上限（建议 50-100）
- 提供 `fields` 选择器，支持按需返回字段
- 大文本仅返回 `content_preview`（默认 280 字符），防止上下文溢出

---

## 4. 排序与打分建议（增强版）

### 4.1 打分公式
- 基础分：语义相似度 \(s_{sem}\)
- 加权项：
  - 时间衰减：\( w_t \cdot \exp(-\Delta t / \tau) \)，默认 \(\tau = 24\) 小时
  - 严重度/风险标签：\( w_r \)
  - 事件类型优先级（如 labs > orders > notes）：\( w_c \)
- 总分：\( score = s_{sem} + w_t + w_r + w_c \)

### 4.2 可解释性增强
- 输出 `score_components` 拆解各项贡献
- 记录 `ranking_params` 到日志与响应，便于回溯
- 支持 `time_decay_tau` 作为入参调整（ICU 场景建议 24-72h）

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

### 5.2 错误处理与稳定性
- 输入校验失败时，返回"可行动"的错误消息（告诉调用方如何缩小范围/调整参数）
- 避免返回栈追踪或技术细节，优先返回自然语言指导
- 支持 `response_format="trace"` 用于调试场景

---

## 6. MVP 工具清单（v0.3 优化版）

### 6.1 核心工具（9个，按优先级排序）
1. **icu_mem.search_events**：统一检索入口（最高优先级）
2. **icu_mem.topic_context**：复合专题工具，支持 `infection` 主题
3. **icu_mem.antibiotic_history**：抗菌药物史（高频独立调用）
4. **icu_mem.fluid_balance**：液体平衡计算
5. **icu_mem.extract_from_notes**：中文叙述文本抽取
6. **icu_mem.time_context**：时间相关操作
7. **icu_mem.normalize_labs**：实验室数据归一化
8. **icu_mem.expand_related**：关系扩展
9. **icu_mem.synonyms_catalog**：同义词词典

### 6.2 实现阶段建议
- **Phase 1**：`search_events` + `topic_context(infection)` + `antibiotic_history`
- **Phase 2**：`fluid_balance` + `extract_from_notes` + `time_context`
- **Phase 3**：`normalize_labs` + `expand_related` + `synonyms_catalog`

### 6.3 兼容性支持
- 保留旧工具名为薄封装，参数转发到新工具
- 记录 deprecation 日志，引导迁移到新工具

---

## 7. 典型问答到工具序列映射（v0.3 优化版）

### 7.1 抗生素决策（优化前：4次调用 → 优化后：2次调用）
- **优化前**：
  1) `parse_question_to_filters` → 感染主题
  2) `get_infection_bundle` + `get_antibiotic_history`
  3) `get_renal_status` → 判断是否需肾功剂量调整
  4) `summarize_events_for_question("antibiotic escalation", …)`

- **优化后**：
  1) `icu_mem.topic_context(topic="infection")` → 一站式感染证据聚合
  2) `icu_mem.antibiotic_history` → 抗菌药物史与剂量建议

### 7.2 CRRT 抗凝评估（优化前：4次调用 → 优化后：1次调用）
- **优化前**：
  1) `get_rrt_sessions` + `get_bleeding_and_coag_profile`
  2) `find_anticoagulation_contraindications`
  3) `get_anchor_window(首个 CRRT 会话, ±24h)`
  4) `summarize_events_for_question("CRRT anticoagulation decision", …)`

- **优化后**：
  1) `icu_mem.topic_context(topic="renal")` → 内部合并 RRT 会话、凝血/出血风险、禁忌扫描与时间窗

### 7.3 液体管理（优化前：3次调用 → 优化后：1次调用）
- **优化前**：
  1) `compute_fluid_balance` + `extract_structured_from_notes`（出入量）
  2) `get_hemodynamic_support`
  3) `summarize_events_for_question("fluid management", …)`

- **优化后**：
  1) `icu_mem.fluid_balance` → 内部完成入出量计算、从 notes 抽取胸引/尿量等

---

## 8. 示例接口（v0.3 优化版）

### 8.1 核心工具接口
```python
from typing import List, Optional, Dict, Any
from agent_engine.agent_logger import AgentLogger

logger = AgentLogger(__name__)

def icu_mem_search_events(
    query: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    relations: Optional[Dict[str, Any]] = None,
    ranking: Optional[Dict[str, Any]] = None,
    response_format: str = "simple",
    fields: Optional[List[str]] = None,
    pagination: Optional[Dict[str, Any]] = None,
    result_limit: Optional[int] = None,
    char_limit_per_event: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Unified semantic + structured search with relation expansion.
    Combines: search_events_semantic + search_events_structured + regex_keyword_search
    """
    logger.info("icu_mem_search_events called", extra={
        "query": query, "filters": filters, "response_format": response_format
    })
    # 1) embed query → 2) vector search → 3) apply filters → 4) relation expansion → 5) ranking → 6) return schema
    ...

def icu_mem_topic_context(
    topic: str,  # "infection", "renal", "hemodynamic", "respiratory", "hepatic", "endocrine", "drug_safety"
    time_range: Optional[Dict[str, str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    response_format: str = "simple",
    fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    One-stop decision context builder for clinical topics.
    Internally combines multiple specialized tools based on topic.
    """
    logger.info("icu_mem_topic_context called", extra={
        "topic": topic, "time_range": time_range, "response_format": response_format
    })
    # Topic-specific evidence aggregation with unified schema output
    ...

def icu_mem_antibiotic_history(
    drug_names_or_classes: List[str],
    time_range: Optional[Dict[str, str]] = None,
    include_dose: bool = True,
    response_format: str = "simple",
) -> Dict[str, Any]:
    """
    Antibiotic history with drug name→class mapping and dose parsing.
    Includes renal dose adjustment recommendations.
    """
    logger.info("icu_mem_antibiotic_history called", extra={
        "drug_names_or_classes": drug_names_or_classes, "include_dose": include_dose
    })
    # Drug mapping + dose parsing + renal adjustment suggestions
    ...
```

### 8.2 兼容性封装示例
```python
# Deprecated wrapper (v0.3 compatibility)
def search_events_semantic(
    query: str,
    top_k: int = 50,
    time_range: Optional[Dict[str, str]] = None,
    event_types: Optional[List[str]] = None,
    sub_types: Optional[List[str]] = None,
    risk_tags: Optional[List[str]] = None,
    must_include: Optional[List[str]] = None,
    must_exclude: Optional[List[str]] = None,
) -> Dict[str, Any]:
    logger.warning("search_events_semantic is deprecated, use icu_mem_search_events instead")
    
    # Parameter mapping
    filters = {
        "event_types": event_types,
        "sub_types": sub_types,
        "risk_tags": risk_tags,
        "must_include": must_include,
        "must_exclude": must_exclude,
    }
    pagination = {"page_size": top_k}
    
    return icu_mem_search_events(
        query=query,
        filters=filters,
        pagination=pagination,
        response_format="simple"
    )
```

---

## 9. 待办/下一步（v0.3）

### 9.1 实现优先级
- **Phase 1**：实现 `icu_mem_search_events` + `icu_mem_topic_context(infection)` + `icu_mem_antibiotic_history`
- **Phase 2**：实现 `icu_mem_fluid_balance` + `icu_mem_extract_from_notes` + `icu_mem_time_context`
- **Phase 3**：实现 `icu_mem_normalize_labs` + `icu_mem_expand_related` + `icu_mem_synonyms_catalog`

### 9.2 技术债务
- 基于真实 `icu_patients` 数据字段，细化关键词/同义词与 `parse_question_to_filters`
- 在 `patient_memory` 中梳理缓存与主存的写入策略，确保可检索的一致性
- 血流动力学与呼吸通气的专题工具补充高级指标（如 ScvO2、静脉血气/肺泡-动脉氧梯度等）

### 9.3 兼容性迁移
- 实现旧工具名的薄封装，记录 deprecation 日志
- 制定 v0.4 废弃时间线与 v0.5 完全移除计划

---

## 10. 变更历史

- **v0.3**：工具收敛与命名空间优化，统一输入/输出协议，复合工具替代链式调用，token 效率策略，可解释性打分增强
- **v0.2**：新增"数据画像与字段映射"，补充"解析与抽取"分层、文化/药敏与 ABG 工具，扩展返回 Schema（extractions/units），细化 MVP 对应真实数据来源
- **v0.1**：首次方案整理，定义分层、统一返回 Schema、MVP 工具与映射示例



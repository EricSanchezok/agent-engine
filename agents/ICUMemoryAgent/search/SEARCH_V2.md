### V2 检索算法设计（医学语义增强）

#### 目标
- 在 ICU 场景中，利用 UMLS 等医学本体与知识库，将“纯文本相似 + 时间接近”的检索升级为“医学语义相关性 + 关系感知 + 轻量时间因子”的综合排序。
- 重点从“语义逻辑关系”出发（如 药物→治疗→疾病、检查→指示→病情、手术/护理→作用→部位/器官），而非仅仅“像不像”的文本相似。

#### 设计总览
- 双通道召回、语义重排：
  1) 文本/向量通道：沿用 ANN 召回作为回退与补充。
  2) 医学概念通道：对事件做 UMLS 概念抽取与规范化（CUI），基于概念重合、语义类型兼容性、知识图谱近邻与关系触发进行召回。
  3) 多因子融合排序：S = λ_concept·S_concept + λ_relation·S_relation + λ_text·S_text + λ_time·S_time。
- 运行形态：
  - 事件入库/首次检索时，进行“概念抽取+规范化+缓存”。
  - 检索时，以“概念倒排索引”做主召回，向量召回作补充；最终融合排序。

---

### 一、医学概念抽取与规范化（UMLS）

#### 1) 文本预处理
- 输入使用 `event_content`（已为英文翻译版即可）。
- 轻量清洗：小写化、去噪符号、合并空白；保留医学缩写与数值单位。

#### 2) 候选术语生成（两档方案）
- 方案 A（零依赖、快速落地）
  - 利用 n-gram（1-6）+ 停用词过滤 + 数字/单位短语保留，生成候选片段。
  - 按词频/启发式对候选排序（长词优先、含专业后缀优先）。
- 方案 B（更优识别，依赖三方）
  - 引入临床/生物医学 NER/EL 工具：scispaCy、QuickUMLS、MedSpaCy、cTAKES、SapBERT（EL）。
  - 能显著提升召回与精度，适合后续迭代。

#### 3) UMLS 查询与归一（使用 `core/umls/umls_client.py`）
- 对候选片段调用 `UMLSClient.search_all`：
  - 过滤或加权的 root_source：优先 `SNOMEDCT_US`（临床概念）、`RXNORM`（药物）、可扩展 `LNC`（检验）、`ICD10CM` 等。
  - 可选 searchType：`words`/`phrase` 以控粒度，并记录返回顺序得分作候选置信度。
- 调用 `get_cui_details` 获取 CUI 的 `semantic_types`（STY）。
- 可选 `map_cui_to_source_codes`：将 CUI 映射到特定词表（如 RxNorm/SNOMED）以利于下游计算与可解释性。

#### 4) 概念筛选与加权
- 语义类型白名单（示例，可根据 ICU 任务裁剪）：
  - 疾病/临床发现（Disease/Disorder, Finding）
  - 药物与化学物质（Pharmacologic Substance, Clinical Drug）
  - 手术/操作/护理（Procedure, Therapeutic/Preventive Procedure, Nursing）
  - 检验与生命体征（Laboratory Procedure, Sign/Symptom）
  - 解剖学部位/装置（Body Structure, Medical Device）
- 概念权重建议：
  - tf-idf（患者内/全库内）+ UMLS 返回置信度 + 语义类型先验（如“疾病/药物”权重高于“部位/形容词”）。
- 持久化到事件 `metadata.umls`：
  - concepts: [{cui, name, sty, sabs, weight, span(optional)}]
  - 可另外维护 per-patient 概念倒排索引（CUI → event_ids）。

---

### 二、关系感知的相似度构造

#### 1) 概念集合相似（S_concept）
- 基于 CUI 的加权重叠/向量化相似：
  - 加权 Jaccard：J_w = Σ_min(w_q, w_d)/Σ_max(w_q, w_d)。
  - 概念袋（CUI TF-IDF）余弦相似：cos(v_q, v_d)。
  - 概念嵌入相似：将“概念名称+同义词+定义”文本嵌入，求余弦（可缓存 CUI 向量）。
- 类型兼容性提升：若同类型 CUI 对齐（药↔药、病↔病）或典型跨类型（药↔病、检验↔病）匹配，施加加权因子。

#### 2) 关系触发相似（S_relation）
- 目标：反映“语义逻辑关系”，例如：
  - 药物→治疗→疾病（treats/therapy_for）
  - 检查/检验→指示/提示→病情（indicates/diagnoses）
  - 手术/护理→作用于→部位/疾病（acts_on/site_of）
- 实现路径（由简到繁）：
  - R0：类型先验启发式（无图谱时的“弱关系”）：若 q 含“疾病”，d 含“药物”，加分；若 q 含“症状”，d 含“检查/检验”，加分。
  - R1：同/近同概念层级关系（需要词表层级，如 SNOMED `isa`）：最小公共祖先深度或路径长度越短，得分越高。
  - R2：显式语义关系（需要关系数据）：treats/causes/indicates/complicates/contraindicated_with 等——若任意 q 的 CUI 与 d 的 CUI 在知识库中存在这些关系，则强加分。
  - R3：知识图谱路径相似：在 UMLS/SNOMED 图上做 Personalized PageRank 或最短路径/随机游走概率，近邻越近得分越高。
- 备注：
  - 当前 `UMLSClient` 未封装关系端点，可先落地 R0，再择机扩展 R1/R2（需要关系 API 或离线 MRREL/SNOMED 关系表）。

#### 3) 文本/向量相似（S_text）与时间因子（S_time）
- S_text：保留现有 `sim_vec` 作为弱信号（避免全失召回）。
- S_time：降权为辅助（如 λ_time≤0.15），只在分不出胜负时提供微弱排序稳定性。

#### 4) 融合与默认权重
- 统一评分：
  - S = λ_concept·S_concept + λ_relation·S_relation + λ_text·S_text + λ_time·S_time。
- 初始建议：
  - λ_concept=0.5、λ_relation=0.3、λ_text=0.15、λ_time=0.05。
  - 可按标注题集/医师反馈做网格/贝叶斯优化。

---

### 三、候选构造与召回策略

#### 1) 主召回：概念倒排
- 以 q 的 CUI 集合做倒排拉取：
  - 直接重合（共享 CUI）
  - 概念近邻（若已支持 R1/R2/R3：同义/上位/下位/显式关系的 1-2 跳邻居）
- 候选规模：topn_concept = max(top_k·10, 200)。

#### 2) 补充召回：向量 ANN
- 维持现有 `patient_mem.search`（topn_vec），与概念召回并集去重。
- 兼容“概念稀薄的事件”（如纯数值/模板化记录）。

#### 3) 过滤与去重
- 近重复（near_duplicate_delta）沿用。
- 去除无时间且无概念的低价值候选（可阈值化）。

---

### 四、数据结构与缓存

#### 1) 事件元数据扩展
- `metadata.umls`：
  - `concepts`: List[{`cui`, `name`, `sty`, `sabs`, `weight`, `span?`}]
  - 可加 `concept_vec`（概念袋嵌入）缓存标记。

#### 2) 倒排与向量缓存
- per-patient 倒排：`cui -> Set[event_id]`，可驻内存 + 周期性落盘（或复用 `ScalableMemory` 的一个专用“索引库”）。
- CUI 嵌入缓存：key = CUI，val = 向量；跨患者共享。
- UMLS 查询缓存：输入串/参数 → [候选 CUI]，Respect UMLS 频控。

---

### 五、查询流程（v2）

1) 取 `query_event_id` 对应事件，若无 `metadata.umls` 则执行概念抽取与缓存。
2) 主召回：按 q 的 CUI 用倒排取候选；若支持关系，加入 1-2 跳近邻。
3) 补充召回：向量 ANN（topn_vec），并集去重。
4) 逐候选计算：
   - S_concept（加权重叠/余弦/概念嵌入）
   - S_relation（类型先验/层级/显式关系/图谱距离）
   - S_text（现有 sim_vec）
   - S_time（指数衰减，低权重）
5) 融合排序，取前 `top_k`。

---

### 六、UMLS 使用与注意事项

- 语言：`event_content` 已英文，可直接用于 UMLS 检索；若未来保留原文，可在线翻译→再检索。
- 参数：
  - `root_source`：按场景优先（SNOMEDCT_US、RXNORM、LNC…）。
  - `searchType`：`words`/`phrase` 依文本形态调节召回。
- 频控与稳定：对 `search`/`get_cui_details`/`map_cui_to_source_codes` 做本地缓存；失败重试/退避与降级（仅概念重叠/仅向量）。
- 关系获取：
  - 短期用“类型先验 + 同/近义映射”支撑；
  - 中期扩展关系端点或离线导入 UMLS MRREL/SNOMED 关系；
  - 长期可引入 SemMedDB（SemRep 抽取的 UMLS 关系）增强临床关系覆盖。

---

### 七、评分细化建议

#### 1) S_concept（建议组合）
- S_concept = (1-η)·cos(CUI-tfidf_bow) + η·cos(embedding(concept_bag_text))；
- `concept_bag_text` 可拼接：概念首选名 + 2-3 同义词 + 语义类型短标签；
- η 初值 0.3。

#### 2) S_relation（建议优先级）
- R0（类型先验） > R1（层级近邻） > R2（显式关系） > R3（图谱距离）。
- 不同触发给不同阈值与加分：例如
  - 药↔病（treats/therapy_for）：+0.25
  - 检验↔病（indicates/diagnoses）：+0.20
  - 手术/护理↔部位/病灶（acts_on/site_of）：+0.15
  - 近义/同义（SY）：+0.12

#### 3) 正则化与截断
- 对各子项做 [0,1] 归一化；极端情况下用温和截断/Logit 压缩防止单因子支配。

---

### 八、评估与调参

- 评估集：以“逻辑相关性”构建小规模标注集（药→病、检验→病、护理→部位/病灶…）。
- 指标：nDCG@K、P@K、HitRate@K + 医师主观可解释性评分。
- 调参：网格/贝叶斯优化 λ、η、topn_concept/topn_vec；分类型加分阈值。

---

### 九、分阶段实施路线

- v2.0（快速版，零外部依赖）
  - A 级候选生成（n-gram）+ UMLS 检索与 STY 过滤；
  - 概念倒排 + S_concept（CUI tf-idf 余弦）+ R0（类型先验）+ S_text + 低权重 S_time。

- v2.1（概念识别增强）
  - 接入 scispaCy/QuickUMLS/SapBERT 任一；
  - 引入概念嵌入缓存，S_concept 融合文本化概念向量（η>0）。

- v2.2（关系增强）
  - 引入层级/显式关系（R1/R2：UMLS MRREL/SNOMED 关系或 SemMedDB）；
  - 图谱近邻召回与 R3（PPR/最短路径分数）。

---

### 十、可借鉴方法（学术/工业）

- 实体识别/链接：
  - QuickUMLS（基于词表的近似匹配，工业界常用）
  - scispaCy（生物医学 NER + 词表链接管道）
  - SapBERT（自对齐预训练，面向 UMLS 实体链接）
  - cTAKES / MedSpaCy（临床文本处理经典方案）

- 关系获取/推理：
  - SemRep / SemMedDB（从文献抽取的 UMLS 三元组）
  - SNOMED CT 分层/关系（isa、finding_site、associated_morphology…）
  - 知识图谱嵌入：TransE/RotatE/ComplEx，用于计算 CUI 近邻与路径相似
  - 图算法：Personalized PageRank、Random Walk with Restart、PathSim/PathCount

- 语义相似：
  - 概念词表向量化（CUIs 的定义/同义词文本）
  - 混合相似：概念袋余弦 + 图谱近邻 + 文本向量

---

### 十一、与 v1 的差异与兼容

- v1：单纯 sim_vec + time_score，易受模板/时间近邻干扰。
- v2：以“概念与关系”为主，文本与时间为辅；在无概念/关系时优雅降级到 v1。
- 存储层面：仅在 `metadata` 增加 `umls.concepts` 与可选缓存结构，对既有索引兼容。

---

### 十二、常见问题（FAQ）

- Q: 事件内容是英文翻译后文本，能直接查 UMLS 吗？
  - A: 可以，UMLS 面向英文术语；翻译质量影响召回，建议保留医用词汇原意与缩写。

- Q: 只做“概念重叠”会不会过严？
  - A: 因此设计加入“概念嵌入相似 + 关系近邻召回”以容错不同表达方式。

- Q: 没有关系 API 怎么办？
  - A: 先用类型先验（药↔病、检验↔病等）与同/近义映射；再逐步引入 MRREL/SNOMED/ SemMedDB。

---

### 十三、输出示例（排序因子解释）

- `reasons` 字段建议包含：
  - concept_overlap（加权重叠/Jaccard）
  - concept_cosine（CUI tf-idf 余弦）
  - concept_embed_cosine（若启用）
  - relation_triggers（触发的关系类型及来源：R0/R1/R2/R3）
  - sim_vec（文本向量相似）
  - time_score（时间衰减）
  - weights（融合权重快照）

---

以上即 v2 的整体实现思路。推荐先实现 v2.0 以尽快获得“医学语义可解释”的效果，再逐步升级识别与关系能力，在不破坏现有 v1 吞吐性能的前提下，提升临床相关性与可用性。



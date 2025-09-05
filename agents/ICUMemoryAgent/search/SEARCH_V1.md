### V1 检索算法说明

#### 目标
- 在 ICU 场景中，基于“查询事件”的内容相似度与时间接近度，检索相关事件并排序返回。

#### 输入参数（关键）
- `patient_mem`：患者专属的 `ScalableMemory`，包含事件文本、向量与元数据。
- `query_event_id`：查询事件的唯一 ID。
- `top_k`：最终返回的结果数（默认 20）。
- `window_hours`：v1 中不再生效（保持参数占位以兼容接口）；仅应用因果约束 `ts < q_t`。
- `weights`：加权系数（默认 `w1=0.5`、`w2=0.5`）。
- `tau_hours`：时间衰减参数 τ（默认 6.0）。
- `near_duplicate_delta`：近重复过滤阈值（默认 0.0 关闭）。若 >0，则过滤与查询向量相似度在 `1.0 - delta` 内的结果。

#### 核心概念
- **sim_vec**：向量相似度。由 ANN 检索返回的余弦距离 `d_cosine` 转换而来，`sim_vec = 1 - d_cosine`，范围约为 [-1, 1]，越接近 1 越相似。
- **time_score**：时间接近度分数，使用指数衰减：\( \text{time\_score} = \exp(-\lvert\Delta t\rvert / \tau) \)。
- **最终得分**：\( \text{score} = w_1 \cdot \text{sim\_vec} + w_2 \cdot \text{time\_score} \)。

#### 候选构造
1) 读取查询事件（`query_event_id`）的向量与时间戳；若查询向量缺失则直接返回空结果。

2) 向量召回（主召回）：
   - 对查询向量执行 ANN 检索：`patient_mem.search(q_vec, top_k=topn_vec)`，其中 `topn_vec = max(top_k * 10, 200)`。
   - 仅应用因果约束（保留 `ts < q_t` 的候选），不再进行任何窗口过滤。
   - 若启用 `near_duplicate_delta > 0`，则在 ANN 层面过滤与查询“几乎相同”的结果（例如 delta=0.01 会过滤 sim_vec ≥ 0.99 的条目）。

3) 时间窗口回召：
   - v1 不使用时间窗口过滤或补充候选。

4) 计算时间分数与最终得分：
   - \( \text{time\_score} = \exp(-\lvert\Delta t\rvert / \tau) \)，其中 \(\Delta t\) 为候选与查询的时间差（小时）。
   - \( \text{score} = w_1 \cdot \text{sim\_vec} + w_2 \cdot \text{time\_score} \)。

5) 排序与截断：
   - 按 `score` 降序排序，取前 `top_k`。

#### 为什么会看到 sim_vec = 0.0？
- 这通常表示该候选来自“时间窗口回召”，并未在“向量召回”中命中，为了参与重排而用 0.0 作为占位。
- 也可能存在极少数“向量几乎正交”的情况使相似度接近 0，但更常见的是前者（时间回召占位）。
- 即便 `sim_vec=0.0`，若时间非常接近（`time_score≈1`），该候选仍可能在最终排序中占较高位置。

#### 调参建议
- **召回覆盖**：增大 `topn_vec`（比如由 `top_k*5` 提高到 `top_k*10`）以减少仅时间回召的候选比例。
- **去重**：设置 `near_duplicate_delta`（如 0.01）过滤与查询“几乎相同”的重复事件。
- **权重**：提高 `w1`（向量）降低 `w2`（时间），让语义相似度影响更大；反之亦然。
- **时间衰减**：根据需要调整 `tau_hours`。τ 越大，时间影响越弱；越小，时间影响越强。
- **附加过滤（可选）**：可在排序前丢弃 `sim_vec==0.0` 的候选，或为 `sim_vec` 设定最小阈值（需在算法中额外实现）。

#### 典型调用
```python
results = await memory.search_related_events(
    patient_id=pid,
    event_id=ev_id,
    top_k=10,
    window_hours=24,
    tau_hours=6.0,
    version="v1",
    near_duplicate_delta=0.01,  # 可选：过滤近重复
)
```

#### 性能与注意事项
- 向量检索依赖 ANN 索引（优先使用 hnswlib），通常为毫秒级到十数毫秒级；
- 时间窗口回召当前通过遍历 `patient_mem.get_all()` 并逐条判窗，仅保留 `ts < q_t` 的事件；若库很大可能带来一定开销（可按需优化，如增加时间索引或缓存）。
- 若查询事件无时间戳，可正常进行向量检索，但时间分数将为 0。



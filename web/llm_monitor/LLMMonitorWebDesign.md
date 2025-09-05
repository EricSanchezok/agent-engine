### LLM Monitor Web 前端设计说明

#### 一、产品定位与目标
- **核心目标**：为开发与运维场景提供 LLM 对话监控的可视化界面，支持基于 `trace_id` 快速定位某次对话，查看上下文、参数、回复、用量、耗时与状态，并进行检索、筛选、导出与对比。
- **数据来源**：`agent_engine.llm_client.llm_monitor.LLMChatMonitor` 存储于项目根目录 `.llm_monitoring/llm_chats`（DuckDB/SQLite）。
- **设计原则**：
  - 可定位（trace-first）：所有视图均围绕 `trace_id` 可深链分享。
  - 可读性优先：默认折叠大文本，逐步展开；JSON 结构化展示。
  - 低耦合：前端通过统一 API 访问监控数据；后端薄封装 `ScalableMemory`。

#### 二、用户与关键场景
- **开发/调试**：
  - 通过 `trace_id` 或时间范围快速找到某次对话。
  - 对比两次对话（参数、消息、回复差异）。
  - 查看原始请求消息与回复全文，定位模型异常或提示词问题。
- **测试/验收**：
  - 批量筛选某模型/状态的对话，抽检内容与用量。
  - 按时间统计成功率、平均耗时、token 用量变化。
- **运维/监控**：
  - 观察错误峰值、异常时段，溯源到具体 `trace_id`。
  - 导出数据用于离线审计或复盘。

#### 三、信息架构与页面划分
- **1) Dashboard（概览）**：
  - 指标：总次数、成功率、平均/分位耗时、token 用量（输入/输出/总计）。
  - 趋势：按时间维度的请求量与错误率折线；模型占比、状态分布。
  - 快速筛选：时间范围、模型、状态、提供方（provider）。

- **2) Sessions（列表）**：
  - 列：`time(started_at)`、`trace_id`、`model`、`status`、`duration`、`input_tokens`、`output_tokens`、`total_tokens`、`user_prompt` 摘要。
  - 搜索/筛选：
    - 元数据精准过滤：`model_name`、`provider`、`status`、时间范围。
    - 关键字：对 `content` 进行 LIKE 检索（后端实现），用于临时全文查询。
  - 交互：固定列 `trace_id` 可复制；点击行进入详情；支持多选后导出 JSON/CSV。

- **3) Session Detail（详情）**：
  - 头部：`trace_id`（复制按钮）、状态徽章、时间与耗时、模型与 provider、主要参数（max_tokens/temperature）。
  - 分区：
    - Request：
      - Messages：按 `role` 分组展示（system/user/assistant），默认折叠长内容。
      - Params：结构化卡片与 JSON viewer 切换。
    - Response：回复全文（可复制/折叠），Usage（token 用量），Raw（可选）。
    - Timing：开始/结束时间线与 duration。
    - Metadata：`trace_id`、`model_name`、`provider`、`status` 等。
    - JSON：原始存储 `content` 的只读 JSON 视图（复制/下载）。
  - 关联：跳转至同一时间邻近的会话；按相同模型的最近 10 次。
  - 日志联动：显示 `trace_id` 并提供“打开日志文件”的本地链接或下载（若允许）。

- **4) Compare（对比）**：
  - 选择两个 `trace_id`，左右分栏对比：Messages、Params、Response、Usage、Timing；差异高亮。

- **5) Settings（设置）**：
  - 数据保留策略（展示与只读提醒）。
  - 导出策略（JSON/CSV 批量）。
  - 数据路径与后端信息（DB 类型、数量、大小）。

#### 四、交互细节与 UX
- 支持 URL 深链：`/sessions?trace_id=...` 直接定位详情；筛选条件持久化到 URL。
- 大文本处理：默认显示前 N 行，展开后异步加载剩余（防止阻塞渲染）。
- JSON Viewer：可折叠节点、复制/下载；在详情/对比中复用。
- 表格虚拟滚动：大列表流畅滚动；行高自适应摘要长度。
- 键盘操作：`/` 聚焦搜索，`Esc` 清空，`c` 复制 trace_id。
- 亮/暗模式：跟随系统或手动切换。

#### 五、后端 API 设计（最小可用）
说明：后端可用 FastAPI（或等价）薄封装 `ScalableMemory`（非向量）。DuckDB/SQLite 由现有实现托管。

- `GET /api/llm/sessions`
  - Query：`status?`、`model?`、`provider?`、`from?`、`to?`、`q?`（LIKE 关键字）、`offset?`、`limit?`
  - Return：`{ items: [{ trace_id, model_name, provider, status, started_at, ended_at, duration_ms, usage: {input_tokens, output_tokens, total_tokens}, user_prompt_snippet }], total }`

- `GET /api/llm/sessions/{trace_id}`
  - Return：`{ content, metadata }`（直接映射 `LLMChatMonitor.get_chat` 返回）。

- `GET /api/llm/stats`
  - Return：聚合指标（总量、成功率、耗时分位、token 汇总，按时间分桶）。

- `POST /api/llm/export`
  - Body：`{ trace_ids: string[] }` 或 `{ filter: {...} }`
  - Return：文件下载（JSON/CSV）。

实现要点：
- 列表查询基于 `get_all_metadata()` 结果做内存过滤或 DB 语句过滤（SQLite 已建 `idx_items_content`，DuckDB 用 `LIKE`）。
- 详情查询直接 `get_by_id(trace_id)`。
- 聚合可在 Python 层做（初版），后续需要时再用 SQL 聚合优化。

#### 六、性能与大文本策略
- 默认折叠 + 按需渲染：长消息/回复仅首屏渲染，展开后再渲染剩余。
- 分页/虚拟化：列表页面分页 + 虚拟滚动，避免一次加载过多行。
- 网络优化：接口支持 `fields` 精简返回字段（列表不回传全文）。
- 底层 DB：DuckDB 优先，SQLite 退化模式依然可用；需要时可加二级索引。

#### 七、安全与权限
- 本地开发默认开放；生产建议：
  - 只读 API，禁用删除/写入。
  - CORS 白名单；鉴权（Basic/OAuth）按需启用。
  - 脱敏开关：对可能包含敏感信息的字段做遮盖显示（可在后端开关）。

#### 八、可观测性与链路打通
- `trace_id` 作为统一关联键：前端可复制并在日志中检索。
- 若提供日志路径/下载接口，可一键查看该 `trace_id` 的相关日志片段（后端按行过滤返回）。

#### 九、技术栈建议
- 前端：React + Vite + TypeScript + Ant Design（或 Tailwind + Headless UI）。
- 状态管理：React Query（请求 + 缓存）+ URL 同步。
- 代码组织：
  - `pages/`（Dashboard/Sessions/Detail/Compare/Settings）
  - `components/`（FilterBar/SessionTable/SessionDetail/JSONViewer/Timeline/CompareDrawer/StatsCards/Charts）
  - `api/`（llm.ts：封装上述 API）
  - `utils/`（formatters、copy-to-clipboard、time helpers）

#### 十、实施计划（里程碑）
- M1（最小可用）
  - Sessions 列表 + 详情页（含 JSON Viewer、折叠查看）。
  - 基本筛选（状态/模型/时间）与关键字查询（LIKE）。
  - 通过 `trace_id` 深链到详情。
- M2（可用性增强）
  - Dashboard 指标与趋势；导出 JSON/CSV；对比视图。
  - 表格虚拟滚动与性能优化；暗色主题。
- M3（运维增强）
  - 日志联动、聚合分桶优化；权限/脱敏开关；批量工具。

#### 十一、数据契约样例
- 列表项：
```
{
  "trace_id": "llm_20250101_120102_123_abcd1234",
  "model_name": "o3-mini",
  "provider": "azure",
  "status": "success",
  "started_at": "2025-01-01T12:01:02.123Z",
  "ended_at": "2025-01-01T12:01:03.456Z",
  "duration_ms": 1333,
  "usage": {"input_tokens": 26, "output_tokens": 216, "total_tokens": 242},
  "user_prompt_snippet": "Say hello and include..."
}
```

- 详情：
```
{
  "content": { /* 与 LLMChatMonitor content 保持一致 */ },
  "metadata": { /* 与存储 metadata 保持一致 */ }
}
```

#### 十二、风险与边界
- DuckDB/SQLite 在极端大 JSON 文本下的查询性能；必要时改为分页读取/拆字段存储。
- LIKE 检索的性能与准确率有限；后续可加专用全文索引或外部搜索服务。
- 日志文件访问的安全边界（仅在本地开发开放）。



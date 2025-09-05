### LLM 对话监控设计说明（llm_client/llm_monitor.py）

#### 背景与目标
- **目标**: 为每次 LLM chat 创建可检索的、结构化的监控记录，避免在日志中直接输出超长上下文导致日志爆炸；支持根据一次 chat 的唯一 ID 快速定位所有上下文、模型与参数、回复、耗时与用量（若可用）等信息。
- **现状痛点**: 当前 `LLMClient._setup_monitoring` 会在 `logs/llm` 落很多散文件，不利于检索、聚合与清理，且无法与 `AgentLogger` 产生的日志条目形成一一对应。

#### 设计原则
- **可定位**: 每次 chat 对应一个全局唯一的 `trace_id`，日志与监控存储均使用该 ID 打通。
- **结构化**: 对话上下文、模型参数、回复内容、耗时、用量信息结构化保存为 JSON，便于查询与二次分析。
- **低侵入**: 在不改变业务调用方式的前提下，于 `LLMClient`/`AzureClient` 的关键位置插入监控钩子。
- **可配置**: 通过环境变量或构造参数开关监控；支持保留/裁剪/脱敏策略；可配置持久化库（DuckDB/SQLite）。
- **高可用**: 监控失败不影响主流程；写入使用 `ScalableMemory` 的本地 DB，`enable_vectors=False` 仅按 ID 检索，无向量化成本。

#### 架构概览
- 新增模块 `llm_client/llm_monitor.py`，提供 `LLMChatMonitor`：
  - 负责生成/接收 `trace_id`，并将本次对话的完整信息落库（`ScalableMemory`）。
  - 提供 start/complete/fail 三态写入，以及 get_by_id 查询能力。
- 对接 `AgentLogger`：
  - 在每次 chat 开始时由 `AgentLogger` 生成 `trace_id`（或由 Monitor 生成后回填给 Logger），所有日志行均附带该 `trace_id`。
  - 形成一致的可观测链路：日志 -> `trace_id` -> 数据库记录。

#### 存储方案（ScalableMemory）
- 使用 `agent_engine.memory.scalable_memory.ScalableMemory`：
  - `name`: `llm_chats`
  - `enable_vectors`: `False`（仅按 ID 查，不做向量化/ANN 索引）
  - DB: 默认优先 `DuckDB`，不可用时回退 `SQLite`（已在 `ScalableMemory` 内处理）
- 写入策略：
  - `content`: JSON 字符串（详见数据模型），存完整上下文与结果；
  - `metadata`: 轻量索引字段（如 `trace_id`、`model_name`、`status`、`timestamps`）便于快速过滤（后续可扩展元数据查询）。
  - `item_id`: 使用 `trace_id`，保证一对一更新（start -> complete/fail）。

#### 数据模型
- `item_id`（即 `trace_id`）：字符串，形如 `llm_20250101_120102_123_ABCDEF` 或短 UUID。
- `content`（JSON）示例：
```
{
  "trace_id": "llm_xxx",
  "provider": "azure",
  "model_name": "o3-mini",
  "request": {
    "system_prompt": "...",
    "user_prompt": "...",
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ],
    "params": {"max_tokens": 8000, "temperature": 0.7, "extra": {}}
  },
  "response": {
    "content": "...",
    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    "raw": null  
  },
  "timing": {"started_at": "ISO-8601", "ended_at": "ISO-8601", "duration_ms": 0},
  "status": "success"  
}
```
- `metadata`（JSON）建议字段：
  - `trace_id`: 与 `item_id` 相同
  - `model_name`: 例如 `o3-mini`
  - `provider`: 例如 `azure`
  - `status`: `pending` | `success` | `failed`
  - `started_at`, `ended_at`

说明：`usage` 如果底层 SDK 返回，用于记录 token 用量；若不可用可置空。`raw` 可选存放原始响应对象的安全摘取（避免超大/敏感字段）。

#### 模块设计（llm_client/llm_monitor.py）
- `class LLMChatMonitor`（依赖 `ScalableMemory`）
  - `__init__(name: str = "llm_chats", enable_vectors: bool = False, ...)`
  - `new_trace_id(prefix: str = "llm") -> str`: 生成全局唯一 ID（也支持从外部传入）。
  - `start_chat(trace_id, system_prompt, user_prompt, model_name, params, provider="azure")`
    - 写入一条 `status=pending` 的记录，包含 `started_at`；
  - `complete_chat(trace_id, response_text, usage=None, raw=None)`
    - 更新记录为 `success`，写入 `response`、`ended_at`、`duration_ms`；
  - `fail_chat(trace_id, error_message, raw=None)`
    - 更新记录为 `failed`，附加错误摘要与 `ended_at`；
  - `get_chat(trace_id) -> Dict[str, Any]`
    - 通过 `get_by_id` 返回结构化内容+元数据。

注意：`start/complete/fail` 都是幂等 upsert，使用同一个 `item_id=trace_id`。

#### 与 AgentLogger 集成
- 需求：每次 chat 时优先由 `AgentLogger` 产出一个 `trace_id`，日志与监控共用（与既有日志体系一致）。
- 首选方案：`AgentLogger` 提供 `new_event_id(prefix="llm")`（或等价接口）生成 `trace_id`；`AzureClient.chat` 将该 `trace_id`：
  - 附加到日志行：
    - `INFO "LLM chat start" trace_id=... model=...`
    - `INFO "LLM chat success" trace_id=... duration_ms=... tokens=...`
    - `ERROR "LLM chat failed" trace_id=... error=...`
  - 传给 `LLMChatMonitor.start_chat/complete_chat/fail_chat` 用作 `item_id`。
- 兜底方案：若短期不改动 `AgentLogger`，则由 `LLMChatMonitor` 生成 `trace_id` 并在日志信息里显式输出该值，保证链路一致。
- 可选增强：为 `AgentLogger` 增加 `context(correlation_id)`，自动把 `trace_id` 注入到该上下文内的所有日志前缀（非必须）。

#### 在 AzureClient/LLMClient 的接入点
- 移除/废弃 `LLMClient._setup_monitoring` 与 `_save_chat_monitoring` 的散文件写入逻辑。
- 在 `AzureClient.chat`：
  1) 生成 `trace_id`；
  2) `monitor.start_chat(...)`（包含 messages/params）；
  3) 调用底层 `call_llm`；
  4) 成功：提取 `content` 与可用 `usage`（若 SDK 提供），`monitor.complete_chat(...)`；
  5) 失败：`monitor.fail_chat(...)`；
  6) 所有日志 `self.logger.info/error` 中包含 `trace_id`。

示例日志（仅示意，必须英文输出）:
- `INFO LLM chat start: trace_id=llm_... model=o3-mini max_tokens=8000`
- `INFO LLM chat success: trace_id=llm_... duration_ms=532 usage={...}`
- `ERROR LLM chat failed: trace_id=llm_... error=RateLimitError(...)`

#### 查询与调试流程（开发/测试）
- 已知 `trace_id`：调用 `monitor.get_chat(trace_id)` 直接获得完整记录。
- 从日志跳转：在日志中复制 `trace_id`，在调试脚本或 REPL 中查询。
- 扩展：后续可在 `metadata` 增加更多精简索引字段，以支持基于 `metadata` 的过滤（如按模型/时间范围）。

#### 失败场景与降级
- 监控写入失败（DB 打不开/序列化失败）不影响主请求；记录一条 `logger.warning`。
- `trace_id` 未生成：由 `LLMChatMonitor` 兜底生成。
- 超长内容策略：
  - 初期直接写入 JSON（SQLite/DuckDB 可承载大文本）；
  - 可选：对 `response.content` 与超长 `messages` 采用截断或 gzip+base64 压缩（后续开关）。

#### 配置项（建议）
- `AGENT_ENGINE_LLM_MONITOR_ENABLED`（默认 `true`）
- `AGENT_ENGINE_LLM_MONITOR_NAME`（默认 `llm_chats`）
- `AGENT_ENGINE_LLM_MONITOR_REDACT`（默认 `false`，开启时对敏感字段做脱敏）
- `AGENT_ENGINE_MEMORY_DB`（已存在：`duckdb` | `sqlite`）

#### 迁移计划
1) 引入 `llm_client/llm_monitor.py` 与最小依赖；
2) 在 `AzureClient.chat` 替换旧 `_save_chat_monitoring` 路径；
3) 保留旧日志输出，但统一添加 `trace_id`；
4) 验证：新建一次 chat，使用 `trace_id` 通过 `get_by_id` 读取记录，确认字段齐全；
5) 移除 `LLMClient._setup_monitoring` 与 `_save_chat_monitoring` 及旧目录清理逻辑。

#### 安全与隐私
- 默认不做脱敏；可通过开关启用对提示词/回复中的敏感信息遮盖。
- 不在日志中输出完整上下文，仅输出 `trace_id` 与简要参数。

#### 后续扩展
- 简单的 `find_chats(metadata_filter)` 封装（利用已存在的 DB 访问能力）。
- 统一消费 `usage/cost` 指标，形成报表。
- 与外部可观测系统对接（如 OTLP trace），`trace_id` 保持兼容。



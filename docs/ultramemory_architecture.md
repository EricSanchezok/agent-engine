## UltraMemory 架构设计（独立、专业、服务化的一体化存储）

UltraMemory 是一个全新设计的统一 Memory 体系，独立于现有实现，专注于“向量检索、时序存储、通用关系/JSON 查询”。它既支持本地嵌入式调用，也内置 HTTP/gRPC 服务，配置端口即可一键部署，让外部机器可直接访问。面向高并发读取、强过滤与灵活 Schema 的工业化场景。

### 1. 愿景与设计目标
- **专业性**：原生支持向量/时序/关系三大工作负载，并在各自领域采用最佳索引与执行策略。
- **服务化**：除本地 SDK 外，内置服务（HTTP/gRPC），支持鉴权、限流、指标监控与健康检查。
- **可插拔后端**：按需选择/切换后端（PostgreSQL+pgvector、TimescaleDB、Qdrant、Milvus、SQLite VSS 等）。
- **高并发与强过滤**：连接池、批量写、表达式/JSONB 索引、ANN 索引、时序分区/压缩/保留策略。
- **自由 Schema**：高频字段类型化，长尾字段用 JSON/属性存储；提供集合级别的 Schema 管理。

### 2. 能力矩阵（Capabilities）
- **Vector ANN**：HNSW/IVF 索引、文本->向量代理（可选）、属性过滤、近重复控制。
- **Timeseries**：分区/压缩/保留策略、连续聚合/下采样、窗口与聚合查询。
- **Relational/JSON**：自由 Schema、JSONB/GIN（PG）、表达式索引、复杂过滤与排序分页。
- **Ops**：批量 upsert、连接池、读写分离（可选）、健康检查、指标、审计与限流。

### 3. 核心抽象与命名
- **Collection**：逻辑集合（命名空间），绑定一种或多种能力（vector/timeseries/general）。
- **Entity（通用记录）**：`id`、`attributes: dict`、`content?`、`vector?`、`timestamp?`。
- **Point（时序点）**：`metric`、`timestamp`、`tags: dict[str,str]`、`fields: dict[str,number|bool|string]`。
- **Query DSL**：统一过滤（`eq/in/range/and/or/not/time_range/order_by/limit/offset`），向量检索参数（`top_k/threshold/ef_search`）。

### 4. 架构分层
- **API 层（SDK + Service）**：
  - SDK：Python/后续多语言；方法稳定，屏蔽后端差异。
  - Service：内置 HTTP/gRPC 服务，支持认证、限流、监控；一键启动对外提供能力。
- **路由与执行计划（Router/Planner）**：
  - 根据 Collection 能力与后端映射生成执行计划（SQL/HTTP/gRPC）。
- **存储适配层（Storage Adapters）**：
  - PostgreSQL（关系/JSON）、pgvector（向量）、TimescaleDB（时序）、Qdrant/Milvus（服务化向量）、SQLite VSS（嵌入式回退）。
- **索引与模式管理（Index/Schema Manager）**：
  - 建表/索引/表达式/JSONB/ANN/hypertable；向量维度/度量；保留与压缩策略。
- **摄取与缓存（Ingestion/Cache）**：
  - 批写、流式摄取、短期结果缓存（TTL）、热点读加速。
- **可观测与安全（Observability/Security）**：
  - 指标、日志、追踪；API Key/JWT、RBAC、TLS。

### 5. 数据模型与 Schema 策略
- **类型化列 + JSON 混合**：高频过滤字段提升为显式列；长尾字段进入 JSONB/属性映射。
- **向量字段**：固定维度，集合级配置；可支持多向量列（如 `text_vec`、`image_vec`）。
- **时序字段**：强制 `timestamp`；按时间分区（hypertable/chunk），可附派生列（`yyyymm`）或函数索引。

### 6. 统一 API（高层）
- 写入/更新：
  - `create_collection(spec)`、`describe_collection(name)`、`drop_collection(name)`
  - `upsert(records, collection, batch_size=?, dedupe_key=?, vector_autogen=?)`
  - `append_points(points, metric, retention=? , batch_size=?)`
- 读取/检索：
  - `get(id, collection)`、`mget(ids, collection)`
  - `search_vectors(vector|text, collection, top_k, filter?, threshold?, ef_search?)`
  - `query(collection, filter, projection?, order_by?, limit?, offset?)`
  - `query_timeseries(metric, time_range, group_by?, aggregate?, downsample?)`
- 维护/诊断：
  - `stats()`、`create_indexes()`、`compact()`、`set_retention()`、`health()`

### 7. HTTP/gRPC 服务接口（示意）
- 基础路由（HTTP）：
  - `POST /v1/collections`、`GET /v1/collections/{name}`、`DELETE /v1/collections/{name}`
  - `POST /v1/{collection}:upsert`（批量）
  - `POST /v1/{collection}:search`（向量检索）
  - `POST /v1/{collection}:query`（通用过滤）
  - `POST /v1/timeseries/{metric}:append`、`POST /v1/timeseries/{metric}:query`
  - `GET /healthz`、`GET /metrics`
- 安全：`Authorization: Bearer <token>`、TLS（可选）、租户隔离（可选）。

### 8. 后端适配与选型建议
- PostgreSQL（通用）
  - JSONB/GIN、表达式索引、窗口/聚合；批量 COPY/批 upsert；读副本适配。
- pgvector（向量）
  - 列类型 `vector(dim)`；HNSW/IVF；向量+标量同库同查询。
- TimescaleDB（时序）
  - hypertable/压缩/连续聚合/保留策略；对 SQL 工作流友好。
- Qdrant（服务化向量）
  - HNSW、payload filter、分片/副本；HTTP/gRPC；易运维与快照。
- Milvus（服务化向量）
  - IVF/HNSW/PQ，面向海量规模与集群；需要运维基础。
- SQLite VSS（嵌入式）
  - 轻量回退；开发/边缘场景可用。

### 9. 索引与查询优化策略
- 向量：HNSW（默认）或 IVF；按数据量自动调参（`ef_search`、`M`、`nlist`）。
- JSON/属性：PG 首选 JSONB+GIN；高频过滤列/函数索引（如 `to_char(timestamp,'YYYYMM')`）。
- 时序：按时间分区 + 压缩；大窗口走连续聚合；冷热分层与保留策略。

### 10. 并发、吞吐与一致性
- 连接池：按后端配置池大小；HTTP 服务层支持并发 worker。
- 批量写入：统一批尺寸与并发度；支持 backpressure 与重试，幂等键保障去重。
- 读扩展：读副本/分片（依赖后端能力）；服务层只读路由（可选）。
- 一致性：遵循后端事务隔离；对服务化请求提供重试与幂等。

### 11. 部署形态与一键启动
- 嵌入式（Library）
  - 在应用内直接实例化 UltraMemory，连接本地或远程后端。
- 服务化（Service）
  - 内置 HTTP/gRPC 服务器，支持端口、主机、worker、TLS、鉴权、连接池配置。
  - 在本项目中，可通过 `./run.bat <server_entry>.py --port=8080` 一键启动（示意）。
  - 提供 Dockerfile 与 K8s 清单（可选），便于跨机器访问与弹性伸缩。

### 12. 可观测性与安全
- 可观测性：/metrics（Prometheus）、结构化日志、分布式追踪（可选）。
- 安全：API Key/JWT、RBAC/租户、IP 白名单、TLS 证书热更新（可选）。

### 13. 配置示例（YAML）
```yaml
ultramemory:
  service:
    enabled: true
    host: 0.0.0.0
    port: 8080
    workers: 8
    tls:
      enabled: false
  mode: mixed              # vector | timeseries | general | mixed
  backend: postgres_pgvector # postgres_pgvector | timescaledb | qdrant | milvus | sqlite_vss
  connection:
    dsn: postgresql://user:pass@host:5432/db
    pool_min: 4
    pool_max: 32
  vector:
    dimensions:
      text_vec: 1536
    default_index: hnsw
    params:
      m: 16
      ef_construction: 200
  timeseries:
    default_metric: events
    retention_days: 365
    continuous_aggregate: P1D
  schema:
    elevate_fields: ["categories", "yyyymm"]
    json_fields: ["metadata"]
  tuning:
    batch_size: 1000
    enable_cache: true
    cache_ttl_seconds: 300
```

### 14. 统一 Filter DSL（示例）
```json
{
  "and": [
    {"eq": ["categories", "cs.LG"]},
    {"range": {"timestamp": ["2024-06-01T00:00:00Z", "2024-06-30T23:59:59Z"]}}
  ],
  "order_by": [["timestamp", "desc"]],
  "limit": 100
}
```

### 15. 路线图
- **v0**：PG+pgvector、TimescaleDB、SQLite VSS；HTTP 服务；统一 DSL；批写/索引策略。
- **v1**：Qdrant/Milvus；只读副本/分片路由；多租户/配额；更细粒度缓存与限流。
- **v2**：联邦查询（多集合/多后端）；跨 Region 容灾；在线 Schema 演化工具。

### 16. 风险与权衡
- 多后端抽象需权衡“最小公分母”与“最佳特性”的冲突；通过能力探针与条件降级缓解。
- 服务化增加网络与治理成本，但可显著提升并发读与可运维性。

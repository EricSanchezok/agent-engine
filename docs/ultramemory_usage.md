## UltraMemory 使用手册（Windows/macOS/Ubuntu + Docker + PostgreSQL/pgvector）

### 目标受众
- 希望在本机/内网快速搭建可持久化的向量/关系存储，并通过简单 Python API 或 HTTP 服务访问的开发者。

### 前置条件
- Windows：已安装 Docker Desktop；macOS：Docker Desktop；Ubuntu：docker engine/service。
- 已通过 `uv sync` 同步依赖（本项目已切换为 `psycopg[binary]`，Windows 无需额外安装 libpq）。
- 可访问公共镜像（或已配置镜像加速/私有镜像）。

### 一键启动 PostgreSQL + pgvector（Docker）
- 代码已内置工具，自动创建/复用容器并返回 DSN（连接串）。
- 默认容器名：`ultramemory-pg`，默认主机端口：`55432`（避免与本机 5432 冲突）。

示例（Python）：
```python
from agent_engine.memory.ultra_memory.docker_pg import ensure_docker_postgres, DockerPGConfig

dsn = ensure_docker_postgres(DockerPGConfig())
print("dsn:", dsn)  # 形如 postgresql://ultra:ultra@127.0.0.1:55432/ultra
```

注意：
- Windows：工具会尝试启动 Docker Desktop 并等待；macOS/Ubuntu 需自行确保 docker 守护进程已启动（Ubuntu 可能需要 `sudo systemctl start docker`，或将用户加入 `docker` 组）。
- 首次运行会拉取镜像 `pgvector/pgvector:pg16`（失败将尝试 ghcr、daocloud 镜像）。
- 如网络受限：在 Docker Desktop → Settings → Docker Engine 配置镜像加速；或传入 `image_candidates=[...]` 使用私有镜像。

### Python 端使用 UltraMemory（单后端 = 单实例）
```python
from agent_engine.memory.ultra_memory import UltraMemory, UltraMemoryConfig, CollectionSpec, Record, Filter
from agent_engine.memory.ultra_memory.docker_pg import ensure_docker_postgres

# 1) 获取 DSN（或者自备 DSN）
dsn = ensure_docker_postgres()  # postgresql://ultra:ultra@127.0.0.1:55432/ultra

# 2) 初始化 UltraMemory（一个实例只连接一个后端）
um = UltraMemory(UltraMemoryConfig(
    backend="postgres_pgvector",
    dsn=dsn,
))

# 3) 创建集合（定义向量列维度等）
um.create_collection(CollectionSpec(
    name="papers",
    mode="vector",
    vector_dimensions={"text_vec": 3},  # 示例：3 维，实际用 768/1024/1536 等
))

# 4) 写入/更新（upsert）
ids = um.upsert("papers", [
    Record(id="p1", attributes={"categories": "cs.AI"}, content="alpha",  vector=[1, 0, 0]),
    Record(id="p2", attributes={"categories": "cs.LG"}, content="bravo",  vector=[0.9, 0.1, 0]),
])
print("upsert ids:", ids)

# 5) 按属性过滤查询（Filter DSL）
rows = um.query("papers", Filter(expr={"eq": ["categories", "cs.LG"]}, limit=10))
print("query:", rows)

# 6) 向量检索（top_k + 属性过滤）
res = um.search_vectors(
    "papers",
    vector_or_text=[1, 0, 0],
    top_k=2,
    flt=Filter(expr={"in": ["categories", ["cs.AI", "cs.LG"]]}),
)
print("search:", res)

# 7) 基础运维信息
print("stats:", um.stats())
print("health:", um.health())
### 多实例（多个 UltraMemory）的创建方式
- 推荐：一个后端对应一个 UltraMemory（一个 dsn），需要多个后端就创建多个 UltraMemory。每个 UltraMemory 下可创建多个 collection（同一后端、能力一致）。
- 强隔离：用不同的 Docker 配置启动多个容器，分别返回 DSN，分配给各自 UltraMemory 实例。

示例（同 DSN，多集合）：
```python
dsn = ensure_docker_postgres()
um = UltraMemory(UltraMemoryConfig(backend="postgres_pgvector", dsn=dsn))
um.create_collection(CollectionSpec(name="papers", mode="vector", vector_dimensions={"text_vec": 1536}))
um.create_collection(CollectionSpec(name="images", mode="vector", vector_dimensions={"img_vec": 512}))
```

### 名词解释：collection、表、attributes
- **collection（集合）**：逻辑数据集合，对应数据库中的“一张表”。不同集合拥有各自的 schema（向量列维度、索引等）。
- **表（table）**：在 PostgreSQL 中会创建 `um_<集合名>` 的表名，包含固定列 `id/content/attributes/ts` 与自定义的向量列（来自 `vector_dimensions`）。TimescaleDB 会创建 hypertable（`ts/tags/fields`）。
- **attributes**：JSONB 存储的自由属性字段，便于扩展与灵活过滤；高频字段可“提升”为显式列（后续可扩展表达式索引等）。

### 去重策略（dedupe_key）与 upsert 语义
- upsert 的唯一性基于 `id`（PRIMARY KEY）。
- 新增参数 `dedupe_key`（可选）：当未显式传入 `id` 时，若指定了 `dedupe_key`，则会从 `attributes[dedupe_key]` 读取该值作为 `id`（例如用业务主键），实现稳定去重；若 `dedupe_key` 不存在且仍未提供 `id`，会回退用 attributes 的 JSON 串生成一个 `id`。
- upsert 更新规则：同 `id` 再次写入会覆盖已有行的 `content/attributes/ts/向量列`。

### 删除数据
- 删除整集合：`drop_collection(name)`（删除该集合对应的表）。
- 删除单条：`delete_by_id(collection, id)`（新增）
- 条件删除：`delete_by_filter(collection, Filter)`（新增），例如 `{"eq": ["categories", "cs.AI"]}` 或 `{"range": {"timestamp": ["2025-01-01", "2025-01-31"]}}`。

```

### HTTP 服务（可选）
UltraMemory 内置最小 HTTP 服务，方便跨机器访问。

示例脚本（仅示例，不会自动创建）：
```python
# ultramemory_server_demo.py
from agent_engine.memory.ultra_memory.service_http import serve

cfg = {
  "backend": "postgres_pgvector",
  "dsn": "postgresql://ultra:ultra@127.0.0.1:55432/ultra",
  "mode": "mixed"
}

serve(host="0.0.0.0", port=8080, config=cfg)
```

运行（Windows PowerShell）：
```bash
./run.bat ultramemory_server_demo.py
```

HTTP 接口（示例）：
- 健康检查：`GET /healthz`
- 统计信息：`GET /stats`
- 创建集合：`POST /v1/collections`（JSON：`{ name, mode, vector_dimensions, ... }`）
- 批量 upsert：`POST /v1/{collection}:upsert`（`{"records": [...]}`）
- 过滤查询：`POST /v1/{collection}:query`（`Filter` JSON）
- 向量检索：`POST /v1/{collection}:search`（`{"vector": [...], "top_k": 5, "filter": {...}}`）

### Filter DSL 速查
- 逻辑：`{"and": [ ... ]}`、`{"or": [ ... ]}`、`{"not": { ... }}`
- 比较：`{"eq": ["field", value]}`、`{"in": ["field", [v1, v2]]}`、`{"range": {"field": [lo, hi]}}`
- 文本：`{"like": ["field", "%pattern%"]}`
- JSON 包含（attributes）：`{"contains": {"k": "v"}}`
- 排序分页：`order_by=[["timestamp", "desc"]]`、`limit`、`offset`

字段路径说明：
- `id`、`content`、`timestamp(ts)` 为显式列。
- 其它属性默认在 `attributes` JSON 中，支持点号路径（如 `author.name`）。

### 常见问题与排错
- 运行时回退为内存模式（in_memory）：
  - 确认 `pyproject.toml` 依赖为 `psycopg[binary]`，并执行 `uv sync`。
  - 确认 DSN 正确、PostgreSQL 可连通，`vector` 扩展存在（工具会尝试 `CREATE EXTENSION IF NOT EXISTS vector`）。
- 无法拉取镜像：
  - 在 Docker Desktop 配置镜像加速；或使用 `DockerPGConfig(image_candidates=[...])` 指定可访问镜像；或预拉取/离线导入镜像后重试。
- 端口冲突：
  - 修改 `DockerPGConfig.port`（如 55433），或停止占用 55432 的进程。
- 向量检索报错 `invalid input syntax for type vector`：
  - 确认传入的向量是 `[float, ...]`，维度与建表时定义一致。

### 数据持久化与清理
- 数据保存在 Docker 卷：`ultramemory_pg_data`。
- 停止容器（数据保留）：
```bash
docker stop ultramemory-pg
```
- 删除容器（保留数据卷）：
```bash
docker rm ultramemory-pg
```
- 删除数据卷（危险，清空数据）：
```bash
docker volume rm ultramemory_pg_data
```

### 参考测试
- 直接体验真实数据库落盘：
```bash
./run.bat test/ultramemory_docker_pg_test.py
```
该脚本会自动启动/复用 Docker PostgreSQL，并运行 upsert/query/search。输出中的 `PostgresPgvectorAdapter initialized (postgres)` 表示正在使用真实数据库。



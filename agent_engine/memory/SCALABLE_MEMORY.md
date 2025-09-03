### ScalableMemory 使用说明

ScalableMemory 是一个高性能、可扩展的向量记忆组件，专为 Agent/RAG 场景设计：
- 优先使用 ANN 检索（hnswlib），并自动回退到 Annoy 或暴力检索。
- 持久化存储优先使用 DuckDB，回退 SQLite（开启 WAL 以提升并发）。
- 支持自定义 ID、upsert、批量导入、元数据过滤与相似度阈值。
- 可选接入外部 `llm_client` 的 `embedding` 方法进行向量化；未提供时回退内置 `Embedder`。
- 线程安全（读写分离 RWLock）、索引自动持久化与按需重建。

---

### 可选依赖
- hnswlib（推荐，最佳检索性能）
- annoy（次优，构建/查询较稳定）
- duckdb（推荐，持久化与并发更友好）

未安装上述依赖也可运行，系统将自动回退到性能较低的替代方案。

---

### 快速开始
```python
from agent_engine.memory.scalable_memory import ScalableMemory

# 基本用法（默认优先 DuckDB + HNSW，如不可用自动回退）
mem = ScalableMemory(name="agent_rag")

item_id = mem.add("Large Language Models are helpful", metadata={"source": "doc1"})
results = mem.search("helpful models", top_k=5, threshold=0.3)
print(results)  # [(content, similarity, {"id": ..., "source": ...}), ...]
```

---

### 使用外部 llm_client 进行嵌入
当提供 `llm_client` 与 `embed_model` 时，ScalableMemory 在内部向量化时会优先调用其 `embedding` 方法（异步接口已内置同步封装）。

```python
from agent_engine.llm_client.azure_client import AzureClient
from agent_engine.memory.scalable_memory import ScalableMemory

azure = AzureClient(api_key="<YOUR_API_KEY>")

mem = ScalableMemory(
    name="rag_with_llm",
    llm_client=azure,                  # 可选：使用外部客户端做嵌入
    embed_model="text-embedding-3-large",  # 可选：传给 llm_client.embedding 的模型名
    index_backend="hnswlib",          # 可选：hnswlib | annoy | bruteforce
    db_backend="duckdb",              # 可选：duckdb | sqlite
)

mem.add("RAG combines retrieval and generation", metadata={"source": "doc3"})
print(mem.search("What is RAG?", top_k=3, threshold=0.4))
```

说明：
- 如果提供了 `llm_client`，将优先使用其 `embedding(text | List[text], model_name=embed_model, **kwargs)`。
- 未提供时回退到内置 `Embedder`（Sentence Transformers 或 TF-IDF fallback）。

---

### 主要 API

- add
```python
item_id = mem.add(
    content="Transformers are attention-based models",
    metadata={"source": "doc2"},
    # item_id="自定义id"  # 可选，不传则默认使用 content+metadata 哈希
)
```

- add_many（批量导入）
```python
ids = mem.add_many([
    {"content": "A", "metadata": {"source": "s1"}},
    {"content": "B", "id": "custom-1"},
])
```

- upsert（按 id 更新或插入）
```python
mem.upsert(item_id, metadata={"tag": "updated"})
```

- search（文本或向量查询）
```python
# 文本查询
results = mem.search("attention models", top_k=5, threshold=0.35, metadata_filter={"source": "doc2"})

# 向量查询
query_vec = [0.01, 0.02, ...]  # 与存储维度一致
results = mem.search(query_vec, top_k=5, threshold=0.5)

# 结果格式: [(content, similarity, metadata_with_id), ...]
```

- 读取与删除
```python
content, vector, meta = mem.get_by_id(item_id)
vec, meta = mem.get_by_content("some text")    # 使用 md5(content) 作为 id 兼容旧逻辑
first_content, first_meta = mem.get_by_vector(query_vec, threshold=0.9)

mem.delete_by_id(item_id)
mem.delete_by_content("some text")
mem.delete_by_vector(query_vec, threshold=0.9)
```

- 统计与遍历
```python
print(mem.count())
print(mem.get_all_contents())
print(mem.get_all_vectors())     # {id: vector}
print(mem.get_all_metadata())    # [{..."id": id}, ...]
print(mem.get_info())            # 基本信息（索引/存储/维度/数量等）
```

- 清理
```python
mem.clear()  # 清空 items，保留 labels 映射；索引将按需重建
```

---

### 配置与持久化
- 默认持久化目录：`<project_root>/.memory/<name>/`
  - 数据库：`<name>.duckdb` 或 `<name>.sqlite3`
  - 索引：`index_hnsw.bin` / `index_annoy.ann` / `index_bruteforce.json`
- 初始化参数：
  - `index_backend`: `hnswlib` | `annoy` | `bruteforce`
  - `db_backend`: `duckdb` | `sqlite`
  - `persist_dir`: 自定义持久化目录
  - `llm_client`, `embed_model`: 外部嵌入配置
- 环境变量（可选）：
  - `AGENT_ENGINE_MEMORY_DB`: 选择数据库（`duckdb`|`sqlite`）
  - `AGENT_ENGINE_MEMORY_INDEX`: 选择索引（`hnswlib`|`annoy`|`bruteforce`）
  - `AGENT_ENGINE_EMBED_MODEL`: 作为默认 `embed_model`
- 索引持久化：默认每 ~2s 至多持久化一次（批量导入时自动节流）。

---

### 性能建议
- 优先安装并使用 `hnswlib` 与 `duckdb`。
- 批量导入使用 `add_many`，减少 I/O 与索引写次数。
- 对 HNSW 可通过 `index_params` 调整（如 `M`, `ef_construction`），查询可传入 `ef_search` 提升召回。
- 在高并发读取场景下，读多写少的负载受益于内部 RWLock 设计。

---

### 兼容与迁移
- 不修改原 `agent_engine/memory/memory.py`。需要迁移时可将原调用替换为 `ScalableMemory`。
- `get_by_content` 仍使用 `md5(content)` 作为默认 id，兼容旧逻辑。

---

### 常见问题（FAQ）
- 维度如何确定？
  - 若配置了 `llm_client`，系统会通过一次探针嵌入推断维度；失败时回退到内置 `Embedder` 的固定维度。
- 如何重建索引？
  - 删除对应的索引文件后重新初始化，或调用 `clear()` 再批量写入进行重建。
- 没有安装 hnswlib/annoy/duckdb 会怎样？
  - 系统按顺序回退到仍可用的后端，功能可用但性能会下降。



## UMLSClient 使用说明

本模块位于 `core/umls/umls_client.py`，提供对 UMLS UTS API 的专业封装，内置：
- 分页聚合
- 指数退避重试（429/5xx）
- 类型注解返回（便于上层调用）
- 使用项目 `AgentLogger` 统一输出日志

### 环境变量
- `UMLS_API_KEY`: UTS Profile 的 API Key（支持 `.env` 注入）

### 快速开始
```python
from core.umls import UMLSClient

client = UMLSClient(api_key="YOUR_UTS_API_KEY")

# 1) 基础搜索（仅第一页，返回原始 JSON）
raw = client.search("sepsis", root_source="SNOMEDCT_US", search_type="exact")

# 2) 聚合搜索（跨页聚合，返回结构化结果）
all_results = client.search_all("sepsis", root_source="SNOMEDCT_US", search_type="exact")

# 3) 直接得到 CUI 列表（跨页聚合）
cuis = client.search_cuis("sepsis", root_source="SNOMEDCT_US", search_type="exact")

# 4) CUI 详情（含语义类型）
detail = client.get_cui_details(cuis[0])

# 5) CUI 映射到源词表编码（如 SNOMED / RxNorm）
codes = client.map_cui_to_source_codes(cuis[0], sabs=["SNOMEDCT_US","RXNORM"]).results
```

### API 区别
- `search(query, ...) -> dict`
  - 调用 `/search/{version}`，仅返回“单页”原始 JSON。
  - 适合需要完整原始返回并自行处理分页的场景。

- `iterate_search(query, ...) -> Generator[List[UMLSResult]]`
  - 迭代器方式自动翻页，每页产出结构化结果列表 `UMLSResult`。

- `search_all(query, ...) -> List[UMLSResult]`
  - 自动跨页聚合，直接返回“全部结果”的结构化列表。
  - 适合一般用途（简洁、拿来即用）。

- `search_cuis(term, ...) -> List[str]`
  - 在默认返回类型下，UI 通常即为 CUI；封装后直接返回 CUI 列表（跨页聚合）。
  - 适合“把术语快速转为 CUI”的场景。

- `get_cui_details(cui) -> CUIDetails`
  - 调用 `/content/{version}/CUI/{cui}`，获取 CUI 名称与语义类型（`semantic_types`）。

- `map_cui_to_source_codes(cui, sabs=[...]) -> CodeSearchResult`
  - 通过 `returnIdType=code` 的搜索，将 CUI 映射到指定源词表编码；结果中 `ui` 即代码。

### 测试脚本
- 见 `test/test_umls_client.py`，可用命令：
```
run.bat test\test_umls_client.py
```
脚本会自动加载 `.env`，依次演示 `search`、`search_cuis`、`get_cui_details`、`map_cui_to_source_codes`。

### 在 ICUMemoryAgent 中的用途（建议）
- 概念抽取后的规范化：
  - 通过 `search_cuis` 将中文/英文术语规范化到 CUI
  - 使用 `map_cui_to_source_codes` 对接 LOINC / RxNorm / SNOMED 等编码，便于 `concept_overlap` 与临床规则匹配

### 注意
- 不同词表的可用性与授权有所不同（如 SNOMED CT）；请确保遵循目标词表许可条款。
- UMLS 的搜索结果受 `searchType` / `rootSource` / `sabs` 影响，建议按需调参。



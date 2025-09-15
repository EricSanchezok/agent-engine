# GitHub Fetcher

一个专注于GitHub仓库数据获取的工具，提供简洁的接口来搜索和检索GitHub仓库，使用GitHub API而不与存储或数据库关注点耦合。

## 功能特性

- 🔍 **仓库搜索**: 使用灵活的查询参数搜索仓库
- 📊 **仓库详情**: 获取特定仓库的详细信息
- 👤 **用户仓库**: 获取特定用户的仓库
- 🏢 **组织仓库**: 获取组织的仓库
- 🔄 **速率限制**: 内置速率限制和重试逻辑
- 🛡️ **错误处理**: 强大的错误处理和重试机制
- 🚀 **异步支持**: 完整的async/await支持，高性能
- 🔑 **身份验证**: 支持GitHub个人访问令牌

## 安装

GitHub Fetcher是agent-engine包的一部分。确保您有所需的依赖项：

```bash
pip install aiohttp requests python-dotenv
```

## 快速开始

### 基本用法

```python
import asyncio
from core.github_fetcher.github_fetcher import GitHubFetcher

async def main():
    # Initialize the fetcher
    fetcher = GitHubFetcher()
    
    # Search for Python repositories
    repos = await fetcher.search_repositories("language:python stars:>1000", max_results=10)
    
    for repo in repos:
        print(f"{repo.full_name}: {repo.stars} stars")
    
    # Get a specific repository
    repo = await fetcher.get_repository("microsoft", "vscode")
    if repo:
        print(f"Description: {repo.description}")

asyncio.run(main())
```

### 使用身份验证

```python
import asyncio
from core.github_fetcher.github_fetcher import GitHubFetcher

async def main():
    # Initialize with GitHub token for higher rate limits
    fetcher = GitHubFetcher(token="your_github_token_here")
    
    # Search with authentication
    repos = await fetcher.search_repositories("machine learning", max_results=50)
    
    print(f"Found {len(repos)} repositories")

asyncio.run(main())
```

### 使用环境变量

在您的项目根目录创建一个`.env`文件：

```bash
# .env
GITHUB_API_KEY=your_github_token_here
```

然后在您的代码中使用：

```python
import os
from dotenv import load_dotenv
from core.github_fetcher.github_fetcher import GitHubFetcher

# 加载环境变量
load_dotenv()

async def main():
    token = os.getenv('GITHUB_API_KEY')
    fetcher = GitHubFetcher(token=token)
    
    # 您的代码在这里...

asyncio.run(main())
```

## API 参考

### GitHubFetcher 类

#### 构造函数

```python
GitHubFetcher(token: Optional[str] = None, base_url: str = "https://api.github.com")
```

- `token`: GitHub个人访问令牌（可选，但建议用于更高的速率限制）
- `base_url`: GitHub API基础URL（默认为公共API）

#### 方法

##### search_repositories()

在GitHub上搜索仓库。

```python
async def search_repositories(
    self,
    query: str,
    sort: str = "stars",
    order: str = "desc",
    per_page: int = 100,
    max_results: int = 1000
) -> List[GitHubRepository]
```

**参数:**
- `query`: 搜索查询字符串（例如："language:python machine learning"）
- `sort`: 排序字段（stars, forks, help-wanted-issues, updated）
- `order`: 排序顺序（asc, desc）
- `per_page`: 每页结果数量（最大100）
- `max_results`: 返回的最大总结果数

**示例:**
```python
# 搜索超过1000星的Python仓库
repos = await fetcher.search_repositories("language:python stars:>1000")

# 搜索机器学习仓库
repos = await fetcher.search_repositories("machine learning OR deep learning")

# 使用自定义排序搜索
repos = await fetcher.search_repositories("language:javascript", sort="updated", order="desc")
```

##### get_repository()

通过所有者和名称获取特定仓库。

```python
async def get_repository(self, owner: str, repo: str) -> Optional[GitHubRepository]
```

**参数:**
- `owner`: 仓库所有者用户名
- `repo`: 仓库名称

**示例:**
```python
repo = await fetcher.get_repository("microsoft", "vscode")
if repo:
    print(f"Stars: {repo.stars}")
    print(f"Language: {repo.language}")
    print(f"Description: {repo.description}")
```

##### get_user_repositories()

获取特定用户的仓库。

```python
async def get_user_repositories(
    self,
    username: str,
    type: str = "all",
    sort: str = "updated",
    direction: str = "desc",
    per_page: int = 100,
    max_results: int = 1000
) -> List[GitHubRepository]
```

**参数:**
- `username`: GitHub用户名
- `type`: 仓库类型（all, owner, public, private, member）
- `sort`: 排序字段（created, updated, pushed, full_name）
- `direction`: 排序方向（asc, desc）
- `per_page`: 每页结果数量（最大100）
- `max_results`: 返回的最大总结果数

**示例:**
```python
# 获取用户的所有公共仓库
repos = await fetcher.get_user_repositories("octocat", type="public")

# 按创建日期排序获取仓库
repos = await fetcher.get_user_repositories("octocat", sort="created", direction="desc")
```

##### get_organization_repositories()

获取特定组织的仓库。

```python
async def get_organization_repositories(
    self,
    org: str,
    type: str = "all",
    sort: str = "updated",
    direction: str = "desc",
    per_page: int = 100,
    max_results: int = 1000
) -> List[GitHubRepository]
```

**参数:**
- `org`: GitHub组织名称
- `type`: 仓库类型（all, public, private, forks, sources, member）
- `sort`: 排序字段（created, updated, pushed, full_name）
- `direction`: 排序方向（asc, desc）
- `per_page`: 每页结果数量（最大100）
- `max_results`: 返回的最大总结果数

**示例:**
```python
# 获取Microsoft组织的所有仓库
repos = await fetcher.get_organization_repositories("microsoft")

# 只获取公共仓库
repos = await fetcher.get_organization_repositories("microsoft", type="public")
```

##### get_rate_limit_info()

获取当前速率限制信息。

```python
def get_rate_limit_info(self) -> Dict[str, Any]
```

**示例:**
```python
rate_info = fetcher.get_rate_limit_info()
print(f"Remaining requests: {rate_info['rate']['remaining']}")
print(f"Rate limit resets at: {rate_info['rate']['reset']}")
```

### GitHubRepository 类

`GitHubRepository`数据类包含所有仓库信息：

```python
@dataclass
class GitHubRepository:
    id: int
    name: str
    full_name: str
    description: Optional[str]
    html_url: str
    clone_url: str
    ssh_url: str
    language: Optional[str]
    stars: int
    forks: int
    watchers: int
    open_issues: int
    created_at: datetime
    updated_at: datetime
    pushed_at: Optional[datetime]
    size: int
    topics: List[str]
    owner: Dict[str, Any]
    private: bool
    archived: bool
    disabled: bool
```

## 便利函数

### search_python_repositories()

使用常用过滤器搜索Python仓库。

```python
async def search_python_repositories(
    query: str = "",
    min_stars: int = 100,
    max_results: int = 100
) -> List[GitHubRepository]
```

**示例:**
```python
from core.github_fetcher.github_fetcher import search_python_repositories

# 搜索机器学习的Python仓库
repos = await search_python_repositories("machine learning", min_stars=500)

# 搜索至少1000星的Python仓库
repos = await search_python_repositories(min_stars=1000)
```

### search_machine_learning_repositories()

搜索机器学习仓库。

```python
async def search_machine_learning_repositories(max_results: int = 100) -> List[GitHubRepository]
```

**示例:**
```python
from core.github_fetcher.github_fetcher import search_machine_learning_repositories

repos = await search_machine_learning_repositories(max_results=50)
```

## 搜索查询示例

GitHub的搜索API支持各种限定符。以下是一些有用的示例：

### 基于语言的搜索
```python
# Python仓库
repos = await fetcher.search_repositories("language:python")

# JavaScript仓库
repos = await fetcher.search_repositories("language:javascript")

# 多种语言
repos = await fetcher.search_repositories("language:python OR language:javascript")
```

### 基于星数的搜索
```python
# 超过1000星的仓库
repos = await fetcher.search_repositories("stars:>1000")

# 星数在100到1000之间的仓库
repos = await fetcher.search_repositories("stars:100..1000")

# 恰好500星的仓库
repos = await fetcher.search_repositories("stars:500")
```

### 基于日期的搜索
```python
# 2023年1月1日之后创建的仓库
repos = await fetcher.search_repositories("created:>2023-01-01")

# 上个月更新的仓库
repos = await fetcher.search_repositories("pushed:>2023-12-01")
```

### 基于主题的搜索
```python
# 具有特定主题的仓库
repos = await fetcher.search_repositories("topic:machine-learning")

# 具有多个主题的仓库
repos = await fetcher.search_repositories("topic:machine-learning topic:python")
```

### 组合搜索
```python
# 超过1000星且在2023年后创建的Python仓库
repos = await fetcher.search_repositories("language:python stars:>1000 created:>2023-01-01")

# Python或JavaScript的机器学习仓库
repos = await fetcher.search_repositories("machine learning (language:python OR language:javascript)")
```

## 错误处理

GitHub Fetcher包含强大的错误处理：

```python
async def safe_search():
    try:
        fetcher = GitHubFetcher(token="your_token")
        repos = await fetcher.search_repositories("language:python")
        return repos
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

# 获取器还会自动处理速率限制
# 并包含网络问题的重试逻辑
```

## 速率限制

GitHub API有速率限制：
- **未认证**: 每小时60个请求
- **已认证**: 每小时5,000个请求

获取器自动处理速率限制并包含重试逻辑。

## 测试

运行测试以验证一切正常工作：

```bash
# 运行所有测试
python test/github_fetcher/run_tests.py

# 只运行单元测试（不需要API调用）
pytest test/github_fetcher/test_github_fetcher.py -v -k "not Integration"

# 只运行集成测试（需要API令牌）
pytest test/github_fetcher/test_github_fetcher.py -v -k "Integration"
```

## 示例

### 示例1：查找流行的Python库

```python
import asyncio
from core.github_fetcher.github_fetcher import GitHubFetcher

async def find_popular_python_libs():
    fetcher = GitHubFetcher()
    
    # 搜索流行的Python库
    repos = await fetcher.search_repositories(
        "language:python stars:>5000",
        sort="stars",
        order="desc",
        max_results=20
    )
    
    print("顶级Python库:")
    for i, repo in enumerate(repos, 1):
        print(f"{i:2d}. {repo.full_name:<30} {repo.stars:>6,} stars")

asyncio.run(find_popular_python_libs())
```

### 示例2：分析用户的仓库组合

```python
import asyncio
from core.github_fetcher.github_fetcher import GitHubFetcher

async def analyze_user_portfolio(username):
    fetcher = GitHubFetcher()
    
    # 获取用户的仓库
    repos = await fetcher.get_user_repositories(username, max_results=100)
    
    # 分析语言
    languages = {}
    total_stars = 0
    
    for repo in repos:
        if repo.language:
            languages[repo.language] = languages.get(repo.language, 0) + 1
        total_stars += repo.stars
    
    print(f"{username}的分析:")
    print(f"总仓库数: {len(repos)}")
    print(f"总星数: {total_stars:,}")
    print("\n语言:")
    for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {count} 个仓库")

asyncio.run(analyze_user_portfolio("octocat"))
```

### 示例3：监控组织活动

```python
import asyncio
from datetime import datetime, timedelta
from core.github_fetcher.github_fetcher import GitHubFetcher

async def monitor_org_activity(org_name):
    fetcher = GitHubFetcher()
    
    # 获取组织仓库
    repos = await fetcher.get_organization_repositories(org_name, max_results=50)
    
    # 过滤最近更新的仓库
    recent_cutoff = datetime.now() - timedelta(days=30)
    recent_repos = [repo for repo in repos if repo.updated_at > recent_cutoff]
    
    print(f"{org_name}的最近活动:")
    print(f"总仓库数: {len(repos)}")
    print(f"最近更新（过去30天）: {len(recent_repos)}")
    
    print("\n最近更新的仓库:")
    for repo in sorted(recent_repos, key=lambda x: x.updated_at, reverse=True):
        print(f"  {repo.full_name} - 更新于: {repo.updated_at.strftime('%Y-%m-%d')}")

asyncio.run(monitor_org_activity("microsoft"))
```

## 贡献

为GitHub Fetcher做贡献时：

1. 遵循现有的代码风格
2. 为新功能添加测试
3. 如果添加新功能，请更新此README
4. 所有注释和文档使用英文
5. 提交前确保所有测试通过

## 许可证

此模块是agent-engine项目的一部分，遵循相同的许可证条款。

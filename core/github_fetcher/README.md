# GitHub Fetcher

A focused GitHub repository data fetching utility that provides clean interfaces for searching and retrieving GitHub repositories using the GitHub API without coupling to storage or database concerns.

## Features

- 🔍 **Repository Search**: Search repositories with flexible query parameters
- 📊 **Repository Details**: Get detailed information about specific repositories
- 👤 **User Repositories**: Fetch repositories for specific users
- 🏢 **Organization Repositories**: Fetch repositories for organizations
- 🔄 **Rate Limiting**: Built-in rate limiting and retry logic
- 🛡️ **Error Handling**: Robust error handling with retry mechanisms
- 🚀 **Async Support**: Full async/await support for high performance
- 🔑 **Authentication**: Support for GitHub personal access tokens

## Installation

The GitHub Fetcher is part of the agent-engine package. Make sure you have the required dependencies:

```bash
pip install aiohttp requests python-dotenv
```

## Quick Start

### Basic Usage

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

### With Authentication

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

### Using Environment Variables

Create a `.env` file in your project root:

```bash
# .env
GITHUB_API_KEY=your_github_token_here
```

Then use it in your code:

```python
import os
from dotenv import load_dotenv
from core.github_fetcher.github_fetcher import GitHubFetcher

# Load environment variables
load_dotenv()

async def main():
    token = os.getenv('GITHUB_API_KEY')
    fetcher = GitHubFetcher(token=token)
    
    # Your code here...

asyncio.run(main())
```

## API Reference

### GitHubFetcher Class

#### Constructor

```python
GitHubFetcher(token: Optional[str] = None, base_url: str = "https://api.github.com")
```

- `token`: GitHub personal access token (optional, but recommended for higher rate limits)
- `base_url`: GitHub API base URL (defaults to public API)

#### Methods

##### search_repositories()

Search for repositories on GitHub.

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

**Parameters:**
- `query`: Search query string (e.g., "language:python machine learning")
- `sort`: Sort field (stars, forks, help-wanted-issues, updated)
- `order`: Sort order (asc, desc)
- `per_page`: Number of results per page (max 100)
- `max_results`: Maximum total results to return

**Example:**
```python
# Search for Python repositories with more than 1000 stars
repos = await fetcher.search_repositories("language:python stars:>1000")

# Search for machine learning repositories
repos = await fetcher.search_repositories("machine learning OR deep learning")

# Search with custom sorting
repos = await fetcher.search_repositories("language:javascript", sort="updated", order="desc")
```

##### get_repository()

Get a specific repository by owner and name.

```python
async def get_repository(self, owner: str, repo: str) -> Optional[GitHubRepository]
```

**Parameters:**
- `owner`: Repository owner username
- `repo`: Repository name

**Example:**
```python
repo = await fetcher.get_repository("microsoft", "vscode")
if repo:
    print(f"Stars: {repo.stars}")
    print(f"Language: {repo.language}")
    print(f"Description: {repo.description}")
```

##### get_user_repositories()

Get repositories for a specific user.

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

**Parameters:**
- `username`: GitHub username
- `type`: Repository type (all, owner, public, private, member)
- `sort`: Sort field (created, updated, pushed, full_name)
- `direction`: Sort direction (asc, desc)
- `per_page`: Number of results per page (max 100)
- `max_results`: Maximum total results to return

**Example:**
```python
# Get all public repositories for a user
repos = await fetcher.get_user_repositories("octocat", type="public")

# Get repositories sorted by creation date
repos = await fetcher.get_user_repositories("octocat", sort="created", direction="desc")
```

##### get_organization_repositories()

Get repositories for a specific organization.

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

**Parameters:**
- `org`: GitHub organization name
- `type`: Repository type (all, public, private, forks, sources, member)
- `sort`: Sort field (created, updated, pushed, full_name)
- `direction`: Sort direction (asc, desc)
- `per_page`: Number of results per page (max 100)
- `max_results`: Maximum total results to return

**Example:**
```python
# Get all repositories for Microsoft organization
repos = await fetcher.get_organization_repositories("microsoft")

# Get only public repositories
repos = await fetcher.get_organization_repositories("microsoft", type="public")
```

##### get_rate_limit_info()

Get current rate limit information.

```python
def get_rate_limit_info(self) -> Dict[str, Any]
```

**Example:**
```python
rate_info = fetcher.get_rate_limit_info()
print(f"Remaining requests: {rate_info['rate']['remaining']}")
print(f"Rate limit resets at: {rate_info['rate']['reset']}")
```

### GitHubRepository Class

The `GitHubRepository` dataclass contains all repository information:

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

## Convenience Functions

### search_python_repositories()

Search for Python repositories with common filters.

```python
async def search_python_repositories(
    query: str = "",
    min_stars: int = 100,
    max_results: int = 100
) -> List[GitHubRepository]
```

**Example:**
```python
from core.github_fetcher.github_fetcher import search_python_repositories

# Search for Python repositories with machine learning
repos = await search_python_repositories("machine learning", min_stars=500)

# Search for Python repositories with at least 1000 stars
repos = await search_python_repositories(min_stars=1000)
```

### search_machine_learning_repositories()

Search for machine learning repositories.

```python
async def search_machine_learning_repositories(max_results: int = 100) -> List[GitHubRepository]
```

**Example:**
```python
from core.github_fetcher.github_fetcher import search_machine_learning_repositories

repos = await search_machine_learning_repositories(max_results=50)
```

## Search Query Examples

GitHub's search API supports various qualifiers. Here are some useful examples:

### Language-based searches
```python
# Python repositories
repos = await fetcher.search_repositories("language:python")

# JavaScript repositories
repos = await fetcher.search_repositories("language:javascript")

# Multiple languages
repos = await fetcher.search_repositories("language:python OR language:javascript")
```

### Star-based searches
```python
# Repositories with more than 1000 stars
repos = await fetcher.search_repositories("stars:>1000")

# Repositories with stars between 100 and 1000
repos = await fetcher.search_repositories("stars:100..1000")

# Repositories with exactly 500 stars
repos = await fetcher.search_repositories("stars:500")
```

### Date-based searches
```python
# Repositories created after 2023-01-01
repos = await fetcher.search_repositories("created:>2023-01-01")

# Repositories updated in the last month
repos = await fetcher.search_repositories("pushed:>2023-12-01")
```

### Topic-based searches
```python
# Repositories with specific topics
repos = await fetcher.search_repositories("topic:machine-learning")

# Repositories with multiple topics
repos = await fetcher.search_repositories("topic:machine-learning topic:python")
```

### Combined searches
```python
# Python repositories with more than 1000 stars, created after 2023
repos = await fetcher.search_repositories("language:python stars:>1000 created:>2023-01-01")

# Machine learning repositories in Python or JavaScript
repos = await fetcher.search_repositories("machine learning (language:python OR language:javascript)")
```

## Error Handling

The GitHub Fetcher includes robust error handling:

```python
async def safe_search():
    try:
        fetcher = GitHubFetcher(token="your_token")
        repos = await fetcher.search_repositories("language:python")
        return repos
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

# The fetcher also handles rate limiting automatically
# and includes retry logic for network issues
```

## Rate Limiting

GitHub API has rate limits:
- **Unauthenticated**: 60 requests per hour
- **Authenticated**: 5,000 requests per hour

The fetcher automatically handles rate limiting and includes retry logic.

## Testing

Run the tests to verify everything works:

```bash
# Run all tests
python test/github_fetcher/run_tests.py

# Run only unit tests (no API calls needed)
pytest test/github_fetcher/test_github_fetcher.py -v -k "not Integration"

# Run only integration tests (requires API token)
pytest test/github_fetcher/test_github_fetcher.py -v -k "Integration"
```

## Examples

### Example 1: Find Popular Python Libraries

```python
import asyncio
from core.github_fetcher.github_fetcher import GitHubFetcher

async def find_popular_python_libs():
    fetcher = GitHubFetcher()
    
    # Search for popular Python libraries
    repos = await fetcher.search_repositories(
        "language:python stars:>5000",
        sort="stars",
        order="desc",
        max_results=20
    )
    
    print("Top Python Libraries:")
    for i, repo in enumerate(repos, 1):
        print(f"{i:2d}. {repo.full_name:<30} {repo.stars:>6,} stars")

asyncio.run(find_popular_python_libs())
```

### Example 2: Analyze User's Repository Portfolio

```python
import asyncio
from core.github_fetcher.github_fetcher import GitHubFetcher

async def analyze_user_portfolio(username):
    fetcher = GitHubFetcher()
    
    # Get user's repositories
    repos = await fetcher.get_user_repositories(username, max_results=100)
    
    # Analyze languages
    languages = {}
    total_stars = 0
    
    for repo in repos:
        if repo.language:
            languages[repo.language] = languages.get(repo.language, 0) + 1
        total_stars += repo.stars
    
    print(f"Analysis for {username}:")
    print(f"Total repositories: {len(repos)}")
    print(f"Total stars: {total_stars:,}")
    print("\nLanguages:")
    for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {count} repositories")

asyncio.run(analyze_user_portfolio("octocat"))
```

### Example 3: Monitor Organization Activity

```python
import asyncio
from datetime import datetime, timedelta
from core.github_fetcher.github_fetcher import GitHubFetcher

async def monitor_org_activity(org_name):
    fetcher = GitHubFetcher()
    
    # Get organization repositories
    repos = await fetcher.get_organization_repositories(org_name, max_results=50)
    
    # Filter recently updated repositories
    recent_cutoff = datetime.now() - timedelta(days=30)
    recent_repos = [repo for repo in repos if repo.updated_at > recent_cutoff]
    
    print(f"Recent activity for {org_name}:")
    print(f"Total repositories: {len(repos)}")
    print(f"Recently updated (last 30 days): {len(recent_repos)}")
    
    print("\nRecently updated repositories:")
    for repo in sorted(recent_repos, key=lambda x: x.updated_at, reverse=True):
        print(f"  {repo.full_name} - Updated: {repo.updated_at.strftime('%Y-%m-%d')}")

asyncio.run(monitor_org_activity("microsoft"))
```

## Contributing

When contributing to the GitHub Fetcher:

1. Follow the existing code style
2. Add tests for new functionality
3. Update this README if adding new features
4. Use English for all comments and documentation
5. Ensure all tests pass before submitting

## License

This module is part of the agent-engine project and follows the same license terms.

"""
GitHubFetcher - A focused GitHub repository data fetching utility.

This module provides clean interfaces for searching and retrieving GitHub repositories
using the GitHub API without coupling to storage or database concerns.
"""

from __future__ import annotations

import asyncio
import aiohttp
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from urllib.parse import urlencode

from agent_engine.agent_logger import AgentLogger


@dataclass
class GitHubRepository:
    """Data class representing a GitHub repository."""
    
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
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> GitHubRepository:
        """Create GitHubRepository from GitHub API response."""
        return cls(
            id=data['id'],
            name=data['name'],
            full_name=data['full_name'],
            description=data.get('description'),
            html_url=data['html_url'],
            clone_url=data['clone_url'],
            ssh_url=data['ssh_url'],
            language=data.get('language'),
            stars=data['stargazers_count'],
            forks=data['forks_count'],
            watchers=data['watchers_count'],
            open_issues=data['open_issues_count'],
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00')),
            pushed_at=datetime.fromisoformat(data['pushed_at'].replace('Z', '+00:00')) if data.get('pushed_at') else None,
            size=data['size'],
            topics=data.get('topics', []),
            owner=data['owner'],
            private=data['private'],
            archived=data['archived'],
            disabled=data['disabled']
        )


class GitHubFetcher:
    """
    A focused GitHub repository data fetching utility.
    
    This class handles searching and retrieving GitHub repositories using
    the GitHub API without coupling to storage or database concerns.
    """
    
    def __init__(self, token: Optional[str] = None, base_url: str = "https://api.github.com"):
        """
        Initialize the GitHub fetcher.
        
        Args:
            token: GitHub personal access token for authentication (optional)
            base_url: GitHub API base URL (defaults to public API)
        """
        self.logger = AgentLogger(self.__class__.__name__)
        self.base_url = base_url.rstrip('/')
        self.token = token
        
        # HTTP client configuration
        self.max_retries = 3
        self.base_retry_delay = 1  # seconds
        self.timeout = aiohttp.ClientTimeout(total=60)
        self.rate_limit_delay = 1  # seconds between requests
        
        # Prepare headers
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'GitHubFetcher/1.0'
        }
        if self.token:
            self.headers['Authorization'] = f'token {self.token}'
        
        self.logger.info(f"GitHubFetcher initialized with base URL: {self.base_url}")
    
    async def search_repositories(
        self,
        query: str,
        sort: str = "stars",
        order: str = "desc",
        per_page: int = 100,
        max_results: int = 1000
    ) -> List[GitHubRepository]:
        """
        Search for repositories on GitHub.
        
        Args:
            query: Search query string (e.g., "language:python machine learning")
            sort: Sort field (stars, forks, help-wanted-issues, updated)
            order: Sort order (asc, desc)
            per_page: Number of results per page (max 100)
            max_results: Maximum total results to return
            
        Returns:
            List of GitHubRepository objects
        """
        if not query:
            self.logger.warning("No search query provided")
            return []
        
        self.logger.info(f"Searching GitHub repositories with query: {query}")
        
        repositories = []
        page = 1
        
        async with aiohttp.ClientSession(
            headers=self.headers,
            timeout=self.timeout
        ) as session:
            while len(repositories) < max_results:
                try:
                    # Calculate how many results to fetch for this page
                    remaining = max_results - len(repositories)
                    current_per_page = min(per_page, remaining)
                    
                    # Build search URL
                    params = {
                        'q': query,
                        'sort': sort,
                        'order': order,
                        'per_page': current_per_page,
                        'page': page
                    }
                    
                    url = f"{self.base_url}/search/repositories?{urlencode(params)}"
                    
                    # Make request with retry logic
                    response_data = await self._make_request_with_retry(session, url)
                    
                    if not response_data:
                        break
                    
                    # Process results
                    items = response_data.get('items', [])
                    if not items:
                        self.logger.info("No more results available")
                        break
                    
                    for item in items:
                        try:
                            repo = GitHubRepository.from_api_response(item)
                            repositories.append(repo)
                        except Exception as e:
                            self.logger.warning(f"Failed to process repository: {e}")
                            continue
                    
                    self.logger.info(f"Fetched {len(items)} repositories from page {page}")
                    
                    # Check if we've reached the end
                    if len(items) < current_per_page:
                        break
                    
                    page += 1
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    self.logger.error(f"Error fetching page {page}: {e}")
                    break
        
        self.logger.info(f"Successfully fetched {len(repositories)} repositories")
        return repositories
    
    async def get_repository(self, owner: str, repo: str) -> Optional[GitHubRepository]:
        """
        Get a specific repository by owner and name.
        
        Args:
            owner: Repository owner username
            repo: Repository name
            
        Returns:
            GitHubRepository object or None if not found
        """
        url = f"{self.base_url}/repos/{owner}/{repo}"
        
        async with aiohttp.ClientSession(
            headers=self.headers,
            timeout=self.timeout
        ) as session:
            try:
                response_data = await self._make_request_with_retry(session, url)
                
                if response_data:
                    repo_obj = GitHubRepository.from_api_response(response_data)
                    self.logger.info(f"Successfully fetched repository: {repo_obj.full_name}")
                    return repo_obj
                else:
                    self.logger.warning(f"Repository not found: {owner}/{repo}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Error fetching repository {owner}/{repo}: {e}")
                return None
    
    async def get_user_repositories(
        self,
        username: str,
        type: str = "all",
        sort: str = "updated",
        direction: str = "desc",
        per_page: int = 100,
        max_results: int = 1000
    ) -> List[GitHubRepository]:
        """
        Get repositories for a specific user.
        
        Args:
            username: GitHub username
            type: Repository type (all, owner, public, private, member)
            sort: Sort field (created, updated, pushed, full_name)
            direction: Sort direction (asc, desc)
            per_page: Number of results per page (max 100)
            max_results: Maximum total results to return
            
        Returns:
            List of GitHubRepository objects
        """
        self.logger.info(f"Fetching repositories for user: {username}")
        
        repositories = []
        page = 1
        
        async with aiohttp.ClientSession(
            headers=self.headers,
            timeout=self.timeout
        ) as session:
            while len(repositories) < max_results:
                try:
                    # Calculate how many results to fetch for this page
                    remaining = max_results - len(repositories)
                    current_per_page = min(per_page, remaining)
                    
                    # Build URL
                    params = {
                        'type': type,
                        'sort': sort,
                        'direction': direction,
                        'per_page': current_per_page,
                        'page': page
                    }
                    
                    url = f"{self.base_url}/users/{username}/repos?{urlencode(params)}"
                    
                    # Make request with retry logic
                    response_data = await self._make_request_with_retry(session, url)
                    
                    if not response_data:
                        break
                    
                    # Process results
                    if not response_data:
                        self.logger.info("No more results available")
                        break
                    
                    for item in response_data:
                        try:
                            repo = GitHubRepository.from_api_response(item)
                            repositories.append(repo)
                        except Exception as e:
                            self.logger.warning(f"Failed to process repository: {e}")
                            continue
                    
                    self.logger.info(f"Fetched {len(response_data)} repositories from page {page}")
                    
                    # Check if we've reached the end
                    if len(response_data) < current_per_page:
                        break
                    
                    page += 1
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    self.logger.error(f"Error fetching page {page}: {e}")
                    break
        
        self.logger.info(f"Successfully fetched {len(repositories)} repositories for user {username}")
        return repositories
    
    async def get_organization_repositories(
        self,
        org: str,
        type: str = "all",
        sort: str = "updated",
        direction: str = "desc",
        per_page: int = 100,
        max_results: int = 1000
    ) -> List[GitHubRepository]:
        """
        Get repositories for a specific organization.
        
        Args:
            org: GitHub organization name
            type: Repository type (all, public, private, forks, sources, member)
            sort: Sort field (created, updated, pushed, full_name)
            direction: Sort direction (asc, desc)
            per_page: Number of results per page (max 100)
            max_results: Maximum total results to return
            
        Returns:
            List of GitHubRepository objects
        """
        self.logger.info(f"Fetching repositories for organization: {org}")
        
        repositories = []
        page = 1
        
        async with aiohttp.ClientSession(
            headers=self.headers,
            timeout=self.timeout
        ) as session:
            while len(repositories) < max_results:
                try:
                    # Calculate how many results to fetch for this page
                    remaining = max_results - len(repositories)
                    current_per_page = min(per_page, remaining)
                    
                    # Build URL
                    params = {
                        'type': type,
                        'sort': sort,
                        'direction': direction,
                        'per_page': current_per_page,
                        'page': page
                    }
                    
                    url = f"{self.base_url}/orgs/{org}/repos?{urlencode(params)}"
                    
                    # Make request with retry logic
                    response_data = await self._make_request_with_retry(session, url)
                    
                    if not response_data:
                        break
                    
                    # Process results
                    if not response_data:
                        self.logger.info("No more results available")
                        break
                    
                    for item in response_data:
                        try:
                            repo = GitHubRepository.from_api_response(item)
                            repositories.append(repo)
                        except Exception as e:
                            self.logger.warning(f"Failed to process repository: {e}")
                            continue
                    
                    self.logger.info(f"Fetched {len(response_data)} repositories from page {page}")
                    
                    # Check if we've reached the end
                    if len(response_data) < current_per_page:
                        break
                    
                    page += 1
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    self.logger.error(f"Error fetching page {page}: {e}")
                    break
        
        self.logger.info(f"Successfully fetched {len(repositories)} repositories for organization {org}")
        return repositories
    
    async def _make_request_with_retry(
        self,
        session: aiohttp.ClientSession,
        url: str
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request with retry logic."""
        for attempt in range(self.max_retries):
            try:
                async with session.get(url) as response:
                    # Handle rate limiting
                    if response.status == 403 and 'rate limit' in (await response.text()).lower():
                        self.logger.warning("Rate limit exceeded, waiting...")
                        await asyncio.sleep(60)  # Wait 1 minute
                        continue
                    
                    response.raise_for_status()
                    return await response.json()
                    
            except aiohttp.ClientError as e:
                self.logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                
                if attempt + 1 == self.max_retries:
                    self.logger.error(f"All {self.max_retries} attempts failed for {url}")
                    break
                
                # Wait before retry
                wait_time = self.base_retry_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)
        
        return None
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """
        Get current rate limit information.
        
        Returns:
            Dictionary containing rate limit information
        """
        import requests
        
        url = f"{self.base_url}/rate_limit"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error fetching rate limit info: {e}")
            return {}


# Convenience functions for common use cases
async def search_python_repositories(
    query: str = "",
    min_stars: int = 100,
    max_results: int = 100
) -> List[GitHubRepository]:
    """
    Search for Python repositories with common filters.
    
    Args:
        query: Additional search terms
        min_stars: Minimum number of stars
        max_results: Maximum number of results
        
    Returns:
        List of GitHubRepository objects
    """
    search_query = f"language:python stars:>={min_stars}"
    if query:
        search_query = f"{search_query} {query}"
    
    fetcher = GitHubFetcher()
    return await fetcher.search_repositories(search_query, max_results=max_results)


async def search_machine_learning_repositories(
    max_results: int = 100
) -> List[GitHubRepository]:
    """
    Search for machine learning repositories.
    
    Args:
        max_results: Maximum number of results
        
    Returns:
        List of GitHubRepository objects
    """
    search_query = "machine learning OR deep learning OR neural network OR AI OR artificial intelligence language:python"
    fetcher = GitHubFetcher()
    return await fetcher.search_repositories(search_query, max_results=max_results)


if __name__ == "__main__":
    async def main():
        """Example usage of GitHubFetcher."""
        fetcher = GitHubFetcher()
        
        # Search for Python repositories
        print("Searching for Python repositories...")
        repos = await fetcher.search_repositories("language:python stars:>1000", max_results=5)
        
        for repo in repos:
            print(f"- {repo.full_name}: {repo.stars} stars, {repo.language}")
        
        # Get a specific repository
        print("\nGetting specific repository...")
        repo = await fetcher.get_repository("microsoft", "vscode")
        if repo:
            print(f"- {repo.full_name}: {repo.description}")
        
        # Check rate limit
        print("\nRate limit info:")
        rate_info = fetcher.get_rate_limit_info()
        print(f"- Remaining: {rate_info.get('rate', {}).get('remaining', 'Unknown')}")
    
    asyncio.run(main())

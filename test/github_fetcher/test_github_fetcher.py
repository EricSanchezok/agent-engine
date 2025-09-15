"""
Test cases for GitHubFetcher

This module contains comprehensive tests for the GitHub fetcher functionality,
including unit tests and integration tests.
"""

import os
import sys
import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file
while project_root.parent != project_root:
    if (project_root / "pyproject.toml").exists():
        break
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from core.github_fetcher.github_fetcher import GitHubFetcher, GitHubRepository


class TestGitHubRepository:
    """Test cases for GitHubRepository dataclass."""
    
    def test_from_api_response(self):
        """Test creating GitHubRepository from API response."""
        api_data = {
            'id': 12345,
            'name': 'test-repo',
            'full_name': 'owner/test-repo',
            'description': 'A test repository',
            'html_url': 'https://github.com/owner/test-repo',
            'clone_url': 'https://github.com/owner/test-repo.git',
            'ssh_url': 'git@github.com:owner/test-repo.git',
            'language': 'Python',
            'stargazers_count': 100,
            'forks_count': 20,
            'watchers_count': 50,
            'open_issues_count': 5,
            'created_at': '2023-01-01T00:00:00Z',
            'updated_at': '2023-12-01T00:00:00Z',
            'pushed_at': '2023-12-01T00:00:00Z',
            'size': 1024,
            'topics': ['python', 'test'],
            'owner': {'login': 'owner', 'id': 1},
            'private': False,
            'archived': False,
            'disabled': False
        }
        
        repo = GitHubRepository.from_api_response(api_data)
        
        assert repo.id == 12345
        assert repo.name == 'test-repo'
        assert repo.full_name == 'owner/test-repo'
        assert repo.description == 'A test repository'
        assert repo.language == 'Python'
        assert repo.stars == 100
        assert repo.forks == 20
        assert repo.watchers == 50
        assert repo.open_issues == 5
        assert repo.topics == ['python', 'test']
        assert repo.private == False
        assert repo.archived == False
        assert repo.disabled == False
        assert isinstance(repo.created_at, datetime)
        assert isinstance(repo.updated_at, datetime)
        assert isinstance(repo.pushed_at, datetime)
    
    def test_from_api_response_with_none_values(self):
        """Test creating GitHubRepository with None values."""
        api_data = {
            'id': 12345,
            'name': 'test-repo',
            'full_name': 'owner/test-repo',
            'description': None,
            'html_url': 'https://github.com/owner/test-repo',
            'clone_url': 'https://github.com/owner/test-repo.git',
            'ssh_url': 'git@github.com:owner/test-repo.git',
            'language': None,
            'stargazers_count': 0,
            'forks_count': 0,
            'watchers_count': 0,
            'open_issues_count': 0,
            'created_at': '2023-01-01T00:00:00Z',
            'updated_at': '2023-12-01T00:00:00Z',
            'pushed_at': None,
            'size': 0,
            'topics': [],
            'owner': {'login': 'owner', 'id': 1},
            'private': False,
            'archived': False,
            'disabled': False
        }
        
        repo = GitHubRepository.from_api_response(api_data)
        
        assert repo.description is None
        assert repo.language is None
        assert repo.pushed_at is None
        assert repo.topics == []


class TestGitHubFetcher:
    """Test cases for GitHubFetcher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.token = os.getenv('GITHUB_API_KEY')
        self.fetcher = GitHubFetcher(token=self.token)
    
    def test_initialization_without_token(self):
        """Test GitHubFetcher initialization without token."""
        fetcher = GitHubFetcher()
        assert fetcher.token is None
        assert 'Authorization' not in fetcher.headers
    
    def test_initialization_with_token(self):
        """Test GitHubFetcher initialization with token."""
        test_token = "test_token_123"
        fetcher = GitHubFetcher(token=test_token)
        assert fetcher.token == test_token
        assert fetcher.headers['Authorization'] == f'token {test_token}'
    
    def test_initialization_with_custom_base_url(self):
        """Test GitHubFetcher initialization with custom base URL."""
        custom_url = "https://api.github.com"
        fetcher = GitHubFetcher(base_url=custom_url)
        assert fetcher.base_url == custom_url
    
    @pytest.mark.asyncio
    async def test_search_repositories_empty_query(self):
        """Test search_repositories with empty query."""
        result = await self.fetcher.search_repositories("")
        assert result == []
    
    @pytest.mark.asyncio
    async def test_search_repositories_with_mock(self):
        """Test search_repositories with mocked response."""
        mock_response = {
            'items': [
                {
                    'id': 12345,
                    'name': 'test-repo',
                    'full_name': 'owner/test-repo',
                    'description': 'A test repository',
                    'html_url': 'https://github.com/owner/test-repo',
                    'clone_url': 'https://github.com/owner/test-repo.git',
                    'ssh_url': 'git@github.com:owner/test-repo.git',
                    'language': 'Python',
                    'stargazers_count': 100,
                    'forks_count': 20,
                    'watchers_count': 50,
                    'open_issues_count': 5,
                    'created_at': '2023-01-01T00:00:00Z',
                    'updated_at': '2023-12-01T00:00:00Z',
                    'pushed_at': '2023-12-01T00:00:00Z',
                    'size': 1024,
                    'topics': ['python', 'test'],
                    'owner': {'login': 'owner', 'id': 1},
                    'private': False,
                    'archived': False,
                    'disabled': False
                }
            ]
        }
        
        # Mock the _make_request_with_retry method directly
        with patch.object(self.fetcher, '_make_request_with_retry', return_value=mock_response):
            result = await self.fetcher.search_repositories("language:python")
            
            assert len(result) == 1
            assert result[0].name == 'test-repo'
            assert result[0].full_name == 'owner/test-repo'
    
    @pytest.mark.asyncio
    async def test_get_repository_with_mock(self):
        """Test get_repository with mocked response."""
        mock_response = {
            'id': 12345,
            'name': 'test-repo',
            'full_name': 'owner/test-repo',
            'description': 'A test repository',
            'html_url': 'https://github.com/owner/test-repo',
            'clone_url': 'https://github.com/owner/test-repo.git',
            'ssh_url': 'git@github.com:owner/test-repo.git',
            'language': 'Python',
            'stargazers_count': 100,
            'forks_count': 20,
            'watchers_count': 50,
            'open_issues_count': 5,
            'created_at': '2023-01-01T00:00:00Z',
            'updated_at': '2023-12-01T00:00:00Z',
            'pushed_at': '2023-12-01T00:00:00Z',
            'size': 1024,
            'topics': ['python', 'test'],
            'owner': {'login': 'owner', 'id': 1},
            'private': False,
            'archived': False,
            'disabled': False
        }
        
        # Mock the _make_request_with_retry method directly
        with patch.object(self.fetcher, '_make_request_with_retry', return_value=mock_response):
            result = await self.fetcher.get_repository("owner", "test-repo")
            
            assert result is not None
            assert result.name == 'test-repo'
            assert result.full_name == 'owner/test-repo'
    
    @pytest.mark.asyncio
    async def test_get_repository_not_found(self):
        """Test get_repository when repository is not found."""
        # Mock the _make_request_with_retry method to return None (not found)
        with patch.object(self.fetcher, '_make_request_with_retry', return_value=None):
            result = await self.fetcher.get_repository("owner", "nonexistent-repo")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_get_user_repositories_with_mock(self):
        """Test get_user_repositories with mocked response."""
        mock_response = [
            {
                'id': 12345,
                'name': 'user-repo',
                'full_name': 'user/user-repo',
                'description': 'User repository',
                'html_url': 'https://github.com/user/user-repo',
                'clone_url': 'https://github.com/user/user-repo.git',
                'ssh_url': 'git@github.com:user/user-repo.git',
                'language': 'Python',
                'stargazers_count': 50,
                'forks_count': 10,
                'watchers_count': 25,
                'open_issues_count': 2,
                'created_at': '2023-01-01T00:00:00Z',
                'updated_at': '2023-12-01T00:00:00Z',
                'pushed_at': '2023-12-01T00:00:00Z',
                'size': 512,
                'topics': ['python'],
                'owner': {'login': 'user', 'id': 1},
                'private': False,
                'archived': False,
                'disabled': False
            }
        ]
        
        # Mock the _make_request_with_retry method directly
        with patch.object(self.fetcher, '_make_request_with_retry', return_value=mock_response):
            result = await self.fetcher.get_user_repositories("user")
            
            assert len(result) == 1
            assert result[0].name == 'user-repo'
            assert result[0].full_name == 'user/user-repo'
    
    @pytest.mark.asyncio
    async def test_get_organization_repositories_with_mock(self):
        """Test get_organization_repositories with mocked response."""
        mock_response = [
            {
                'id': 12345,
                'name': 'org-repo',
                'full_name': 'organization/org-repo',
                'description': 'Organization repository',
                'html_url': 'https://github.com/organization/org-repo',
                'clone_url': 'https://github.com/organization/org-repo.git',
                'ssh_url': 'git@github.com:organization/org-repo.git',
                'language': 'JavaScript',
                'stargazers_count': 200,
                'forks_count': 40,
                'watchers_count': 100,
                'open_issues_count': 10,
                'created_at': '2023-01-01T00:00:00Z',
                'updated_at': '2023-12-01T00:00:00Z',
                'pushed_at': '2023-12-01T00:00:00Z',
                'size': 2048,
                'topics': ['javascript', 'web'],
                'owner': {'login': 'organization', 'id': 1},
                'private': False,
                'archived': False,
                'disabled': False
            }
        ]
        
        # Mock the _make_request_with_retry method directly
        with patch.object(self.fetcher, '_make_request_with_retry', return_value=mock_response):
            result = await self.fetcher.get_organization_repositories("organization")
            
            assert len(result) == 1
            assert result[0].name == 'org-repo'
            assert result[0].full_name == 'organization/org-repo'
    
    def test_get_rate_limit_info(self):
        """Test get_rate_limit_info method."""
        mock_response = {
            'rate': {
                'limit': 5000,
                'remaining': 4999,
                'reset': 1234567890
            }
        }
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response
            
            result = self.fetcher.get_rate_limit_info()
            
            assert 'rate' in result
            assert result['rate']['remaining'] == 4999


class TestGitHubFetcherIntegration:
    """Integration tests for GitHubFetcher (requires real API calls)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.token = os.getenv('GITHUB_API_KEY')
        if not self.token:
            pytest.skip("GITHUB_API_KEY not found in environment variables")
        self.fetcher = GitHubFetcher(token=self.token)
    
    @pytest.mark.asyncio
    async def test_search_repositories_real_api(self):
        """Test search_repositories with real API call."""
        result = await self.fetcher.search_repositories("language:python", max_results=5)
        
        assert len(result) <= 5
        for repo in result:
            assert isinstance(repo, GitHubRepository)
            assert repo.language == 'Python'
            assert repo.stars > 0
    
    @pytest.mark.asyncio
    async def test_get_repository_real_api(self):
        """Test get_repository with real API call."""
        # Test with a well-known repository
        result = await self.fetcher.get_repository("microsoft", "vscode")
        
        assert result is not None
        assert result.name == 'vscode'
        assert result.full_name == 'microsoft/vscode'
        assert result.owner['login'] == 'microsoft'
    
    @pytest.mark.asyncio
    async def test_get_user_repositories_real_api(self):
        """Test get_user_repositories with real API call."""
        # Test with a well-known user
        result = await self.fetcher.get_user_repositories("octocat", max_results=5)
        
        assert len(result) <= 5
        for repo in result:
            assert isinstance(repo, GitHubRepository)
            assert repo.owner['login'] == 'octocat'
    
    @pytest.mark.asyncio
    async def test_get_organization_repositories_real_api(self):
        """Test get_organization_repositories with real API call."""
        # Test with a well-known organization
        result = await self.fetcher.get_organization_repositories("microsoft", max_results=5)
        
        assert len(result) <= 5
        for repo in result:
            assert isinstance(repo, GitHubRepository)
            assert repo.owner['login'] == 'microsoft'
    
    def test_get_rate_limit_info_real_api(self):
        """Test get_rate_limit_info with real API call."""
        result = self.fetcher.get_rate_limit_info()
        
        assert 'rate' in result
        assert 'remaining' in result['rate']
        assert 'limit' in result['rate']
        assert isinstance(result['rate']['remaining'], int)
        assert isinstance(result['rate']['limit'], int)


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    @pytest.mark.asyncio
    async def test_search_python_repositories(self):
        """Test search_python_repositories convenience function."""
        from core.github_fetcher.github_fetcher import search_python_repositories
        
        # Mock the GitHubFetcher to avoid real API calls
        with patch('core.github_fetcher.github_fetcher.GitHubFetcher') as mock_fetcher_class:
            mock_fetcher = AsyncMock()
            mock_fetcher_class.return_value = mock_fetcher
            
            mock_repo = GitHubRepository(
                id=1, name='test', full_name='owner/test', description='Test',
                html_url='https://github.com/owner/test', clone_url='https://github.com/owner/test.git',
                ssh_url='git@github.com:owner/test.git', language='Python', stars=100,
                forks=10, watchers=50, open_issues=5, created_at=datetime.now(),
                updated_at=datetime.now(), pushed_at=datetime.now(), size=1000,
                topics=['python'], owner={'login': 'owner'}, private=False,
                archived=False, disabled=False
            )
            
            mock_fetcher.search_repositories.return_value = [mock_repo]
            
            result = await search_python_repositories("machine learning", min_stars=50, max_results=10)
            
            assert len(result) == 1
            assert result[0].language == 'Python'
            mock_fetcher.search_repositories.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_machine_learning_repositories(self):
        """Test search_machine_learning_repositories convenience function."""
        from core.github_fetcher.github_fetcher import search_machine_learning_repositories
        
        # Mock the GitHubFetcher to avoid real API calls
        with patch('core.github_fetcher.github_fetcher.GitHubFetcher') as mock_fetcher_class:
            mock_fetcher = AsyncMock()
            mock_fetcher_class.return_value = mock_fetcher
            
            mock_repo = GitHubRepository(
                id=1, name='ml-test', full_name='owner/ml-test', description='ML Test',
                html_url='https://github.com/owner/ml-test', clone_url='https://github.com/owner/ml-test.git',
                ssh_url='git@github.com:owner/ml-test.git', language='Python', stars=200,
                forks=20, watchers=100, open_issues=10, created_at=datetime.now(),
                updated_at=datetime.now(), pushed_at=datetime.now(), size=2000,
                topics=['machine-learning'], owner={'login': 'owner'}, private=False,
                archived=False, disabled=False
            )
            
            mock_fetcher.search_repositories.return_value = [mock_repo]
            
            result = await search_machine_learning_repositories(max_results=5)
            
            assert len(result) == 1
            assert result[0].name == 'ml-test'
            mock_fetcher.search_repositories.assert_called_once()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

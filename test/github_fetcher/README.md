# GitHub Fetcher Tests

This directory contains comprehensive tests for the GitHub fetcher functionality.

## Test Structure

- `test_github_fetcher.py` - Main test file with unit and integration tests
- `test_config.py` - Configuration helper for loading environment variables
- `run_tests.py` - Test runner script
- `__init__.py` - Package initialization

## Test Categories

### Unit Tests
- Test GitHubRepository dataclass creation and validation
- Test GitHubFetcher initialization and configuration
- Test API response parsing and error handling
- Test convenience functions

### Integration Tests
- Test real API calls to GitHub (requires API token)
- Test search functionality with live data
- Test repository retrieval with actual repositories

## Setup

### 1. Install Dependencies

```bash
pip install pytest python-dotenv aiohttp requests
```

### 2. Configure GitHub API Token

Create a `.env` file in the project root:

```bash
# .env
GITHUB_API_KEY=your_github_token_here
```

To get a GitHub token:
1. Go to https://github.com/settings/tokens
2. Generate a new personal access token
3. Copy the token to your `.env` file

### 3. Run Tests

#### Option 1: Using the test runner
```bash
python test/github_fetcher/run_tests.py
```

#### Option 2: Using pytest directly
```bash
# Run all tests
pytest test/github_fetcher/test_github_fetcher.py -v

# Run only unit tests (no API calls)
pytest test/github_fetcher/test_github_fetcher.py -v -k "not Integration"

# Run only integration tests (requires API token)
pytest test/github_fetcher/test_github_fetcher.py -v -k "Integration"
```

## Test Coverage

The tests cover:

- ✅ GitHubRepository dataclass functionality
- ✅ GitHubFetcher initialization and configuration
- ✅ Search repositories functionality
- ✅ Get specific repository functionality
- ✅ Get user repositories functionality
- ✅ Get organization repositories functionality
- ✅ Rate limit information retrieval
- ✅ Error handling and edge cases
- ✅ Convenience functions
- ✅ Real API integration (when token is provided)

## Notes

- Unit tests use mocked responses and don't require API access
- Integration tests make real API calls and require a valid GitHub token
- Tests automatically skip integration tests if no token is provided
- Rate limiting is handled automatically in the tests
- All tests use English for logging and output as per project standards

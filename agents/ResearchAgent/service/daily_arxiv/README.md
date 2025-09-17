# Daily arXiv Service

This service provides a complete pipeline for filtering, downloading, and ranking arXiv papers using similarity-based filtering and Swiss tournament ranking.

## Components

### 1. Paper Filter (`paper_filter.py`)
- Filters papers from a specific date based on similarity to qiji library
- Downloads the most relevant papers
- Returns PDF paths for successful downloads

### 2. Swiss Tournament Ranker (`swiss_tournament.py`)
- Ranks papers using pairwise LLM comparisons
- Uses the `arena_referee` prompt for detailed evaluation
- Implements Swiss tournament algorithm for fair ranking

### 3. Configuration (`config.py`)
- Centralized configuration for all components
- Includes Swiss tournament parameters

## Usage

### Basic Usage

```python
import asyncio
from datetime import date
from agents.ResearchAgent.service.daily_arxiv.paper_filter import DailyArxivPaperFilter
from agents.ResearchAgent.service.daily_arxiv.swiss_tournament import SwissTournamentRanker

async def main():
    # Step 1: Filter and download papers
    filter_service = DailyArxivPaperFilter()
    filter_result = await filter_service.filter_and_download_papers(date(2025, 9, 16))
    
    if filter_result["success"]:
        # Step 2: Rank papers using Swiss tournament
        ranker = SwissTournamentRanker()
        rank_result = await ranker.rank_papers_from_pdf_paths(
            filter_result["successful_pdf_paths"]
        )
        
        print("Top Papers:")
        for paper in rank_result["top_papers"]:
            print(f"{paper['rank']}. {paper['paper_id']}: {paper['title']}")

asyncio.run(main())
```

### Using Existing PDFs

```python
# If you already have PDFs for a specific date
ranker = SwissTournamentRanker()
result = await ranker.rank_papers_from_date(date(2025, 9, 16))
```

### Running Tests

```bash
# Test Swiss tournament only
uv run agents/ResearchAgent/service/daily_arxiv/swiss_tournament.py

# Test complete integration
uv run agents/ResearchAgent/service/daily_arxiv/integration_test.py
```

## Configuration

Key configuration parameters in `config.py`:

- `TOP_K_PAPERS`: Number of papers to select from initial filtering (default: 16)
- `SWISS_TOURNAMENT_TOP_N`: Number of top papers to return from ranking (default: 8)
- `SWISS_TOURNAMENT_MODEL`: LLM model for comparisons (default: "gpt-4.1")
- `SWISS_TOURNAMENT_MAX_CONCURRENT`: Max concurrent LLM calls (default: 2)

## Environment Variables

Required environment variables:
- `AZURE_API_KEY`: Azure OpenAI API key
- `QZ_API_KEY`: Qz API key for embeddings
- `USE_ERIC_VPN`: Whether to use Eric VPN (true/false)

## File Structure

```
agents/ResearchAgent/service/daily_arxiv/
├── config.py              # Configuration
├── paper_filter.py        # Paper filtering and download
├── swiss_tournament.py   # Swiss tournament ranking
├── integration_test.py    # Integration tests
├── prompts.yaml          # LLM prompts
└── README.md             # This file
```

## Algorithm Details

### Swiss Tournament Algorithm

1. **Initial Pairing**: Papers are randomly paired for the first round
2. **Subsequent Rounds**: Papers are paired based on current scores (similar scores compete)
3. **Comparison**: Each pair is evaluated using LLM with `arena_referee` prompt
4. **Scoring**: Winners get 1 point, losers get 0 points, ties get 0.5 points each
5. **Ranking**: Final ranking based on total scores

### Evaluation Criteria

The `arena_referee` prompt evaluates papers on five dimensions:
1. **Novelty & Originality**
2. **Significance & Potential Impact**
3. **Methodological Rigor**
4. **Source Authority**
5. **Code Reproducibility**

## Error Handling

The system includes comprehensive error handling:
- PDF parsing failures are logged and skipped
- LLM comparison failures are logged and handled gracefully
- Configuration validation ensures all required parameters are set
- Detailed logging throughout the process

## Performance Considerations

- Concurrent downloads and LLM calls are limited by configuration
- PDF content is limited to 50,000 characters for LLM processing
- Swiss tournament rounds are calculated based on number of papers
- Memory usage is optimized for large paper sets

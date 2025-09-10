# ResearchAgent Test Scripts

This directory contains test scripts for verifying the ResearchAgent functionality and data migration.

## Test Scripts

### 1. `quick_data_check.py` - Quick Data Verification

**Purpose**: Quickly check if data has been successfully migrated to the remote database.

**Usage**:
```bash
./run.bat agents/ResearchAgent/test/quick_data_check.py
```

**What it does**:
- Connects to each segment database (2022H1, 2022H2, 2023H1, etc.)
- Counts records in each segment
- Shows sample record information
- Provides a summary of total records found

**Output**: Simple summary showing whether data migration was successful.

### 2. `test_remote_data_verification.py` - Comprehensive Verification

**Purpose**: Comprehensive testing of remote database connectivity, data integrity, and functionality.

**Usage**:
```bash
./run.bat agents/ResearchAgent/test/test_remote_data_verification.py
```

**What it does**:
- Tests connectivity to all segment databases
- Counts records in each segment
- Tests vector search functionality
- Analyzes record content and structure
- Generates detailed report and saves to JSON file

**Output**: Detailed verification report with statistics and analysis.

### 3. `test_paper_memory_connectivity.py` - Connectivity Test

**Purpose**: Basic connectivity test for PaperMemory system.

**Usage**:
```bash
./run.bat agents/ResearchAgent/test/test_paper_memory_connectivity.py
```

**What it does**:
- Tests basic database connectivity
- Verifies adapter configuration
- Simple connectivity validation

## Configuration

Make sure your database configuration is properly set in `agents/ResearchAgent/config.py`:

```python
PAPER_DSN_TEMPLATE = "postgresql://username:password@host:port/{db}"
```

## Expected Results

After running the migration script (`migrate_arxiv_to_ultra.py`), you should see:

- **Quick Check**: Shows records in multiple segments (2022H1, 2022H2, etc.)
- **Comprehensive Test**: All connectivity tests pass, vector search works
- **Total Records**: Should match the number of papers in your local database

### Current Status (as of testing)

✅ **Migration is working**: Data is being successfully transferred to remote database
✅ **Connection successful**: All segments can connect to remote PostgreSQL
✅ **Data found**: 2022H1 segment has ~34,699 records and growing
⏳ **In progress**: Other segments (2022H2, 2023H1, etc.) are still being processed

The migration script processes segments sequentially, so it's normal to see only one segment populated initially.

## Troubleshooting

If tests fail:

1. **Connection Issues**: Check your `PAPER_DSN_TEMPLATE` configuration
2. **No Data Found**: Migration may still be running or may have failed
3. **Vector Search Issues**: Check if pgvector extension is installed on remote database

## Running Tests

All tests use the project's standard execution method:
```bash
./run.bat <script_name>.py
```

Do not use `python` directly - always use `./run.bat` as specified in the project rules.

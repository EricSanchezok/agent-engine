# Agent Logging Management Guide

## Overview

This guide explains how to use the new logging directory management system that ensures each agent has its own isolated logging directory.

## Problem Solved

Previously, when agents used modules from `agent_engine` or `core`, all logs would be written to the project root's `logs` directory, making it difficult to track which logs belong to which agent.

The new system ensures that:
- Each agent has its own `logs` folder
- All modules (including `agent_engine` and `core` modules) write logs to the agent's designated directory
- Logs are properly separated and organized

## How It Works

### 1. Log Directory Manager

The system uses a `LogDirectoryManager` class that:
- Uses thread-local storage to maintain separate log directories for different processes
- Sets environment variables to communicate the log directory to all modules
- Provides convenience functions for easy management

### 2. Priority System

When creating an `AgentLogger`, the system follows this priority:
1. **Explicit `log_dir` parameter** (highest priority)
2. **`AGENT_LOG_DIR` environment variable** (set by the manager)
3. **Default project root `logs` directory** (fallback)

## Usage

### For Agent Developers

#### 1. In your agent's `__main__.py`

```python
from agents.YourAgent.config import LOG_DIR
from agent_engine.agent_logger import set_agent_log_directory

if __name__ == "__main__":
    # Set the log directory for this agent process
    # This ensures all modules will use this directory for logging
    set_agent_log_directory(str(LOG_DIR))
    
    agent = YourAgent()
    agent.run_server()
```

#### 2. In your agent's `config.py`

```python
from agent_engine.utils import get_current_file_dir

LOG_DIR = get_current_file_dir() / 'logs'
```

### For Module Developers

#### 1. Creating loggers in your modules

```python
from agent_engine.agent_logger import AgentLogger

# The logger will automatically use the agent's log directory if set
logger = AgentLogger(__name__)
```

#### 2. No changes needed

Existing code that uses `AgentLogger` will automatically work with the new system.

## API Reference

### Functions

- `set_agent_log_directory(log_dir: str)` - Set the log directory for the current process
- `get_agent_log_directory() -> Optional[str]` - Get the current log directory
- `clear_agent_log_directory()` - Clear the current log directory

### Classes

- `LogDirectoryManager` - Main manager class for log directory operations

## Examples

### Basic Usage

```python
from agent_engine.agent_logger import set_agent_log_directory, AgentLogger

# Set log directory
set_agent_log_directory("/path/to/agent/logs")

# Create logger - will automatically use the set directory
logger = AgentLogger("MyModule")
logger.info("This log will go to the agent's logs directory")
```

### Multiple Agents

```python
import threading
from agent_engine.agent_logger import set_agent_log_directory, AgentLogger

def agent_process(agent_name, log_dir):
    set_agent_log_directory(log_dir)
    logger = AgentLogger(f"{agent_name}.Main")
    logger.info(f"{agent_name} started")

# Each agent runs in its own thread with its own log directory
thread1 = threading.Thread(target=agent_process, args=("Agent1", "/path/to/agent1/logs"))
thread2 = threading.Thread(target=agent_process, args=("Agent2", "/path/to/agent2/logs"))

thread1.start()
thread2.start()
```

## Directory Structure

After implementation, your project structure will look like:

```
agents/
├── ArxivSearchAgent/
│   ├── logs/
│   │   ├── agent_logger_20250831_151547.log
│   │   ├── ArxivSearchAgent_Main_link.log
│   │   └── ...
│   └── ...
├── PaperFilterAgent/
│   ├── logs/
│   │   ├── agent_logger_20250831_151548.log
│   │   └── ...
│   └── ...
└── ...
```

## Testing

### Run the test script

```bash
.\run.bat test/test_log_directory_management.py
```

### Run the multi-agent demo

```bash
.\run.bat test/demo_multi_agent_logging.py
```

### Test a specific agent

```bash
.\run.bat test/test_agent_logging_simple.py
```

## Benefits

1. **Clear Separation**: Each agent's logs are completely isolated
2. **Easy Debugging**: Developers can easily find logs for specific agents
3. **No Code Changes**: Existing modules automatically work with the new system
4. **Process Safety**: Each process maintains its own log directory
5. **Backward Compatibility**: Still supports explicit log directory specification

## Migration

### Existing Code

No changes needed for existing code. The system is backward compatible.

### New Agents

Simply add the logging setup to your `__main__.py`:

```python
from agent_engine.agent_logger import set_agent_log_directory

if __name__ == "__main__":
    set_agent_log_directory(str(LOG_DIR))
    # ... rest of your code
```

## Troubleshooting

### Logs still going to project root

1. Check that `set_agent_log_directory()` is called before creating any loggers
2. Verify that the `LOG_DIR` path is correct
3. Ensure the function is called in the main process/thread

### Permission errors

1. Check that the log directory exists and is writable
2. Ensure the process has write permissions to the directory

### Multiple log files

This is normal behavior. The system creates separate log files for different logger names and maintains link files for easy access.

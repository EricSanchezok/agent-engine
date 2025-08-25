#!/usr/bin/env bash

# Enable strict error handling
set -euo pipefail

#---------------------------------------------------------------------
# Detect project root (absolute path of the directory containing run.sh)
#---------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#---------------------------------------------------------------------
# Export PYTHONPATH so that the project packages can be imported
#---------------------------------------------------------------------
export PYTHONPATH="$PROJECT_ROOT"

#---------------------------------------------------------------------
# Check input arguments
#---------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: ./run.sh <python_file_path>" >&2
    exit 1
fi

# Build absolute path to the target Python script
SCRIPT_PATH="$PROJECT_ROOT/$1"

#---------------------------------------------------------------------
# Verify that the target script exists
#---------------------------------------------------------------------
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "Error: File does not exist - $SCRIPT_PATH" >&2
    echo "Please check if the file path is correct" >&2
    exit 1
fi

#---------------------------------------------------------------------
# Display run information
#---------------------------------------------------------------------
echo "========================================";
echo "Project Root: $PROJECT_ROOT";
echo "PYTHONPATH: $PYTHONPATH";
echo "Running Script: $SCRIPT_PATH";
echo "========================================";
echo

#---------------------------------------------------------------------
# Execute the Python script using uv (ultraviolet) if available;
# fallback to the current Python interpreter otherwise.
#---------------------------------------------------------------------
if command -v uv &>/dev/null; then
    uv run "$SCRIPT_PATH"
else
    python "$SCRIPT_PATH"
fi

# Capture exit status
EXIT_CODE=$?

#---------------------------------------------------------------------
# Report execution result
#---------------------------------------------------------------------
if [[ $EXIT_CODE -ne 0 ]]; then
    echo
    echo "Script execution failed, error code: $EXIT_CODE" >&2
else
    echo
    echo "Script execution completed"
fi

exit $EXIT_CODE

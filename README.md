Environment setup (using uv)
================================

Prerequisites
-------------
- Install `uv` first. Refer to official docs.

Sync base dependencies
----------------------
- Run in project root:
```
uv sync
```

Install optional extras
-----------------------
- Unified optional group `opts` (agents + service + dev):
```
uv sync --extra opts
```

Notes
-----
- The core package `agent_engine*` only depends on base dependencies.
- `opts` contains dependencies used by `agents/`, `service/`, and dev helpers.
- Use `run.bat your_script.py` to run Python files.

Add dependencies
----------------
- Add to base dependencies:
```
uv add <package>
```
- Add to optional `opts` group:
```
uv add --optional opts <package>
```

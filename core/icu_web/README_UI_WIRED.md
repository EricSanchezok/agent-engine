This file is intentionally minimal to outline how the UI wiring works. The real README is managed elsewhere per project rules.

Endpoints:
- GET /api/patients
- POST /api/connect_patient { session_id, patient_id }
- POST /api/ingestion/update { session_id }
- WS /ws?session_id=xxx

Front-end: web/icu_agents (index.html, styles.css, main.js)

Bridge: core/frontend_bridge.py
Collector adapter: core/icu_collector_adapter.py
Session runner: core/session_runner.py



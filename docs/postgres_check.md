# PostgreSQL database switch check

> Historical run record: preserves the configuration and measurements from that stage. For current modes, demo outcomes and caveats, see [final_review.md](final_review.md). Model 4 remains a stub, not yet verified on a real pair.

Local verification only; nothing deployed.

- PostgreSQL service: `postgres:16-alpine`, defined in `docker-compose.yml`.
- Backend `/api/health`: `{"status": "ok", "worker": "running", "demo_mode": false, "demo_entries": 0}` against PostgreSQL.
- Full test command: `SATQUERY_TEST_POSTGRES_URL=<local URL> .venv/bin/python -m pytest -q`.
- Result: **115 passed, 1 skipped, 143 warnings in 36.70s**. Exit code: **0**.

All Database instances in the suite are redirected to PostgreSQL under this opt-in environment variable, including API fixtures originally specifying SQLite. Each test/URL gets a temporary schema, preserving restart tests while isolating requests; schemas are dropped afterwards. Tests assert the PostgreSQL dialect. Pure routing/model tests do not use a database. Missing Model 4 parity samples remain an explicit skip; no real-pair validation is claimed.

The independent backend process started and passed its health check. Request/result/history/report/cancellation flows are exercised through httpx by the full suite. Logs remain in ignored `.release-logs/`. The check stops its backend and Compose service; it retains the local test database volume and ignored password file for repeat runs. No passwords are printed or stored in this report.

## Repeat

Start Docker, install `backend/requirements.txt` and `backend/requirements-postgres.txt`, then run `.venv/bin/python scripts/check_postgres.py`. Optional ML dependencies/weights/samples enable real-adapter checks; otherwise those tests skip. Use `SATQUERY_POSTGRES_CHECK_PORT` if the default local check port is occupied. See README for container and manual PostgreSQL connection instructions.

"""Start only a local Compose PostgreSQL service, exercise backend and full pytest suite."""
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'docs/postgres_check.md'


def main():
    try:
        available=subprocess.run(['docker','info'],capture_output=True,text=True,timeout=20)
    except (FileNotFoundError,subprocess.TimeoutExpired) as exc:
        available=None;reason=str(exc)
    if available is None or available.returncode:
        reason=reason if available is None else available.stderr.strip()
        REPORT.write_text('# PostgreSQL switch check\n\nNot run: Docker daemon unavailable. No PostgreSQL result is claimed.\n\n'+reason+'\n\nStart Docker, then run `.venv/bin/python scripts/check_postgres.py`. See README for local Compose setup.\n')
        print('Docker unavailable: PostgreSQL check not run.');return 0
    secret=ROOT/'.secrets/postgres_check_password';secret.parent.mkdir(exist_ok=True)
    if not secret.exists():
        descriptor=os.open(secret,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w') as file:file.write(secrets.token_hex(24))
    password=secret.read_text().strip()
    env=dict(os.environ,POSTGRES_PASSWORD_FILE=str(secret),POSTGRES_PORT=os.getenv('SATQUERY_POSTGRES_CHECK_PORT','15432'))
    compose=['docker','compose','-p','satquery-postgres-check','--profile','postgres']
    backend=None;service_started=False
    logdir=ROOT/'.release-logs';logdir.mkdir(exist_ok=True)
    try:
        subprocess.run(compose+['up','-d','--wait','postgres'],cwd=ROOT,env=env,check=True,timeout=240)
        service_started=True
        url=f'postgresql+psycopg://satquery:{quote(password,safe="")}@127.0.0.1:{env["POSTGRES_PORT"]}/satquery'
        with tempfile.TemporaryDirectory(prefix='satquery-postgres-backend-') as temporary:
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            backend_env=dict(env,DATABASE_URL=url,SATQUERY_STORAGE_DIR=temporary+'/uploads',SATQUERY_TRACE_DIR=temporary+'/traces',DEMO_MODE='false',SATQUERY_STATIC_DIR='')
            with (logdir/'postgres_backend.log').open('w') as backend_log:
                backend=subprocess.Popen([sys.executable,'-m','uvicorn','backend.app.api.main:app','--host','127.0.0.1','--port',str(port)],cwd=ROOT,env=backend_env,stdout=backend_log,stderr=subprocess.STDOUT)
                for _ in range(100):
                    try:health=json.load(urlopen(f'http://127.0.0.1:{port}/api/health',timeout=1));break
                    except OSError:time.sleep(.2)
                else:raise RuntimeError('PostgreSQL-backed backend did not become healthy; see .release-logs/postgres_backend.log')
                test_env=dict(env,SATQUERY_TEST_POSTGRES_URL=url,DEMO_MODE='false',SATQUERY_STATIC_DIR='')
                tested=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,env=test_env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                (logdir/'postgres_pytest.log').write_text(tested.stdout)
                summary=next((line for line in reversed(tested.stdout.splitlines()) if re.search(r'\d+ (passed|failed|skipped)',line)),'No pytest totals found')
                REPORT.write_text('# PostgreSQL database switch check\n\nLocal verification only; nothing deployed.\n\n'
                    f'- PostgreSQL service: `postgres:16-alpine`, defined in `docker-compose.yml`.\n'
                    f'- Backend `/api/health`: `{json.dumps(health)}` against PostgreSQL.\n'
                    f'- Full test command: `SATQUERY_TEST_POSTGRES_URL=<local URL> .venv/bin/python -m pytest -q`.\n'
                    f'- Result: **{summary}**. Exit code: **{tested.returncode}**.\n\n'
                    'All Database instances in the suite are redirected to PostgreSQL under this opt-in environment variable, including API fixtures originally specifying SQLite. Each test/URL gets a temporary schema, preserving restart tests while isolating requests; schemas are dropped afterwards. Tests assert the PostgreSQL dialect. Pure routing/model tests do not use a database. Missing Model 4 parity samples remain an explicit skip; no real-pair validation is claimed.\n\n'
                    'The independent backend process started and passed its health check. Request/result/history/report/cancellation flows are exercised through httpx by the full suite. Logs remain in ignored `.release-logs/`. The check stops its backend and Compose service; it retains the local test database volume and ignored password file for repeat runs. No passwords are printed or stored in this report.\n\n'
                    '## Repeat\n\nStart Docker, install `backend/requirements.txt` and `backend/requirements-postgres.txt`, then run `.venv/bin/python scripts/check_postgres.py`. Optional ML dependencies/weights/samples enable real-adapter checks; otherwise those tests skip. Use `SATQUERY_POSTGRES_CHECK_PORT` if the default local check port is occupied. See README for container and manual PostgreSQL connection instructions.\n')
                print(summary);return tested.returncode
    except Exception as exc:
        # Avoid including connection URL/credentials in public documentation.
        REPORT.write_text('# PostgreSQL switch check\n\nVerification did not complete; no passing result is claimed.\n\nError type: '+type(exc).__name__+'\n\nInspect ignored .release-logs/ and rerun `.venv/bin/python scripts/check_postgres.py` with Docker running.\n')
        print('PostgreSQL check failed:',type(exc).__name__);return 1
    finally:
        if backend is not None:
            backend.terminate()
            try:backend.wait(timeout=30)
            except subprocess.TimeoutExpired:backend.kill();backend.wait()
        if service_started:subprocess.run(compose+['stop','postgres'],cwd=ROOT,env=env,stdout=subprocess.DEVNULL)


if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env bash
# Fresh local demo profile; never deletes existing DBs/uploads or downloads weights.
set -euo pipefail
SATQUERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SATQUERY_ROOT"
[[ -x .venv/bin/python ]] || { echo 'Run scripts/start_dev.sh first to set up the local environment.' >&2; exit 1; }
SATQUERY_DEMO_RUN="$(.venv/bin/python -c 'from pathlib import Path; from datetime import datetime,timezone; from uuid import uuid4; p=Path(".demo-rehearsal")/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-ui-"+uuid4().hex[:6]);p.mkdir(parents=True);print(p.resolve())')"
export DATABASE_URL="sqlite:///$SATQUERY_DEMO_RUN/demo.db"
export SATQUERY_STORAGE_DIR="$SATQUERY_DEMO_RUN/uploads"
export SATQUERY_TRACE_DIR="$SATQUERY_DEMO_RUN/traces"
export SATQUERY_RS_VLM_MODE=real SATQUERY_GROUNDING_MODE=real SATQUERY_CHANGE_MODE=real
export SATQUERY_FUSION_MODE="${SATQUERY_FUSION_MODE:-real}"
export DEMO_MODE="${DEMO_MODE:-false}"
echo "Fresh demo profile: $SATQUERY_DEMO_RUN"
echo "Models 1–3 request real mode; missing prerequisites are labelled stub fallback. Model 4 requested mode=$SATQUERY_FUSION_MODE; current real-pair verification is pending. DEMO_MODE=$DEMO_MODE"
exec ./scripts/start_dev.sh

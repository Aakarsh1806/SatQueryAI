import asyncio
from datetime import timedelta
from io import BytesIO
import json
from pathlib import Path
from threading import Event

import httpx
import numpy as np
from PIL import Image
import pytest
import rasterio
from sqlalchemy import select

from backend.app.api.config import Settings
from backend.app.api.main import create_app as application_factory
from backend.app.db import ModelRecord, RequestRecord, ResultRecord, TraceRecord, UploadRecord
from backend.app.db.models import utcnow


def create_app(settings):
    from backend.app.models.registry import ModelRegistry
    registry = ModelRegistry()
    for entry in registry.entries.values(): entry.mode = "stub"
    return application_factory(settings, registry)


def png(mode="RGB"):
    output = BytesIO()
    Image.new(mode, (20, 16), color=20 if mode == "L" else (10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


def configured(tmp_path, **kwargs):
    return Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}", storage_dir=tmp_path / "storage",
                    trace_dir=tmp_path / "traces", **kwargs)


async def terminal(client, request_id):
    for _ in range(500):
        response = await client.get(f"/api/requests/{request_id}")
        assert response.status_code == 200
        if response.json()["status"] in {"done", "failed"}:
            return response.json()
        await asyncio.sleep(.01)
    pytest.fail("Job did not reach terminal status")


def test_analyze_background_status_and_persistence(tmp_path):
    async def scenario():
        settings = configured(tmp_path)
        app = create_app(settings)
        gate, entered = Event(), Event()
        original = app.state.jobs.agent.run
        def blocked(*args, **kwargs):
            entered.set()
            assert gate.wait(timeout=10)
            return original(*args, **kwargs)
        app.state.jobs.agent.run = blocked
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/api/analyze", data={"query": "Where are buildings?"},
                                             files=[("files", ("../../unsafe name.png", png(), "image/png"))])
                assert response.status_code == 202
                request_id = response.json()["request_id"]
                assert response.json()["status"] == "queued"
                try:
                    assert await asyncio.to_thread(entered.wait, 3)
                    running = (await client.get(f"/api/requests/{request_id}")).json()
                    assert running["status"] == "running"
                    assert running["result"] is None
                    assert (await client.get("/api/health")).json()["status"] == "ok"
                finally:
                    gate.set()
                done = await terminal(client, request_id)
                assert done["status"] == "done", done["error"]
                assert done["result"]["metadata"]["mode"] == "stub"
                assert done["routing"]["model_id"] == "grounding"
                assert len(done["preview_urls"]) == len(done["evidence_overlay_urls"]) == 1
                assert (await client.get(done["preview_urls"][0])).headers["content-type"] == "image/png"
                assert (await client.get(done["evidence_overlay_urls"][0])).status_code == 200
                trace = (await client.get(done["trace_url"])).json()
                assert trace["request_id"] == request_id
                assert trace["chosen_model"]["mode"] == "stub"
                assert trace["api_artifacts"]["evidence_overlays"]
                history = (await client.get("/api/requests?limit=1")).json()
                assert history["total"] == 1 and history["requests"][0]["request_id"] == request_id
                with app.state.database.session() as session:
                    upload = session.scalar(select(UploadRecord))
                    assert upload.filename == "unsafe_name.png"
                    assert upload.modality == "optical" and upload.bands == 3
                    assert upload.size == len(png())
                    assert Path(upload.stored_path).is_relative_to(settings.storage_dir)
                    result = session.get(ResultRecord, request_id)
                    assert result.answer_text and result.report_path is None
                    assert session.get(TraceRecord, request_id).success
                    assert len(session.scalars(select(ModelRecord)).all()) == 4
                # Raw files and arbitrary paths must never be served.
                assert (await client.get(f"/api/assets/{request_id}/0_unsafe_name.png")).status_code == 404
                assert (await client.get("/api/assets/../raw/secret.tif")).status_code == 404
        # Restart: history survives, completed jobs are not replayed.
        app2 = create_app(settings)
        async with app2.router.lifespan_context(app2):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app2), base_url="http://test") as client:
                assert (await client.get(f"/api/requests/{request_id}")).json()["status"] == "done"
    asyncio.run(scenario())


def test_inspect_types_limits_and_cors(tmp_path):
    async def scenario():
        app = create_app(configured(tmp_path, max_upload_bytes=300, max_request_bytes=3000))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                inspected = await client.post("/api/inspect", files={"file": ("image.png", png(), "image/png")})
                assert inspected.status_code == 200
                assert inspected.json()["guessed_modality"] == "optical"
                assert inspected.json()["preview_data_url"].startswith("data:image/png;base64,")
                assert "path" not in inspected.json()
                overridden = await client.post("/api/inspect", data={"modality_override": "optical"},
                                               files={"file": ("gray.png", png("L"), "image/png")})
                assert overridden.json()["modality"] == "optical"
                assert overridden.json()["guessed_modality"] == "sar"
                assert not list((app.state.settings.storage_dir / "raw").glob("inspect-*"))
                for name, contents in [("x.exe", b"bad"), ("fake.png", b"not an image")]:
                    result = await client.post("/api/analyze", data={"query": "Describe"}, files={"files": (name, contents)})
                    assert result.status_code == 415
                large = await client.post("/api/analyze", data={"query": "Describe"}, files={"files": ("large.png", png() + b"x" * 400)})
                assert large.status_code == 413
                total = await client.post("/api/inspect", files={"file": ("large.png", png() + b"x" * 4000)})
                assert total.status_code == 413
                count = await client.post("/api/analyze", data={"query": "Describe"}, files=[("files", (f"{i}.png", png())) for i in range(3)])
                assert count.status_code == 422
                overrides = await client.post("/api/analyze", data={"query": "Describe", "modality_overrides": '["radar"]'}, files={"files": ("x.png", png())})
                assert overrides.status_code == 422
                blank = await client.post("/api/analyze", data={"query": " "}, files={"files": ("x.png", png())})
                assert blank.status_code == 422
                assert (await client.get("/api/report/anything")).status_code == 404
                assert (await client.get("/api/requests/not-an-id")).status_code == 404
                assert (await client.get("/api/traces/00000000-0000-0000-0000-000000000000")).status_code == 404
                cors = await client.options("/api/analyze", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
                assert cors.headers["access-control-allow-origin"] == "http://localhost:5173"
                models = (await client.get("/api/models")).json()["models"]
                assert len(models) == 4 and all(row["mode"] == "stub" for row in models)
                assert (await client.get("/api/requests")).json()["total"] == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("query,gray,overrides,expected", [
    ("What is visible?", False, None, "done"),
    ("Describe", True, None, "failed"),
    ("What changed?", False, None, "failed"),
    ("Describe", True, '["optical"]', "done"),
])
def test_stub_routes_and_failure_traces(tmp_path, query, gray, overrides, expected):
    async def scenario():
        app = create_app(configured(tmp_path))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                data = {"query": query}
                if overrides:
                    data["modality_overrides"] = overrides
                created = await client.post("/api/analyze", data=data, files={"files": ("input.png", png("L" if gray else "RGB"))})
                job = await terminal(client, created.json()["request_id"])
                assert job["status"] == expected
                trace = await client.get(job["trace_url"])
                assert trace.status_code == 200
                if expected == "failed":
                    assert job["error"] and trace.json()["errors"]
    asyncio.run(scenario())


def test_retention_keeps_history_traces_and_active_uploads(tmp_path):
    async def scenario():
        app = create_app(configured(tmp_path, retention_hours=1))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/api/analyze", data={"query": "Describe"}, files={"files": ("x.png", png())})
                request_id = response.json()["request_id"]
                done = await terminal(client, request_id)
                preview_url = done["preview_urls"][0]
                active_id = "00000000-0000-0000-0000-000000000001"
                active_dir = app.state.settings.storage_dir / "raw" / active_id
                active_dir.mkdir()
                (active_dir / "keep.png").write_bytes(png())
                with app.state.database.session.begin() as session:
                    session.get(RequestRecord, request_id).updated_at = utcnow() - timedelta(hours=2)
                    session.add(RequestRecord(id=active_id, query="pending", status="running", updated_at=utcnow() - timedelta(hours=2)))
                await asyncio.to_thread(app.state.jobs.cleanup)
                assert active_dir.exists()
                assert (await client.get(preview_url)).status_code == 404
                expired = (await client.get(f"/api/requests/{request_id}")).json()
                assert expired["artifacts_expired"] and not expired["preview_urls"]
                assert expired["result"]["answer"]
                assert (await client.get(expired["trace_url"])).status_code == 200
                assert not (app.state.settings.storage_dir / "raw" / request_id).exists()
    asyncio.run(scenario())


def test_two_image_change_fusion_and_corrupt_input(tmp_path):
    async def scenario():
        app = create_app(configured(tmp_path))
        sar_path = tmp_path / "sar.tif"
        with rasterio.open(sar_path, "w", driver="GTiff", width=20, height=16, count=2,
                           dtype="float32", transform=rasterio.transform.from_origin(10, 50, 1, 1)) as dst:
            dst.write(np.full((2, 16, 20), -12, dtype="float32"))
            dst.descriptions = ("VV", "VH")
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                for files, task in [
                    ([("files", ("before.png", png())), ("files", ("after.png", png()))], "change_caption"),
                    ([("files", ("sar.tif", sar_path.read_bytes())), ("files", ("optical.png", png()))], "classification"),
                ]:
                    created = await client.post("/api/analyze", data={"query": "Describe"}, files=files)
                    done = await terminal(client, created.json()["request_id"])
                    assert done["status"] == "done", done["error"]
                    assert done["task"] == task and len(done["preview_urls"]) == 2
                    if task == "classification":
                        report = await client.get(f"/api/report/{done['request_id']}")
                        assert report.status_code == 200
                        assert "68.0%" in report.text and "uncalibrated" in report.text
                        assert "Top-3 classes" in report.text and "sar" in report.text
                corrupt = await client.post("/api/analyze", data={"query": "Describe"}, files={"files": ("bad.png", b"\x89PNG\r\n\x1a\ninvalid")})
                failed = await terminal(client, corrupt.json()["request_id"])
                assert failed["status"] == "failed" and failed["trace_url"]
                assert "Cannot read" in failed["error"]
    asyncio.run(scenario())


def test_chunked_total_body_limit(tmp_path):
    async def scenario():
        app = create_app(configured(tmp_path, max_request_bytes=1000))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                async def chunks():
                    yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="x.png"\r\nContent-Type: image/png\r\n\r\n'
                    yield png() + b"x" * 1500
                    yield b"\r\n--boundary--\r\n"
                response = await client.post("/api/inspect", content=chunks(), headers={"Content-Type": "multipart/form-data; boundary=boundary"})
                assert response.status_code == 413
    asyncio.run(scenario())


def test_worker_recovers_persisted_jobs(tmp_path):
    from backend.app.db import Database
    from backend.app.models.registry import ModelRegistry
    settings = configured(tmp_path)
    database = Database(settings.database_url)
    database.initialize(ModelRegistry())
    request_id = "00000000-0000-0000-0000-000000000002"
    raw = settings.storage_dir / "raw" / request_id
    raw.mkdir(parents=True)
    path = raw / "image.png"
    path.write_bytes(png())
    with database.session.begin() as session:
        session.add(RequestRecord(id=request_id, query="Describe", status="running", options={}))
        session.flush()
        session.add(UploadRecord(request_id=request_id, filename="image.png", stored_path=str(path), size=path.stat().st_size))
    database.engine.dispose()
    partial = settings.storage_dir / "prepared" / request_id
    partial.mkdir(parents=True)
    (partial / "partial.png").write_bytes(png())
    async def scenario():
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                done = await terminal(client, request_id)
                assert done["status"] == "done", done["error"]
                assert done["trace_url"] == f"/api/traces/{request_id}"
    asyncio.run(scenario())


@pytest.mark.parametrize("query,inputs,expected", [
    ('Where are the tanks? <script>alert("x")</script>', 1, 'Grounding boxes'),
    ('What changed between these images?', 2, 'A · Before'),
    ('Describe this image', 1, 'Evidence: not available'),
])
def test_download_report_persists_self_contained_html(tmp_path, query, inputs, expected):
    async def scenario():
        app = create_app(configured(tmp_path))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                created = await client.post('/api/analyze', data={'query': query}, files=[('files', (f'input{i}.png', png())) for i in range(inputs)])
                request_id = created.json()['request_id']
                done = await terminal(client, request_id)
                assert done['status'] == 'done'
                report = await client.get(f'/api/report/{request_id}')
                assert report.status_code == 200
                assert report.headers['content-type'].startswith('text/html')
                assert 'attachment' in report.headers['content-disposition']
                assert 'stub output' in report.text
                assert 'data:image/png;base64,' in report.text
                assert 'optical' in report.text and expected in report.text
                assert 'Confidence' in report.text and 'not available' in report.text
                assert 'Compact execution trace' in report.text
                assert 'Domain gap' in report.text and 'uncalibrated' in report.text
                assert '<script>' not in report.text
                if '<script>' in query:
                    assert '&lt;script&gt;' in report.text
                with app.state.database.session() as session:
                    path = Path(session.get(ResultRecord, request_id).report_path)
                    assert path.exists() and path.read_text() == report.text
                # Cached reports retain embedded imagery even after raw inputs expire.
                with app.state.database.session.begin() as session:
                    session.get(RequestRecord, request_id).updated_at = utcnow() - timedelta(days=2)
                await asyncio.to_thread(app.state.jobs.cleanup)
                assert (await client.get(f'/api/report/{request_id}')).text == report.text
                assert (await client.get('/api/report/00000000-0000-0000-0000-000000000099')).status_code == 404
    asyncio.run(scenario())


def test_report_not_ready_and_expired_inputs(tmp_path):
    async def scenario():
        app = create_app(configured(tmp_path))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                request_id = '00000000-0000-0000-0000-000000000088'
                with app.state.database.session.begin() as session:
                    session.add(RequestRecord(id=request_id, query='Pending', status='queued'))
                assert (await client.get(f'/api/report/{request_id}')).status_code == 409
                created = await client.post('/api/analyze', data={'query': 'Describe'}, files={'files': ('x.png', png())})
                request_id = created.json()['request_id']
                await terminal(client, request_id)
                with app.state.database.session.begin() as session:
                    session.get(RequestRecord, request_id).updated_at = utcnow() - timedelta(days=2)
                await asyncio.to_thread(app.state.jobs.cleanup)
                report = await client.get(f'/api/report/{request_id}')
                assert report.status_code == 200
                assert 'retention' in report.text and 'not available' in report.text
    asyncio.run(scenario())


def test_cancel_running_and_queued_requests(tmp_path):
    async def scenario():
        app = create_app(configured(tmp_path))
        gate, entered = Event(), Event()
        original = app.state.jobs.agent.run
        def blocked(*args, **kwargs):
            entered.set()
            assert gate.wait(10)
            return original(*args, **kwargs)
        app.state.jobs.agent.run = blocked
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                ids = []
                for _ in range(2):
                    accepted = await client.post('/api/analyze', data={'query': 'Describe this image'}, files=[('files', ('image.png', png(), 'image/png'))])
                    ids.append(accepted.json()['request_id'])
                try:
                    assert await asyncio.to_thread(entered.wait, 3)
                    for request_id in ids:
                        response = await client.post(f'/api/requests/{request_id}/cancel')
                        assert response.status_code == 202
                        assert response.json()['cancellation_requested']
                finally:
                    gate.set()
                for request_id in ids:
                    for _ in range(500):
                        job = (await client.get(f'/api/requests/{request_id}')).json()
                        if job['status'] == 'cancelled': break
                        await asyncio.sleep(.01)
                    assert job['status'] == 'cancelled', job
                    assert job['result'] is None
                    trace = (await client.get(job['trace_url'])).json()
                    assert trace['cancelled']
                    assert trace['chosen_model'] is None  # cancelled before loading
                assert (await client.post(f'/api/requests/{ids[0]}/cancel')).json()['status'] == 'cancelled'
                models = (await client.get('/api/models')).json()['models']
                assert all(model['mode'] == 'stub' and model['fallback_reason'] for model in models)
                assert 'no sample' in next(model for model in models if model['id']=='fusion')['fallback_reason']
                info = (await client.post('/api/inspect', files={'file': ('image.png', png(), 'image/png')})).json()
                assert not info['fusion_eligibility']['eligible_individually']
                assert 'PNG/JPG' in info['fusion_eligibility']['message']
    asyncio.run(scenario())


def test_cancel_intent_survives_restart(tmp_path):
    from uuid import uuid4
    async def scenario():
        app = create_app(configured(tmp_path))
        app.state.database.initialize(app.state.jobs.registry)
        request_id = str(uuid4())
        with app.state.database.session.begin() as session:
            session.add(RequestRecord(id=request_id, query='Describe', status='running',
                                      options={'cancellation_requested': True}))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                for _ in range(500):
                    job = (await client.get(f'/api/requests/{request_id}')).json()
                    if job['status'] == 'cancelled': break
                    await asyncio.sleep(.01)
                assert job['status'] == 'cancelled'
                assert (await client.get(job['trace_url'])).json()['chosen_model'] is None
    asyncio.run(scenario())


def test_slow_loading_progress_cancellation_and_clear_load_failure(tmp_path):
    from backend.app.models.stubs import VLMStub
    async def scenario():
        app=create_app(configured(tmp_path))
        gate,entered=Event(),Event()
        class Loading(VLMStub):
            def load(self):
                super().load()
                self.progress('SIMULATED download/loading state; no real network transfer in this test.')
                entered.set()
                assert gate.wait(10)
        loading=Loading('Test')
        app.state.jobs.registry.get=lambda _:loading
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                accepted=await client.post('/api/analyze',data={'query':'Describe this image'},files={'files':('image.png',png())})
                identifier=accepted.json()['request_id']
                try:
                    assert await asyncio.to_thread(entered.wait,3)
                    job=(await client.get('/api/requests/'+identifier)).json()
                    assert job['status']=='running' and job['elapsed_seconds'] is not None
                    assert 'SIMULATED download' in job['progress_message']
                    assert job['result'] is None
                    assert (await client.post('/api/requests/'+identifier+'/cancel')).json()['cancellation_requested']
                finally:gate.set()
                for _ in range(300):
                    job=(await client.get('/api/requests/'+identifier)).json()
                    if job['status']=='cancelled':break
                    await asyncio.sleep(.01)
                assert job['status']=='cancelled' and job['result'] is None
                assert not loading.is_loaded
                class Broken(VLMStub):
                    def load(self):raise RuntimeError('Missing/incompatible checkpoint (test double)')
                app.state.jobs.registry.get=lambda _:Broken('Test')
                accepted=await client.post('/api/analyze',data={'query':'Describe this image'},files={'files':('image.png',png())})
                failed=await terminal(client,accepted.json()['request_id'])
                assert failed['status']=='failed' and failed['result'] is None
                assert 'Model could not be loaded' in failed['error']
                assert 'No live result was produced' in failed['error']
    asyncio.run(scenario())

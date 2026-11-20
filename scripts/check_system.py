"""Run a genuine local sample through each real adapter and the public API.

Does not alter models.yaml. A no-samples result is skipped, never a pass.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def samples_for(model_id, directory):
    from scripts.sample_inputs import optical_images, change_samples
    if model_id == 'rs_vlm':
        images = optical_images(directory)
        return ([images[0]], 'What is visible?') if images else None
    if model_id == 'grounding':
        images = optical_images(directory / 'grounding')
        return ([images[0]], 'Where are the buildings?') if images else None
    if model_id == 'change':
        pairs = change_samples(directory / 'change')
        return ([pairs[0][0], pairs[0][1]], 'What changed?') if pairs else None
    for optical in sorted((directory / 'fusion').rglob('optical.tif')):
        sar = optical.with_name('sar.tif')
        if sar.is_file():
            return [optical, sar], 'Classify the land cover'
    return None


async def check(model_id, directory):
    sample = samples_for(model_id, directory)
    if not sample:
        return {'status': 'skipped', 'reason': 'no samples found; no model loaded'}
    import httpx
    from backend.app.api.main import create_app
    from backend.app.api.config import Settings
    from backend.app.models.registry import ModelRegistry
    registry = ModelRegistry()
    registry.entries[model_id].mode = 'real'
    if registry.resolve(model_id)['mode'] != 'real':
        return {'status': 'failed', 'reason': registry.resolve(model_id)['fallback_reason']}
    paths, query = sample
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix='satquery-check-') as temporary:
        target = Path(temporary)
        app = create_app(Settings(database_url=f'sqlite:///{target / "check.db"}', storage_dir=target/'uploads', trace_dir=target/'traces'), registry)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://check') as client:
                accepted = await client.post('/api/analyze', data={'query': query}, files=[('files', (path.name, path.read_bytes())) for path in paths])
                accepted.raise_for_status()
                request_id = accepted.json()['request_id']
                while True:
                    job = (await client.get(f'/api/requests/{request_id}')).json()
                    if job['status'] not in {'queued', 'running'}: break
                    await asyncio.sleep(.1)
                passed = job['status'] == 'done' and job['result']['metadata']['mode'] == 'real'
                return {'status': 'passed' if passed else 'failed', 'elapsed_seconds': perf_counter()-started,
                        'samples': [str(path.resolve()) for path in paths], 'job': job,
                        'trace': (await client.get(job['trace_url'])).json() if job['trace_url'] else None,
                        'scope': 'local execution check only; does not measure dataset accuracy or confirm sample provenance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['all', 'rs_vlm', 'grounding', 'change', 'fusion'], default='all')
    parser.add_argument('--samples', type=Path, default=ROOT/'data_samples')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/local_model_checks.json')
    args = parser.parse_args()
    records = {}
    for model_id in (['rs_vlm', 'grounding', 'change', 'fusion'] if args.model == 'all' else [args.model]):
        try: records[model_id] = asyncio.run(check(model_id, args.samples))
        except Exception as exc: records[model_id] = {'status': 'failed', 'reason': str(exc)}
        print(model_id, json.dumps(records[model_id]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2))
    return int(any(record['status'] == 'failed' for record in records.values()))


if __name__ == '__main__':
    raise SystemExit(main())

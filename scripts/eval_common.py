"""Shared lightweight evaluation utilities; no ML imports until a real adapter is requested."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]


def arguments(description, sample_dir, report, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--samples', type=Path, default=ROOT/'data_samples'/sample_dir)
    parser.add_argument('--output', type=Path, default=ROOT/'docs'/report)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    return parser.parse_args(argv)


def real_model(model_id, device=None):
    from backend.app.models.registry import ModelRegistry
    registry = ModelRegistry()
    entry = registry.entries[model_id].model_copy(deep=True)
    entry.mode = 'real'
    if device: entry.settings['device'] = device
    # Evaluation never downloads model weights or silently evaluates a stub.
    if model_id == 'rs_vlm': entry.settings['local_files_only'] = True
    registry.entries[model_id] = entry
    resolution = registry.resolve(model_id)
    if resolution['mode'] != 'real':
        raise RuntimeError(f'Real {model_id} unavailable: {resolution["fallback_reason"]}')
    return registry.get(model_id)


def read_rows(path):
    if not path.is_file(): return []
    rows = json.loads(path.read_text())
    if not isinstance(rows, list): raise ValueError(f'Expected JSON list: {path}')
    return rows


def sample_path(directory, name):
    path = (directory/name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f'Sample path escapes directory: {name}')
    if not path.is_file(): raise FileNotFoundError(f'Sample image missing: {path}')
    return path


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().strip().split())


def lenient(text):
    words = re.findall(r'\w+', normalize(text))
    numbers = dict(zip('zero one two three four five six seven eight nine ten'.split(), map(str, range(11))))
    return [numbers.get(word, word) for word in words if word not in {'a', 'an', 'the'}]


def matches(prediction, reference):
    exact = normalize(prediction) == normalize(reference)
    p, r = lenient(prediction), lenient(reference)
    # Reference tokens must occur as a contiguous whole-token phrase. Never
    # accept substring matches such as "no" in "north", or 2 in 20.
    loose = bool(r) and any(p[i:i+len(r)] == r for i in range(len(p)-len(r)+1))
    return exact, loose


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0, min(ax2, bx2)-max(ax1, bx1)) * max(0, min(ay2, by2)-max(ay1, by1))
    union = max(0, ax2-ax1)*max(0, ay2-ay1) + max(0, bx2-bx1)*max(0, by2-by1) - intersection
    return intersection/union if union else 0.


def best_iou(predicted, truth):
    if not truth: raise ValueError('Ground-truth boxes cannot be empty')
    return max((iou(p, g) for p in predicted for g in truth), default=0.)


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def save_report(path, title, introduction, records, summary, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    environment = {}
    for package in ['torch', 'torchvision', 'transformers', 'peft', 'pycocoevalcap']:
        try: environment[package] = version(package)
        except PackageNotFoundError: pass
    document = dict(timestamp=datetime.now(timezone.utc).isoformat(), summary=summary, environment=environment, results=records)
    path.with_suffix('.json').write_text(json.dumps(document, indent=2, allow_nan=False)+'\n')
    lines = ['# '+title, '', introduction, '', 'Run timestamp: '+document['timestamp'], '',
             '## Results', '', *[f'- **{key}**: {value}' for key, value in summary.items()], '',
             'Environment: '+', '.join(f'{k} {v}' for k,v in environment.items()), '']
    if columns:
        lines += ['| '+' | '.join(label for key,label in columns)+' |',
                  '| '+' | '.join('---' for _ in columns)+' |']
    for record in records:
        lines.append('| '+' | '.join(cell(record.get(key, '')) for key,label in columns)+' |')
    path.write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2))
    print(f'Report: {path}')


def no_samples(args, title):
    print('no samples found')
    save_report(args.output, title, 'The test has not been run yet: no samples found. No model was loaded.', [],
                {'status': 'not run', 'samples': 0}, [])
    return 0

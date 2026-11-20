"""Opt-in, exact-input playback. Never generate or relabel a live prediction."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from ..schemas import StandardResult


def file_sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


class DemoCatalog:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.entries = []
        if not self.path.is_file():
            return  # No catalogue means no playback, never a made-up answer.
        data = json.loads(self.path.read_text())
        if data.get('format_version') != 1 or not isinstance(data.get('entries'), list):
            raise ValueError('Unsupported demo manifest; regenerate with scripts/prepare_demo.py')
        seen = set()
        for entry in data['entries']:
            if entry['id'] in seen:
                raise ValueError('Duplicate demo entry id')
            seen.add(entry['id'])
            if entry['model_id'] == 'fusion' and not entry.get('placeholder'):
                raise ValueError('Fusion demo entries must be explicitly marked placeholder; use live real mode for verified inference later.')
            StandardResult.model_validate(entry['result'])
            self.entries.append(entry)

    def lookup(self, images, query, decision):
        hashes = [file_sha256(image['path']) for image in images]
        modalities = [image['modality'] for image in images]
        for entry in self.entries:
            if (entry['sha256'] == hashes and entry['modalities'] == modalities
                and entry['query'] == query.strip() and entry['model_id'] == decision.model_id
                and entry['task'] == decision.task):
                result = StandardResult.model_validate(deepcopy(entry['result']))
                overlay = getattr(result.evidence, 'overlay_path', None)
                if overlay:
                    path = (self.path.parent / overlay).resolve()
                    if not path.is_relative_to(self.path.parent) or not path.is_file():
                        raise ValueError('Demo overlay missing or outside the catalogue directory')
                    result.evidence.overlay_path = str(path)
                original_runtime = result.metadata.runtime_ms
                original_device = result.metadata.device
                result.metadata.execution_type = 'pre-computed'
                result.metadata.device = 'playback (no model executed)'
                result.metadata.demo_info = dict(entry_id=entry['id'], recorded_at=entry['recorded_at'],
                    placeholder=bool(entry.get('placeholder')), original_runtime_ms=original_runtime,
                    original_device=original_device, source_model_mode=result.metadata.mode,
                    provenance=entry.get('provenance', 'to be added'))
                result.metadata.warnings.append('Pre-computed playback: no live model inference was performed for this request.')
                if entry.get('placeholder'):
                    result.metadata.warnings.append('PLACEHOLDER OUTPUT: Model 4 has not been verified on a real pair; this is not a model prediction.')
                return result
        return None

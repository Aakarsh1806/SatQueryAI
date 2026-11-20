"""Lightweight sample discovery shared by the optional real-model checks."""
from pathlib import Path

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def optical_images(directory):
    from backend.app.preprocessing import inspect
    paths = []
    for path in sorted(Path(directory).rglob("*")):
        if path.suffix.lower() not in EXTENSIONS or not path.is_file(): continue
        try:
            if inspect(path)["modality"] == "optical": paths.append(path)
        except ValueError as exc:
            print(f"Skipping {path}: {exc}")
    return paths


def change_samples(directory):
    """Manifest: before-relative-path TAB after-relative-path TAB reference.

    A single pair may instead use before.png, after.png, captions.txt with one
    or more plain reference-caption lines. Subdirectories can use the same layout.
    """
    directory = Path(directory)
    samples = []
    for captions in sorted(directory.rglob("captions.txt")):
        lines = [line.strip() for line in captions.read_text().splitlines() if line.strip() and not line.startswith("#")]
        manifest = [line.split("\t",2) for line in lines if len(line.split("\t",2)) == 3]
        if manifest:
            for before,after,reference in manifest:
                a,b = captions.parent/before,captions.parent/after
                if a.is_file() and b.is_file(): samples.append((a,b,reference))
                else: print(f"Skipping missing pair listed in {captions}: {before}, {after}")
        else:
            a,b = captions.parent/'before.png',captions.parent/'after.png'
            if a.is_file() and b.is_file() and lines: samples.append((a,b," | ".join(lines)))
    return samples


def caption_concerns(caption, reference):
    words = caption.lower().split()
    notes = []
    if not words: notes.append("empty caption")
    if len(words) >= 8 and len(set(words))/len(words) < .5: notes.append("repetitive caption")
    if words and not (set(words) & set(reference.lower().split())): notes.append("no word overlap with reference (a weak diagnostic, not a quality score)")
    return notes

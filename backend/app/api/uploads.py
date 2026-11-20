"""Bounded uploads and safe storage names. Public images live separately."""
from pathlib import Path
import re

from fastapi import HTTPException

SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def safe_filename(name):
    basename = (name or "image").replace("\\", "/").split("/")[-1]
    suffix = Path(basename).suffix.lower()
    if suffix not in SUFFIXES:
        raise HTTPException(415, "Use GeoTIFF/TIFF, PNG, or JPG image files.")
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(basename).stem)[:100] or "image"
    return stem + suffix


def valid_signature(header, suffix):
    if suffix in {".tif", ".tiff"}:
        return header[:4] in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}
    if suffix == ".png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    return header.startswith(b"\xff\xd8\xff")


async def save_upload(file, directory, index, limit):
    filename = safe_filename(file.filename)
    path = directory / f"{index}_{filename}"
    directory.mkdir(parents=True, exist_ok=True)
    size, header = 0, b""
    try:
        with path.open("wb") as output:
            while chunk := await file.read(64 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"Each image must be at most {limit} bytes.")
                if len(header) < 8:
                    header += chunk[:8 - len(header)]
                output.write(chunk)
        if not valid_signature(header, path.suffix.lower()):
            raise HTTPException(415, "The file contents do not match the supported image extension.")
        return dict(filename=filename, stored_path=str(path), size=size)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


class BodyLimitMiddleware:
    """Enforce total bytes even without Content-Length, before multipart parsing."""
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        from starlette.responses import JSONResponse
        headers = dict(scope["headers"])
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length."}, 400)(scope, receive, send)
        if declared > self.limit:
            return await JSONResponse({"detail": "Request body exceeds the upload size limit."}, 413)(scope, receive, send)
        total = 0
        async def limited_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.limit:
                raise HTTPException(413, "Request body exceeds the upload size limit.")
            return message
        await self.app(scope, limited_receive, send)

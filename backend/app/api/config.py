from dataclasses import dataclass, field
import os
from pathlib import Path

from ..models.registry import PROJECT_ROOT
from dotenv import load_dotenv

# Environment values win over an optional, ignored local .env file.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def retention_hours():
    if os.getenv("SATQUERY_RETENTION_DAYS"):
        return float(os.environ["SATQUERY_RETENTION_DAYS"]) * 24
    return float(os.getenv("SATQUERY_RETENTION_HOURS", "24"))


def optional_path(name):
    value = os.getenv(name)
    return Path(value) if value else None


@dataclass
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'satquery.db'}"))
    storage_dir: Path = field(default_factory=lambda: Path(os.getenv("SATQUERY_STORAGE_DIR", str(PROJECT_ROOT / "uploads"))))
    trace_dir: Path = field(default_factory=lambda: Path(os.getenv("SATQUERY_TRACE_DIR", str(PROJECT_ROOT / "traces"))))
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("SATQUERY_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))))
    max_request_bytes: int = field(default_factory=lambda: int(os.getenv("SATQUERY_MAX_REQUEST_BYTES", str(41 * 1024 * 1024))))
    retention_hours: float = field(default_factory=retention_hours)
    cleanup_interval_seconds: float = field(default_factory=lambda: float(os.getenv("SATQUERY_CLEANUP_INTERVAL_SECONDS", "3600")))
    cors_origins: list[str] = field(default_factory=lambda: os.getenv("SATQUERY_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","))

    demo_mode: bool = field(default_factory=lambda: os.getenv("DEMO_MODE", "false").lower() == "true")
    demo_manifest: Path = field(default_factory=lambda: Path(os.getenv("DEMO_MANIFEST", str(PROJECT_ROOT / "demo/manifest.json"))))
    static_dir: Path | None = field(default_factory=lambda: optional_path("SATQUERY_STATIC_DIR"))

    def __post_init__(self):
        self.demo_manifest = Path(self.demo_manifest).resolve()
        if self.static_dir is not None:
            self.static_dir = Path(self.static_dir).resolve()
        self.storage_dir = Path(self.storage_dir).resolve()
        self.trace_dir = Path(self.trace_dir).resolve()
        self.cors_origins = [origin.strip() for origin in self.cors_origins if origin.strip()]
        if min(self.max_upload_bytes, self.max_request_bytes, self.retention_hours, self.cleanup_interval_seconds) <= 0:
            raise ValueError("Upload limits, retention, and cleanup interval must be positive.")

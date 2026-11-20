"""Portable relational metadata; imagery and reports remain files on disk."""
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class RequestRecord(Base):
    __tablename__ = "requests"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    query: Mapped[str] = mapped_column(Text)
    task: Mapped[str | None] = mapped_column(String(40))
    model_id: Mapped[str | None] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(16), index=True)
    error: Mapped[str | None] = mapped_column(Text)
    options: Mapped[dict] = mapped_column(JSON, default=dict)
    artifacts_expired: Mapped[bool] = mapped_column(Boolean, default=False)


class UploadRecord(Base):
    __tablename__ = "uploads"
    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[str] = mapped_column(ForeignKey("requests.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(Text)
    modality: Mapped[str | None] = mapped_column(String(16))
    modality_override: Mapped[str | None] = mapped_column(String(16))
    bands: Mapped[int | None] = mapped_column(Integer)
    size: Mapped[int] = mapped_column(Integer)  # bytes on disk
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)


class TraceRecord(Base):
    __tablename__ = "traces"
    request_id: Mapped[str] = mapped_column(ForeignKey("requests.id"), primary_key=True)
    path: Mapped[str] = mapped_column(Text)
    model_id: Mapped[str | None] = mapped_column(String(40))
    version: Mapped[str | None] = mapped_column(String(100))
    mode: Mapped[str | None] = mapped_column(String(16))
    device: Mapped[str | None] = mapped_column(String(40))
    runtime_ms: Mapped[float | None] = mapped_column(Float)
    success: Mapped[bool] = mapped_column(Boolean)


class ResultRecord(Base):
    __tablename__ = "results"
    request_id: Mapped[str] = mapped_column(ForeignKey("requests.id"), primary_key=True)
    answer_text: Mapped[str | None] = mapped_column(Text)
    evidence_paths: Mapped[list] = mapped_column(JSON, default=list)
    report_path: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON)  # structured output/metadata, never image bytes


class ModelRecord(Base):
    __tablename__ = "models"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(100))
    accepted_inputs: Mapped[list] = mapped_column(JSON)
    tasks: Mapped[list] = mapped_column(JSON)
    output_type: Mapped[str] = mapped_column(String(40))
    version: Mapped[str] = mapped_column(String(100))
    weights_path: Mapped[str] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean)
    mode: Mapped[str] = mapped_column(String(16))

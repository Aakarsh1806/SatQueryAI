import os

from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ..models.registry import PROJECT_ROOT
from .models import Base, ModelRecord


class Database:
    def __init__(self, url=None):
        url = url or os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'satquery.db'}")
        parsed = make_url(url)
        kwargs = {"pool_pre_ping": True}
        if parsed.get_backend_name() == "sqlite":
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if parsed.database in {None, "", ":memory:"}:
                kwargs["poolclass"] = StaticPool
        self.engine = create_engine(url, **kwargs)
        if parsed.get_backend_name() == "sqlite":
            @event.listens_for(self.engine, "connect")
            def configure_sqlite(connection, _):
                connection.execute("PRAGMA foreign_keys=ON")
        self.session = sessionmaker(self.engine, expire_on_commit=False)

    def initialize(self, registry):
        Base.metadata.create_all(self.engine)
        with self.session.begin() as session:
            existing = {row.id: row for row in session.query(ModelRecord).all()}
            for entry in registry.entries.values():
                session.merge(ModelRecord(**entry.model_dump(exclude={"preprocessing", "settings", "validation_status"})))
                existing.pop(entry.id, None)
            for row in existing.values():
                session.delete(row)

"""Optional full-suite database switch: isolate every test in PostgreSQL schemas."""
import os
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def postgres_database_switch(monkeypatch):
    url=os.getenv('SATQUERY_TEST_POSTGRES_URL')
    if not url:
        yield
        return
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    from backend.app.db import Database
    admin=create_engine(url,pool_pre_ping=True)
    original=Database.__init__
    schemas={};engines=[]
    def switched(self, url=None):
        key=str(url or 'default')
        if key not in schemas:
            name='satquery_test_'+uuid4().hex
            with admin.begin() as connection: connection.execute(text(f'CREATE SCHEMA "{name}"'))
            schemas[key]=name
        target=make_url(os.environ['SATQUERY_TEST_POSTGRES_URL']).update_query_dict({'options':f'-csearch_path={schemas[key]}'})
        original(self,target)
        assert self.engine.dialect.name=='postgresql'
        engines.append(self.engine)
    monkeypatch.setattr(Database,'__init__',switched)
    try:yield
    finally:
        for engine in engines:engine.dispose()
        with admin.begin() as connection:
            for name in schemas.values():connection.execute(text(f'DROP SCHEMA "{name}" CASCADE'))
        admin.dispose()

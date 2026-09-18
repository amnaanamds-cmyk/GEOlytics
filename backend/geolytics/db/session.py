"""Engine and session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from geolytics.config import Settings, get_settings
from geolytics.db.models import Base


@lru_cache
def get_engine(dsn: str | None = None) -> Engine:
    settings: Settings = get_settings()
    return create_engine(dsn or settings.postgres_dsn, pool_pre_ping=True, future=True)


@lru_cache
def get_session_factory(dsn: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(dsn), expire_on_commit=False, future=True)


def create_all(dsn: str | None = None) -> None:
    """Create tables directly.

    Fine for development and for a project of this size. Introduce Alembic
    before the schema has to change against data you care about.
    """
    Base.metadata.create_all(get_engine(dsn))


@contextmanager
def session_scope(dsn: str | None = None) -> Iterator[Session]:
    session = get_session_factory(dsn)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

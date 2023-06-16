"""
Функции для работы с БД.
"""
from db import session_factory
from sqlalchemy import select
from typing import Type
from db import Base


def add_entity(entity_class: Type[Base], **kwargs):
    """Добавление объекта в БД."""
    session = session_factory()
    entity = entity_class(**kwargs)
    session.add(entity)
    session.commit()
    session.refresh(entity)
    session.close()
    return entity


def update_entity(entity_class: Type[Base], entity_id: int, **kwargs):
    """Обновляет в БД поля у объекта с указанным id."""
    session = session_factory()
    entity = session.scalars(
        select(entity_class).where(entity_class.id == entity_id)
    ).one()
    for key, value in kwargs.items():
        setattr(entity, key, value)
    session.commit()
    session.refresh(entity)
    session.close()
    return entity

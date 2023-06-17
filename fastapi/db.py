"""
Подключение к БД. Структура БД (ORM).
"""
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import (
    Mapped, sessionmaker, mapped_column, relationship
)
from sqlalchemy import (
    create_engine, String, Text, ForeignKey,
    UniqueConstraint
)
from typing import Optional, Literal, List
from typing_extensions import Annotated
import datetime

engine = create_engine('sqlite:///author_id.db')
connection = engine.connect()

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


intpk = Annotated[int, mapped_column(primary_key=True)]
timestamp = Annotated[
    datetime.datetime,
    mapped_column(server_default=func.CURRENT_TIMESTAMP()),
]
unique_name = Annotated[
    str, mapped_column(String(30), nullable=False, unique=True)]
str_50 = Annotated[str, mapped_column(String(50))]
str_100 = Annotated[str, mapped_column(String(100))]
training_status = Literal[
    "NOT STARTED", "PREPROCESSING", "FEATURE EXTRACTION",
    "TRAINING", "FINISHED"
]


def session_factory():
    Base.metadata.create_all(engine)
    return SessionLocal()


class MLModel(Base):
    """Таблица для хранения информации о моделях машинного обучения."""
    __tablename__ = 'ml_model'

    id: Mapped[intpk]
    name: Mapped[unique_name]
    description: Mapped[str] = mapped_column(Text)
    created_at: Mapped[timestamp]
    file: Mapped[Optional[str_100]]

    training: Mapped[List["Training"]] \
        = relationship(back_populates="ml_model", cascade="all, delete")


class Dataset(Base):
    """Таблица для хранения информации о датасетах."""
    __tablename__ = 'dataset'

    id: Mapped[intpk]
    name: Mapped[unique_name]
    description: Mapped[Optional[str]] = mapped_column(Text)
    file: Mapped[Optional[str_100]]
    created_at: Mapped[timestamp]
    n_texts: Mapped[int]
    n_samples: Mapped[int]
    n_authors: Mapped[int]
    version: Mapped[int] = mapped_column(default=0)

    authors: Mapped[List["AuthorDatasetMapping"]] \
        = relationship(back_populates="dataset", cascade="all, delete")
    training: Mapped[List["Training"]] \
        = relationship(back_populates="dataset")
    tests: Mapped[List["Test"]] = relationship(back_populates="dataset")


class Author(Base):
    """Таблица для хранения информации об авторах.
    `name` - короткое уникальное слово, написанное латиницей;
    `first_name`, `surname`, `patronymic` - ФИО автора на русском."""
    __tablename__ = 'author'

    id: Mapped[intpk]
    name: Mapped[unique_name]
    first_name: Mapped[Optional[str_50]]
    surname: Mapped[Optional[str_50]]
    patronymic: Mapped[Optional[str_50]]

    datasets: Mapped[List["AuthorDatasetMapping"]] \
        = relationship(back_populates="author")


class AuthorDatasetMapping(Base):
    """Таблица для связи авторов с датасетами (many-to-many)."""
    __tablename__ = 'author_dataset_mapping'

    id: Mapped[intpk]
    author_id: Mapped[int] = mapped_column(
        ForeignKey('author.id', ondelete='RESTRICT'),
        nullable=False
    )
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey('dataset.id', ondelete='CASCADE'),
        nullable=False
    )
    n_texts: Mapped[int]  # Число текстов автора в датасете
    n_samples: Mapped[int]  # Число отрывков из текстов автора в датасете

    author: Mapped["Author"] = relationship(back_populates="datasets")
    dataset: Mapped["Dataset"] = relationship(back_populates="authors")

    __table_args__ = (
        UniqueConstraint("author_id", "dataset_id", name="uix1"),
    )


class Training(Base):
    """Таблица для хранения информации об обучении ML-моделей."""
    __tablename__ = 'training'

    id: Mapped[intpk]
    ml_model_id: Mapped[int] = mapped_column(
        ForeignKey('ml_model.id', ondelete='CASCADE'), nullable=False)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey('dataset.id', ondelete='RESTRICT'), nullable=False)
    dataset_version: Mapped[int] = mapped_column(nullable=False)
    status: Mapped[training_status]
    created_at: Mapped[timestamp]
    last_update: Mapped[timestamp]

    ml_model: Mapped["MLModel"] = relationship(back_populates="training")
    dataset: Mapped["Dataset"] = relationship(back_populates="training")
    tests: Mapped[List["Test"]] = relationship(
        back_populates="training", cascade="all, delete")


class Test(Base):
    """Таблица для хранения информации о тестировании ML-моделей."""
    __tablename__ = 'test'

    id: Mapped[intpk]
    training_id: Mapped[int] = mapped_column(
        ForeignKey('training.id', ondelete='CASCADE'), nullable=False)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey('dataset.id', ondelete='RESTRICT'), nullable=False)
    dataset_version: Mapped[int]
    y_true: Mapped[str] = mapped_column(Text)
    y_pred: Mapped[str] = mapped_column(Text)
    f1_score: Mapped[float]
    created_at: Mapped[timestamp]

    training: Mapped["Training"] = relationship(back_populates="tests")
    dataset: Mapped["Dataset"] = relationship(back_populates="tests")

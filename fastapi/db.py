"""
Подключение к БД. Структура БД (ORM).
"""
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from sqlalchemy import (
    create_engine, Column, String, Integer, Numeric, Text,
    DateTime, ForeignKey
)

engine = create_engine('sqlite:///authorid.db')
connection = engine.connect()

_SessionFactory = sessionmaker(bind=engine)

Base = declarative_base()


def session_factory():
    Base.metadata.create_all(engine)
    return _SessionFactory()


class MLModel(Base):
    """Таблица для хранения информации о моделях машинного обучения."""
    __tablename__ = 'ml_model'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    description = Column(Text)
    datetime = Column(DateTime, server_default=func.now())
    file = Column(String, nullable=True)

    def __init__(self, name, description, file=None):
        self.name = name
        self.description = description
        self.file = file


class Dataset(Base):
    """Таблица для хранения информации о датасетах."""
    __tablename__ = 'dataset'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    description = Column(Text)
    file = Column(String, nullable=False)
    datetime = Column(DateTime, server_default=func.now())
    version = Column(Integer, default=0)

    def __init__(self, name, description, file):
        self.name = name
        self.description = description
        self.file = file
        self.version = 0


class Author(Base):
    """Таблица для хранения информации об авторах."""
    __tablename__ = 'author'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    first_name = Column(String, nullable=True)
    surname = Column(String, nullable=True)
    patronymic = Column(String, nullable=True)

    def __init__(self, name, first_name=None, surname=None, patronymic=None):
        """
        Parameters
        ----------
        name : str
            Короткое уникальное слово, написанное латиницей.

        first_name : str
            Имя автора на русском.

        surname : str
            Фамилия автора на русском.

        patronymic : str
            Отчество автора на русском.
        """
        self.name = name
        self.first_name = first_name
        self.surname = surname
        self.patronymic = patronymic


class AuthorDatasetMapping(Base):
    """Таблица для связи авторов с датасетами (many-to-many)."""
    author_id = Column(
        Integer,
        ForeignKey('author.id', ondelete='RESTRICT'),
        nullable=False,
        primary_key=True
    )
    dataset_id = Column(
        Integer,
        ForeignKey('dataset.id', ondelete='CASCADE'),
        nullable=False,
        primary_key=True
    )
    # Число отрывков из текстов автора в датасете
    n_samples = Column(Integer, nullable=False)


class Training(Base):
    """Таблица для хранения информации об обучении ML-моделей."""
    __tablename__ = 'training'

    id = Column(Integer, primary_key=True)
    ml_model_id = Column(
        Integer,
        ForeignKey('ml_model.id', ondelete='CASCADE'),
        nullable=False
    )
    dataset_id = Column(
        Integer,
        ForeignKey('dataset.id', ondelete='RESTRICT'),
        nullable=False
    )
    dataset_version = Column(Integer, nullable=False)
    status = Column(String, default='NOT STARTED')
    start_datetime = Column(DateTime, nullable=True)
    finish_datetime = Column(DateTime, nullable=True)

    def __init__(self, ml_model: MLModel, dataset: Dataset):
        self.ml_model_id = ml_model.id
        self.dataset_id = dataset.id
        self.ml_model_version = ml_model.version
        self.dataset_version = dataset.version


class Test(Base):
    """Таблица для хранения информации о тестировании ML-моделей."""
    __tablename__ = 'test'

    id = Column(Integer, primary_key=True)
    training_id = Column(
        Integer,
        ForeignKey('training.id', ondelete='CASCADE'),
        nullable=False
    )
    dataset_id = Column(
        Integer,
        ForeignKey('dataset.id', ondelete='SET NULL'),
        nullable=True
    )
    dataset_version = Column(Integer, nullable=False)
    y_true = Column(String, nullable=False)
    y_pred = Column(String, nullable=False)
    f1_score = Column(Numeric, nullable=False)
    datetime = Column(DateTime, server_default=func.now())



"""
Схемы FastAPI.
"""
from pydantic import BaseModel
from datetime import datetime
from typing import List, Union

tags_metadata = [
    {"name": "Dataset", "description": "Работа с датасетами."},
    {"name": "Author", "description":
        "Информация об авторах, тексты которых есть в датасетах."},
    {"name": "Train", "description": "Обучение ML-моделей."},
    {"name": "Test", "description": "Оценка качества ML-моделей."},
    {"name": "Predict", "description": "Применение ML-моделей."},
    {"name": "Tools", "description":
        "Инструменты для формирования датасетов."},
    {"name": "Status", "description": "Проверка доступности сервиса."},
]


class PredictionResults(BaseModel):
    n_texts: int
    predictions: List[Union[int, str]]
    exec_time: float


class TrainResults(BaseModel):
    id: int
    exec_time: float


class Author(BaseModel):
    id: int
    name: str
    first_name: str | None
    surname: str | None
    patronymic: str | None

    class Config:
        orm_mode = True


class DatasetAuthor(BaseModel):
    author: Author
    n_texts: int
    n_samples: int

    class Config:
        orm_mode = True


class Dataset(BaseModel):
    id: int
    name: str
    description: str
    created_at: datetime
    n_authors: int
    n_texts: int
    n_samples: int
    version: int
    authors: List[DatasetAuthor]

    class Config:
        orm_mode = True
    # training: Mapped[List["Training"]] \
    #     = relationship(back_populates="dataset")
    # tests: Mapped[List["Test"]] = relationship(back_populates="dataset")


class DatasetBriefly(BaseModel):
    id: int
    name: str
    description: str | None
    created_at: datetime
    n_authors: int
    n_texts: int
    n_samples: int
    # training:
    # tests:

    class Config:
        orm_mode = True


class AuthorDataset(BaseModel):
    dataset: DatasetBriefly
    n_texts: int
    n_samples: int

    class Config:
        orm_mode = True


class AuthorDetailed(Author):
    datasets: List[AuthorDataset]

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
    {"name": "Model", "description":
        "Информация об ML-моделях. Удаление ML-моделей."},
    {"name": "Tools", "description":
        "Инструменты для формирования датасетов."},
    {"name": "Status", "description": "Проверка доступности сервиса."},
]


class PredictionResults(BaseModel):
    n_texts: int
    predictions: List[Union[int, str]]
    elapsed_time: float


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


class DatasetId(BaseModel):
    id: int

    class Config:
        orm_mode = True


class Dataset(BaseModel):
    id: int
    name: str
    description: str | None
    created_at: datetime
    n_authors: int
    n_texts: int
    n_samples: int
    version: int
    authors: List[DatasetAuthor]

    class Config:
        orm_mode = True


class AuthorDataset(BaseModel):
    dataset: Dataset
    n_texts: int
    n_samples: int

    class Config:
        orm_mode = True


class AuthorDetailed(Author):
    datasets: List[AuthorDataset]


class TestForDataset(BaseModel):
    id: int
    dataset_version: int
    f1_score: float | None
    created_at: datetime
    last_update: datetime
    elapsed_time: float | None
    status: str

    class Config:
        orm_mode = True


class TestShort(TestForDataset):
    dataset: DatasetId


class Test(TestForDataset):
    dataset: Dataset


class TrainingForMLModel(BaseModel):
    id: int
    status: str
    created_at: datetime
    last_update: datetime
    elapsed_time: float | None
    dataset: Dataset
    dataset_version: int
    tests: List[Test]

    class Config:
        orm_mode = True


class MLModelId(BaseModel):
    id: int

    class Config:
        orm_mode = True


class MLModelForTraining(BaseModel):
    id: int
    name: str
    description: str | None
    created_at: datetime

    class Config:
        orm_mode = True


class MLModel(MLModelForTraining):
    training: List[TrainingForMLModel]


class Training(TrainingForMLModel):
    ml_model: MLModelForTraining


class TrainingShort(BaseModel):
    id: int
    status: str
    created_at: datetime
    last_update: datetime
    elapsed_time: float | None
    dataset: DatasetId
    ml_model: MLModelId
    dataset_version: int

    class Config:
        orm_mode = True


class TrainingForDataset(BaseModel):
    id: int
    status: str
    created_at: datetime
    last_update: datetime
    elapsed_time: float | None
    ml_model: MLModelForTraining
    dataset_version: int

    class Config:
        orm_mode = True


class DatasetDetailed(Dataset):
    training: List[TrainingForDataset]
    tests: List[TestForDataset]

    class Config:
        orm_mode = True


class TestDetailed(Test):
    training: TrainingForDataset


class UploadedFilename:
    filename: str

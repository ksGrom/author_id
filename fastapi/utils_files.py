"""
Функции для работы с файлами.
"""
import joblib
from pathlib import Path

import pandas as pd


def __mkdir(path):
    """Создает директорию, если не существует.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_model_file(clf, filename, dir_path="./ml_models"):
    """Сохраняет модель в указанный файл (pickle).
    """
    __mkdir(dir_path)
    path = Path(dir_path, filename)
    joblib.dump(clf, path)
    return path


def save_dataset_file(df: pd.DataFrame, filename: str, dir_path="./datasets"):
    """Сохраняет csv-файл датасета."""
    __mkdir(dir_path)
    path = Path(dir_path, filename)
    df.to_csv(path)
    return path


def delete_file(file_path: str):
    """Удаляет файл."""
    Path(file_path).unlink(missing_ok=True)

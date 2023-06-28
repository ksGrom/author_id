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
    """Сохраняет модель в указанный файл (pickle)."""
    __mkdir(dir_path)
    path = Path(dir_path, filename)
    joblib.dump(clf, path)
    return path


def load_model(file_path: str):
    """Загрузка модели из pickle-файла."""
    return joblib.load(Path(file_path))


def save_dataset_file(
        df: pd.DataFrame,
        filename: str,
        dir_path="./datasets",
        index=True
):
    """Сохраняет csv-файл датасета."""
    __mkdir(dir_path)
    path = Path(dir_path, filename)
    df.to_csv(path, index=index)
    return path


def delete_file(file_path: str):
    """Удаляет файл."""
    Path(file_path).unlink(missing_ok=True)


def save_output_df(
        df: pd.DataFrame,
        filename: str,
        dir_path="./user_csv",
        index=True
):
    return save_dataset_file(df, filename, dir_path, index=index)


def get_user_csv_path(filename: str, dir_path="./user_csv"):
    path = Path(dir_path, filename)
    if not path.is_file():
        return None, None
    orig_filename = "_".join(filename.split("_")[:-1]) + ".csv"
    return path, orig_filename

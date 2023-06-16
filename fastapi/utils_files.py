"""
Функции для работы с файлами.
"""
import joblib
from pathlib import Path


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

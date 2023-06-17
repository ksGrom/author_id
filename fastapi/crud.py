"""
Функции для работы с БД.
"""
import pandas as pd
import re

from db import (
    session_factory, MLModel, Dataset, Author,
    AuthorDatasetMapping,
)
from sqlalchemy import select, and_
from fastapi import UploadFile
from sqlalchemy.orm import Session
from typing import Type, Any
import utils_dataset as utd
import utils_files as uf
from db import Base


def add_entity(db: Session, entity_class: Type[Base], **kwargs):
    """Добавление объекта в БД."""
    entity = entity_class(**kwargs)
    db.add(entity)
    db.commit()
    db.refresh(entity)
    return entity


def update_entity(
        db: Session,
        entity_class: Type[Base],
        entity_id: int,
        **kwargs
):
    """Обновляет в БД поля у объекта с указанным id."""
    entity = db.scalars(
        select(entity_class).where(entity_class.id == entity_id)
    ).one()
    for key, value in kwargs.items():
        setattr(entity, key, value)
    db.commit()
    db.refresh(entity)
    return entity


def get_entity(
        db: Session,
        entity_class: Type[Base],
        column_name: str,
        value: Any
):
    """Получает объект по значению `value` в столбце `column_name`
    (в столбце должны быть уникальные значения)."""
    attr = getattr(entity_class, column_name)
    query = db.execute(
        select(entity_class).where(attr == value)
    ).first()
    if query is None or len(query) == 0:
        return None
    return query[0]


def delete_entity(
        db: Session,
        entity_class: Type[Base],
        column_name: str,
        value: Any
):
    entity = get_entity(db, entity_class, column_name, value)
    db.delete(entity)
    db.commit()
    return entity


def check_uniqueness(
        db: Session,
        entity_class: Type[Base],
        column_name: str,
        value: Any
):
    """Проверяет уникальность значения `value` в столбце `column_name`."""
    if get_entity(db, entity_class, column_name, value) is None:
        return True
    return False


def get_all(
        db: Session,
        entity_class: Type[Base]
):
    """Получает все объекты из таблицы."""
    return [el[0] for el in db.execute(select(entity_class)).all()]


def check_slug_field(text: str, max_len=50):
    """Проверяет текст на соответствие рег. выражению `[a-zA-Z0-9_-]+$`."""
    match = bool(re.match("[a-zA-Z0-9_-]+$", text))
    if len(text) <= max_len and match:
        return True
    return False


def __process_uploaded_dataset_file(upload_file: UploadFile):
    """Вспомогательная функция для преобразования загруженного csv-файла
    в DataFrame с обработанными и нарезанными на отрывки текстами."""
    df = utd.input_file_to_df(upload_file.file, filename=upload_file.filename)
    if 'text' not in df or 'author' not in df:
        raise ValueError("No 'text' or/and 'author' column(s) in input file!")
    df = df[['text', 'author']]
    df.text = df.text.apply(utd.text_preprocessing)
    df = utd.make_dataset_of_excerpts(df)
    return df


def save_dataset_from_uploaded_csv(
        db: Session,
        dataset_name: str,
        dataset_description: str,
        upload_file: UploadFile
):
    """Сохраняет датасет из загруженного csv-файла
    (сохраняет файл, добавляет информацию в БД)."""
    if not check_uniqueness(db, Dataset, 'name', dataset_name):
        raise ValueError("Dataset name must be unique!")
    if not check_slug_field(dataset_name, max_len=25):
        raise ValueError("Invalid dataset name! (Max length: 25. "
                         "Must match regex `[a-zA-Z0-9_-]+$`)")
    df = __process_uploaded_dataset_file(upload_file)
    file_path = uf.save_dataset_file(df, filename=f"{dataset_name}.csv")
    dataset = add_entity(
        db, Dataset, name=dataset_name, n_authors=df.author.unique().shape[0],
        n_texts=df.orig_text_id.unique().shape[0], n_samples=df.shape[0],
        description=dataset_description, file=str(file_path)
    )
    add_author_dataset_entities(db, dataset.id, df)
    return dataset


def add_texts_to_dataset(
        db: Session,
        dataset: Dataset,
        upload_file: UploadFile
):
    """Добавляет тексты в существующий датасет."""
    df_new_texts = __process_uploaded_dataset_file(upload_file)
    df_old = pd.read_csv(dataset.file, index_col=0)
    df_new_texts.orig_text_id = \
        df_new_texts.orig_text_id + df_old.orig_text_id.max() + 1
    df = pd.concat([df_old, df_new_texts], axis=0, ignore_index=True)
    df.to_csv(dataset.file)
    dataset = update_entity(
        db, Dataset, dataset.id, n_authors=df.author.unique().shape[0],
        n_texts=df.orig_text_id.unique().shape[0], n_samples=df.shape[0]
    )
    add_author_dataset_entities(db, dataset.id, df)
    return dataset


def get_dataset_by_id(db: Session, dataset_id: int):
    """Получает датасет по id."""
    return get_entity(db, Dataset, 'id', dataset_id)


def get_dataset_by_name(db: Session, dataset_name: str):
    """Получает датасет по имени."""
    return get_entity(db, Dataset, 'name', dataset_name)


def delete_dataset(db: Session, dataset: Dataset):
    """Удаляет датасет с указанным id."""
    uf.delete_file(dataset.file)
    return delete_entity(db, Dataset, 'id', dataset.id)


def get_all_datasets(db: Session):
    """Получает все датасеты."""
    return get_all(db, Dataset)


def update_dataset_description(
        db: Session,
        dataset_id: int,
        description: str
):
    return update_entity(
        db, Dataset, dataset_id, description=description
    )


def get_author_by_id(db: Session, author_id: int):
    """Получает автора по id."""
    return get_entity(db, Author, 'id', author_id)


def get_author_by_name(db: Session, name: str):
    """Получает автора по «имени»."""
    return get_entity(db, Author, 'name', name)


def update_author(db: Session, author_id, first_name, surname, patronymic):
    """Обновляет информацию об авторе с указанным id."""
    return update_entity(
        db, Author, author_id, first_name=first_name,
        surname=surname, patronymic=patronymic
    )


def get_all_authors(db: Session):
    """Получает всех авторов."""
    return list(get_all(db, Author))


def add_author_dataset_entities(
        db: Session,
        dataset_id: int,
        df: pd.DataFrame
):
    """Добавляет связи 'датасет-автор' в БД.
    Добавляет неизвестных ранее авторов в БД."""
    for author in df.author.unique():
        if check_uniqueness(db, Author, 'name', author):
            author_ent = add_entity(db, Author, name=author)
        else:
            author_ent = get_entity(db, Author, 'name', author)
        df_author = df[df.author == author]
        n_samples = df_author.shape[0]
        n_texts = df_author.orig_text_id.unique().shape[0]
        query = db.execute(
            select(AuthorDatasetMapping).where(
                and_(
                    AuthorDatasetMapping.author_id == author_ent.id,
                    AuthorDatasetMapping.dataset_id == dataset_id
                )
            )
        ).first()
        if query is None or len(query) == 0:
            add_entity(
                db, AuthorDatasetMapping, author_id=author_ent.id,
                dataset_id=dataset_id, n_texts=n_texts, n_samples=n_samples
            )
        else:
            update_entity(
                db, AuthorDatasetMapping, query[0].id,
                n_texts=n_texts, n_samples=n_samples
            )

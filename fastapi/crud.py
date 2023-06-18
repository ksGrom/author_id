"""
Функции для работы с БД.
"""
import pandas as pd
import re

from db import (
    MLModel, Dataset, Author,
    AuthorDatasetMapping, Training, Test
)
from sqlalchemy import select, and_
from fastapi import UploadFile
from sqlalchemy.orm import Session
from typing import Type, Any
import utils_dataset as utd
import utils_files as uf
from db import Base
from time import time
import utils_tfidf_clf as u
from datetime import datetime
from sklearn.metrics import f1_score


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
    df.text = df.text.apply(lambda x: x.strip())
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
        n_texts=df.orig_text_id.unique().shape[0], n_samples=df.shape[0],
        version=dataset.version+1
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


def create_training(db: Session, model: MLModel, dataset: Dataset):
    return add_entity(
        db, Training, ml_model_id=model.id, dataset_id=dataset.id,
        dataset_version=dataset.version, status="NOT STARTED",
        last_update=datetime.now()
    )


def update_training(db: Session, training: Training,
                    status: str, **kwargs):
    return update_entity(
        db, Training, training.id,
        status=status, last_update=datetime.now(), **kwargs
    )


def __training_db_update_status_func(db: Session, training: Training):
    """Возвращает функцию, которая обновляет статус обучения
    в БД (эта функция передается методу `fit` классификатора)."""
    def func(status: str):
        update_training(db, training, status)
    return func


def __test_db_update_status_func(db: Session, test: Test):
    """Возвращает функцию, которая обновляет статус тестирования
    в БД (эта функция передается методу `predict` классификатора)."""
    def func(status: str):
        update_test(db, test, status)
    return func


def update_ml_model(db: Session, ml_model: MLModel, **kwargs):
    return update_entity(db, MLModel, ml_model.id, **kwargs)


def train_model(
        db: Session,
        ml_model: MLModel,
        dataset: Dataset,
        clf: u.AuthorIdTfidfPipeline | None = None
):
    """Обучение ML-модели."""
    start_t = time()
    training = create_training(db, ml_model, dataset)
    df = pd.read_csv(dataset.file)
    status_update_func = __training_db_update_status_func(db, training)
    if clf is None:
        clf = u.AuthorIdTfidfPipeline()
    clf.fit(df, preprocessing=False, logger=status_update_func)
    file_path = uf.save_model_file(clf, f"{ml_model.name}.pkl")
    update_training(db, training, status="FINISHED",
                    elapsed_time=time()-start_t)
    update_ml_model(db, ml_model, file=str(file_path))


def get_last_training(db: Session, ml_model_id: int):
    query = db.execute(select(Training).where(
        Training.ml_model_id == ml_model_id
    ).order_by(Training.dataset_version.desc())).first()
    if query is None or len(query) == 0:
        return None
    return query[0]


def continue_training(db: Session, ml_model: MLModel, dataset: Dataset):
    """Дообучение ML-модели на обновленном датасете."""
    clf = uf.load_model(ml_model.file)
    train_model(db, ml_model, dataset, clf)


def create_test(db: Session, training_id: int, dataset: Dataset):
    return add_entity(
        db, Test, training_id=training_id, dataset_id=dataset.id,
        dataset_version=dataset.version, status="NOT STARTED",
        last_update=datetime.now()
    )


def update_test(db: Session, test: Test,
                status: str, **kwargs):
    return update_entity(
        db, Test, test.id,
        status=status, last_update=datetime.now(), **kwargs
    )


def test_model(
        db: Session,
        ml_model: MLModel,
        last_training: Training,
        dataset: Dataset,
):
    start_t = time()
    test = create_test(db, last_training.id, dataset)
    df = pd.read_csv(dataset.file)
    status_update_func = __test_db_update_status_func(db, test)
    clf = uf.load_model(ml_model.file)
    y_pred = pd.Series(
        clf.predict(df, preprocessing=False, logger=status_update_func)
    )
    y_true = df.author
    f1 = f1_score(y_true, y_pred, average='weighted')
    update_test(
        db, test, status="FINISHED",
        f1_score=f1,
        y_true=y_true.to_csv(index=False),
        y_pred=y_pred.to_csv(index=False),
        elapsed_time=time() - start_t
    )


def get_ml_model_by_name(db: Session, model_name: str):
    """Получает ML-модель по имени."""
    return get_entity(db, MLModel, 'name', model_name)


def get_ml_model_by_id(db: Session, model_id: int):
    """Получает ML-модель по id."""
    return get_entity(db, MLModel, 'id', model_id)


def create_ml_model(db: Session, model_name: str, description: str | None):
    if not check_slug_field(model_name, max_len=25):
        raise ValueError("Invalid model name! (Max length: 25. "
                         "Must match regex `[a-zA-Z0-9_-]+$`)")
    return add_entity(
        db, MLModel, name=model_name, description=description
    )


def delete_ml_model(db: Session, ml_model: MLModel):
    """Удаляет ML-модель с указанным id."""
    uf.delete_file(ml_model.file)
    return delete_entity(db, MLModel, 'id', ml_model.id)


def get_all_ml_models(db: Session):
    """Получает все ML-модели."""
    return get_all(db, MLModel)


def get_training_by_id(db: Session, training_id: int):
    """Получает информацию об обучении по id."""
    return get_entity(db, Training, 'id', training_id)


def get_all_training_entities(db: Session):
    """Получает все записи об обучениях."""
    return get_all(db, Training)


def get_test_by_id(db: Session, test_id: int):
    """Получает информацию о тестировании по id."""
    return get_entity(db, Test, 'id', test_id)


def get_all_tests(db: Session):
    """Получает все записи о тестах."""
    return get_all(db, Test)

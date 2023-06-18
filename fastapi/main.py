from fastapi import (
    FastAPI, HTTPException, BackgroundTasks,
    File, UploadFile, Form, Depends, Response
)
from time import time
import joblib
from db import session_factory
from sqlalchemy.orm import Session

import utils_dataset as ds

import crud
from schemas import *

app = FastAPI(openapi_tags=tags_metadata)


def get_db():
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


@app.get('/', tags=["Status"])
async def root():
    return {'message': 'OK'}


@app.get('/dataset/{dataset_id}', tags=["Dataset"])
async def get_dataset_info(
        dataset_id: int,
        db: Session = Depends(get_db)
) -> DatasetDetailed:
    """Информация о сохраненном датасете.
    Список моделей, использующих датасет."""
    dataset = crud.get_dataset_by_id(db, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404)
    return dataset


@app.get('/dataset', tags=["Dataset"])
async def get_dataset_list(
        dataset_name: str | None = None,
        db: Session = Depends(get_db)
) -> List[Dataset]:
    """Если передается параметр `dataset_name`, то в ответе
    содержится краткая информация о датасете с указанным именем.
    В противном случае в ответе содержится список всех сохраненных
    датасетов."""
    if dataset_name is not None:
        dataset = crud.get_dataset_by_name(db, dataset_name)
        if dataset is None:
            raise HTTPException(status_code=404)
        return [dataset]
    else:
        datasets = crud.get_all_datasets(db)
        return datasets


@app.post('/dataset', tags=["Dataset"], status_code=201)
async def add_dataset(
        name: str = Form(...),
        description: str | None = Form(default=None),
        upload_file: UploadFile = File(...),
        db: Session = Depends(get_db)
) -> DatasetDetailed:
    """Добавление датасета (тренировочного или тестового)."""
    try:
        dataset = crud.save_dataset_from_uploaded_csv(
            db, name, description, upload_file)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
    return dataset


@app.put('/dataset/{dataset_id}', tags=["Dataset"])
async def update_dataset_info(
        dataset_id: int,
        description: str | None = Form(default=None),
        db: Session = Depends(get_db)
) -> DatasetDetailed:
    """Обновление описания датасета."""
    dataset = crud.get_dataset_by_id(db, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404)
    return crud.update_dataset_description(db, dataset.id, description)


@app.patch('/dataset/{dataset_id}', tags=["Dataset"])
async def add_texts_to_dataset(
        dataset_id: int,
        upload_file: UploadFile = File(...),
        db: Session = Depends(get_db)
) -> Dataset:
    """Добавление текстов в датасет."""
    dataset = crud.get_dataset_by_id(db, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404)
    try:
        dataset = crud.add_texts_to_dataset(db, dataset, upload_file)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
    return dataset


@app.delete('/dataset/{dataset_id}', tags=["Dataset"], status_code=204)
async def delete_dataset(
        dataset_id: int,
        db: Session = Depends(get_db)
):
    """Удаление датасета."""
    dataset = crud.get_dataset_by_id(db, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404)
    if len(dataset.training) + len(dataset.tests) > 0:
        raise HTTPException(status_code=422,
                            detail="Delete related ML models first!")
    crud.delete_dataset(db, dataset)
    return Response('', 204)


@app.get('/author/{author_id}', tags=["Author"])
async def get_author_info(
        author_id: int,
        db: Session = Depends(get_db)
) -> AuthorDetailed:
    """Информация об авторе (в каких датасетах есть тексты автора
    и в каком количестве)."""
    author = crud.get_author_by_id(db, author_id)
    if author is None:
        raise HTTPException(status_code=404)
    return author


@app.get('/author', tags=["Author"])
async def get_author_list(
        name: str | None = None,
        db: Session = Depends(get_db)
) -> List[Author]:
    """Если передается параметр `name`, то в ответе содержится
    информация об авторе с указанным «именем». В противном случае в ответе
    содержится список всех известных приложению авторов."""
    if name is not None:
        author = crud.get_author_by_name(db, name)
        if author is None:
            raise HTTPException(status_code=404)
        return [author]
    else:
        authors = crud.get_all_authors(db)
        return authors


@app.patch('/author/{author_id}', tags=["Author"])
async def update_author_info(
        author_id: int,
        first_name: str | None = Form(default=None),
        surname: str | None = Form(default=None),
        patronymic: str | None = Form(default=None),
        db: Session = Depends(get_db)
) -> Author:
    """Обновление информации об авторе (ФИО)."""
    author = crud.get_author_by_id(db, author_id)
    if author is None:
        raise HTTPException(status_code=404)
    return crud.update_author(db, author.id, first_name, surname, patronymic)


@app.get('/train/{training_id}', tags=["Train"])
async def get_training_by_id(
        training_id: int,
        db: Session = Depends(get_db)
) -> Training:
    """Получение информации о запущенном
    или завершенном обучении ML-модели."""
    training = crud.get_training_by_id(db, training_id)
    if training is None:
        raise HTTPException(status_code=404)
    return training


@app.get('/train', tags=["Train"])
async def get_training_list(
        db: Session = Depends(get_db)
) -> List[TrainingShort]:
    """Получение списка всех обучений ML-моделей."""
    return crud.get_all_training_entities(db)


@app.post('/train', tags=["Train"], status_code=202)
async def train_model(
        background_tasks: BackgroundTasks,
        model_name: str = Form(...),
        model_description: str | None = Form(default=None),
        dataset_id: int = Form(...),
        db: Session = Depends(get_db),
) -> MLModel:
    """Создание и обучение модели машинного обучения."""
    if crud.get_ml_model_by_name(db, model_name) is not None:
        raise HTTPException(
            status_code=422, detail="model_name must be unique!")
    dataset = crud.get_dataset_by_id(db, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found!")
    ml_model = crud.create_ml_model(db, model_name, model_description)
    background_tasks.add_task(crud.train_model, db, ml_model, dataset)
    return ml_model


@app.patch('/train/{model_id}', tags=["Train"], status_code=202)
async def continue_training(
        background_tasks: BackgroundTasks,
        model_id: int,
        db: Session = Depends(get_db)
):
    """Дообучение модели (если тренировочный датасет обновился)."""
    ml_model = crud.get_ml_model_by_id(db, model_id)
    if ml_model is None:
        raise HTTPException(status_code=404)
    last_training = crud.get_last_training(db, model_id)
    if last_training.status != 'FINISHED':
        raise HTTPException(
            status_code=422, detail="Model is training at the moment!")
    dataset = crud.get_dataset_by_id(db, last_training.dataset_id)
    if last_training.dataset_version == dataset.version:
        raise HTTPException(status_code=422, detail="Model is up to date.")
    background_tasks.add_task(crud.continue_training, db, ml_model, dataset)
    return Response(status_code=202)


@app.get('/test/{test_id}', tags=["Test"])
async def get_test_by_id(
        test_id: int,
        db: Session = Depends(get_db)
) -> TestDetailed:
    """Получение результатов ранее проведенной оценки качества ML-модели
    по id тестирования."""
    test = crud.get_test_by_id(db, test_id)
    if test is None:
        raise HTTPException(status_code=404)
    return test


@app.get('/test', tags=["Test"])
async def get_test_list(db: Session = Depends(get_db)) -> List[TestShort]:
    """Получение результатов ранее проведенных оценок качества ML-моделей.
    Список можно отфильтровать по id модели (`model_id`)."""
    return crud.get_all_tests(db)


@app.post('/test/{model_id}', tags=["Test"], status_code=202)
async def test_model(
        background_tasks: BackgroundTasks,
        model_id: int,
        dataset_id: int = Form(...),
        db: Session = Depends(get_db)
):
    """Тестирование модели на одном из сохраненных датасетов."""
    ml_model = crud.get_ml_model_by_id(db, model_id)
    last_training = crud.get_last_training(db, model_id)
    dataset = crud.get_dataset_by_id(db, dataset_id)
    if None in (ml_model, last_training, dataset):
        raise HTTPException(status_code=404)
    if last_training.status != 'FINISHED':
        raise HTTPException(
            status_code=422, detail="Model is training at the moment!")
    background_tasks.add_task(
        crud.test_model, db, ml_model, last_training, dataset)
    return Response('', status_code=202)


@app.put('/predict/{model_id}', tags=["Predict"])
async def predict(
        model_id: int,
        upload_file: UploadFile = File(...),
        db: Session = Depends(get_db)
) -> PredictionResults:
    start_t = time()
    model = crud.get_ml_model_by_id(db, model_id)
    if model is None:
        raise HTTPException(status_code=404)
    try:
        df = ds.input_file_to_df(upload_file.file, upload_file.filename)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=f"Invalid file: {err}")
    if crud.get_last_training(db, model_id).status != 'FINISHED':
        raise HTTPException(status_code=422,
                            detail="Model is training at the moment!")
    clf = joblib.load(model.file)
    predictions = clf.predict(df).tolist()
    return PredictionResults(
        n_texts=len(predictions),
        predictions=predictions,
        elapsed_time=time()-start_t
    )


@app.get('/model/{model_id}', tags=["Model"])
async def get_model_info(
        model_id: int,
        db: Session = Depends(get_db)
) -> MLModel:
    """Информация об ML-модели."""
    ml_model = crud.get_ml_model_by_id(db, model_id)
    if ml_model is None:
        raise HTTPException(status_code=404)
    return ml_model


@app.delete('/model/{model_id}', tags=["Model"], status_code=204)
async def delete_model(
        model_id: int,
        db: Session = Depends(get_db)
):
    """Удаление ML-модели и всей связанной с ней информации."""
    ml_model = crud.get_ml_model_by_id(db, model_id)
    if ml_model is None:
        raise HTTPException(status_code=404)
    crud.delete_ml_model(db, ml_model)
    return Response('', 204)


@app.get('/model', tags=["Model"])
async def get_model_list(
        model_name: str | None = None,
        db: Session = Depends(get_db)
) -> List[MLModelForTraining]:
    """Если передается параметр `model_name`, то в ответе
    содержится краткая информация о модели с указанным именем.
    В противном случае в ответе содержится список всех сохраненных
    моделей."""
    if model_name is not None:
        ml_model = crud.get_ml_model_by_name(db, model_name)
        if ml_model is None:
            raise HTTPException(status_code=404)
        return [ml_model]
    else:
        ml_models = crud.get_all_ml_models(db)
        return ml_models


@app.put('/merge_txt', tags=["Tools"])
async def merge_txt(upload_files: UploadFile = File(...)):
    ...


@app.put('/merge_csv', tags=["Tools"])
async def merge_csv(upload_files: UploadFile = File(...)):
    ...


@app.put('/remove_short_texts', tags=["Tools"])
async def remove_short_texts(upload_file: UploadFile = File(...)):
    ...

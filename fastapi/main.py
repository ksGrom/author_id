from fastapi import (
    FastAPI, HTTPException, BackgroundTasks,
    File, UploadFile, Form, Depends, Response
)
from time import time
import joblib
from db import session_factory, MLModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import utils_dataset as ds

import utils_tfidf_clf as u
import crud
import utils_files as uf
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


@app.get('/dataset/{dataset_id}', tags=["Dataset"], response_model=Dataset)
async def get_dataset_info(dataset_id: int, db: Session = Depends(get_db)):
    """Информация о сохраненном датасете.
    Список моделей, использующих датасет."""
    dataset = crud.get_dataset_by_id(db, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404)
    return dataset


@app.get('/dataset', tags=["Dataset"], response_model=List[DatasetBriefly])
async def get_dataset_list(
        dataset_name: str | None = None,
        db: Session = Depends(get_db)
):
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


@app.post('/dataset', tags=["Dataset"], response_model=Dataset,
          status_code=201)
async def add_dataset(
        name: str = Form(...),
        description: str | None = Form(default=None),
        upload_file: UploadFile = File(...),
        db: Session = Depends(get_db)
):
    """Добавление датасета (тренировочного или тестового)."""
    try:
        dataset = crud.save_dataset_from_uploaded_csv(
            db, name, description, upload_file)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
    return dataset


@app.put('/dataset/{dataset_id}', tags=["Dataset"], response_model=Dataset)
async def update_dataset_info(
        dataset_id: int,
        description: str | None = Form(default=None),
        db: Session = Depends(get_db)
):
    """Обновление описания датасета."""
    dataset = crud.get_dataset_by_id(db, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404)
    return crud.update_dataset_description(db, dataset.id, description)


@app.patch('/dataset/{dataset_id}', tags=["Dataset"], response_model=Dataset)
async def add_texts_to_dataset(
        dataset_id: int,
        upload_file: UploadFile = File(...),
        db: Session = Depends(get_db)
):
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
    crud.delete_dataset(db, dataset)
    return Response('', 204)


@app.get('/author/{author_id}', tags=["Author"],
         response_model=AuthorDetailed)
async def get_author_info(author_id: int, db: Session = Depends(get_db)):
    """Информация об авторе (в каких датасетах есть тексты автора
    и в каком количестве)."""
    author = crud.get_author_by_id(db, author_id)
    if author is None:
        raise HTTPException(status_code=404)
    return author


@app.get('/author', tags=["Author"], response_model=List[Author])
async def get_author_list(
        name: str | None = None,
        db: Session = Depends(get_db)
):
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


@app.patch('/author/{author_id}', tags=["Author"], response_model=Author)
async def update_author_info(
        author_id: int,
        first_name: str | None = Form(default=None),
        surname: str | None = Form(default=None),
        patronymic: str | None = Form(default=None),
        db: Session = Depends(get_db)
):
    """Обновить информацию об авторе (ФИО)."""
    author = crud.get_author_by_id(db, author_id)
    if author is None:
        raise HTTPException(status_code=404)
    return crud.update_author(db, author.id, first_name, surname, patronymic)


@app.get('/train', tags=["Train"])
async def get_train_info():
    """Получить информацию о запущенном или завершенном обучении ML-модели."""
    ...


@app.post('/train', tags=["Train"])
async def train_model(
    model_name: str = Form(...),
    description: str | None = Form(default=None),
    upload_file: UploadFile = File(...)
):
    """Инициализация новой модели машинного обучения."""
    start_t = time()
    filename = f"{start_t}.pkl"
    clf = u.AuthorIdTfidfPipeline()
    uf.save_model_file(clf, filename)
    try:
        model = crud.add_entity(
            MLModel, name=model_name, description=description)
        df = ds.input_file_to_df(upload_file.file, upload_file.filename)
        df = ds.make_dataset_of_excerpts(
            df=df, excerpt_num_of_words=u.EXCERPT_LEN)
        clf.fit(df)
        path = uf.save_model_file(clf, filename)
        crud.update_entity(MLModel, model.id, file=str(path))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid file: {e}")
    except IntegrityError:
        raise HTTPException(
            status_code=422, detail="model_name must be unique!")

    return TrainResults(
        id=model.id,
        exec_time=time()-start_t
    )


@app.put('/train', tags=["Train"])
async def continue_training():
    """Дообучение модели (если тренировочный датасет обновился)."""
    ...


@app.get('/test/{test_id}', tags=["Test"])
async def get_test_info_by_id(test_id: int):
    """Получение результатов ранее проведенной оценки качества ML-модели
    по id тестирования."""
    ...


@app.get('/test', tags=["Test"])
async def get_tests_info_by_model_id():
    """Получение результатов ранее проведенных оценок качества ML-модели
    по id или названию ML-модели."""
    ...


@app.post('/test', tags=["Test"])
async def test_model():
    """Тестирование модели на одном из сохраненных датасетов."""
    ...


@app.post('/predict', tags=["Predict"])
def predict(upload_file: UploadFile = File(...)) -> PredictionResults:
    start_t = time()
    try:
        df = ds.input_file_to_df(upload_file.file, upload_file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid file: {e}")
    clf = joblib.load("default.pkl")
    predictions = clf.predict(df).tolist()
    return PredictionResults(
        n_texts=len(predictions),
        predictions=predictions,
        exec_time=time()-start_t
    )


@app.put('/merge_txt', tags=["Tools"])
def merge_txt(upload_files: UploadFile = File(...)):
    ...


@app.put('/merge_csv', tags=["Tools"])
def merge_csv(upload_files: UploadFile = File(...)):
    ...


@app.put('/remove_short_texts', tags=["Tools"])
def remove_short_texts(upload_file: UploadFile = File(...)):
    ...


if __name__ == "__main__":
    session = session_factory()
    people_query = session.query(MLModel)
    session.close()
    people = people_query.all()

    for person in people:
        print(f'{person.id}, {person.file}')
        print(f'{person.name} was born in {person.datetime}')
        print(f'INFO {person.description}')
        print('_' * 30)

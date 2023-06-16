from fastapi import (
    FastAPI, HTTPException, BackgroundTasks,
    File, UploadFile, Form
)
from pydantic import BaseModel
from typing import List, Union
from time import time
import joblib
from db import session_factory, MLModel
from sqlalchemy.exc import IntegrityError

import utils_dataset as ds

import utils_tfidf_clf as u
import utils_db as udb
import utils_files as uf

app = FastAPI()


class PredictionResults(BaseModel):
    n_texts: int
    predictions: List[Union[int, str]]
    exec_time: float


class TrainResults(BaseModel):
    id: int
    exec_time: float


@app.get('/')
async def root():
    return {'message': 'OK'}


@app.get('/dataset')
async def dataset_info():
    """Информация о сохраненном датасете.
    Список моделей, использующих датасет.
    """
    pass


@app.post('/dataset')
async def add_dataset():
    """Добавление тренировочного или тестового датасета.
    """
    pass


@app.patch('/dataset')
async def add_texts_to_dataset():
    """Добавление текстов в датасет."""
    pass


@app.post('/train')
async def train(
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
        model = udb.add_entity(
            MLModel, name=model_name, description=description)
        df = ds.input_file_to_df(upload_file.file, upload_file.filename)
        df = ds.make_dataset_of_excerpts(
            df=df, excerpt_num_of_words=u.EXCERPT_LEN)
        clf.fit(df)
        path = uf.save_model_file(clf, filename)
        udb.update_entity(MLModel, model.id, file=str(path))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid file: {e}")
    except IntegrityError:
        raise HTTPException(
            status_code=422, detail="model_name must be unique!")

    return TrainResults(
        id=model.id,
        exec_time=time()-start_t
    )


@app.put('/train')
async def continue_training():
    """Дообучение модели (если тренировочный датасет обновился)."""
    ...


@app.put('/test')
async def test_model():
    """Тестирование модели на одном их сохраненных датасетов."""
    ...


@app.post('/predict')
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

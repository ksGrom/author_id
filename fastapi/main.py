from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile
from pydantic import BaseModel
from typing import List, Union
from time import time
import joblib

from dataset_utils import (
    input_file_to_df,
)

app = FastAPI()


class UploadedTextsResults(BaseModel):
    n_texts: int
    predictions: List[Union[int, str]]
    exec_time: float


@app.get('/')
async def root():
    return {'message': 'OK'}


@app.post("/predict_items")
def predict_items(upload_file: UploadFile = File(...)) -> UploadedTextsResults:
    start_t = time()
    try:
        df = input_file_to_df(upload_file.file, upload_file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid file: {e}")
    clf = joblib.load("default.pkl")
    predictions = clf.predict(df).tolist()
    return UploadedTextsResults(
        n_texts=len(predictions),
        predictions=predictions,
        exec_time=time()-start_t
    )

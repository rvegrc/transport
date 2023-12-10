from fastapi import FastAPI
from fastapi import UploadFile, File

from pydantic import BaseModel

# standard library imports
import pandas as pd
import numpy as np
import sklearn

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import xgboost as xgb
import optuna

import json

from optuna.visualization.matplotlib import plot_param_importances

import mlflow


from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline as skl_pipeline

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

sklearn.set_config(transform_output='pandas')

# load metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

from typing import List
import typing

pd.set_option('display.float_format', '{:.3f}'.format)

# constant
RAND_ST = 345

model = joblib.load('best_model.joblib')


class DfRow(BaseModel):
    
    distance: int
    own_container: int
    complect_send: int
    container_train: int
    transportation_type: int
    days: int

class DataF(BaseModel):
    data: List[DfRow]


class DfOut(BaseModel):
    '''df output format'''
    distance: int
    own_container: int
    complect_send: int
    container_train: int
    transportation_type: int
    days: int


class XtestOut(BaseModel):
    '''xtest output format'''
    distance: list
    own_container: list
    complect_send: list
    container_train: list
    transportation_type: list
    days: list

class DFOut2(BaseModel):
    data: XtestOut




class TextOut(BaseModel):
    '''format text out'''
    test: str


class SerialOut(BaseModel):
    '''serias out'''
    data: int


class XtestDout(BaseModel):
    '''xtest out from dict form to_dict'''
    distance: dict
    own_container: dict
    complect_send: dict
    container_train: dict
    transportation_type: dict
    days: dict



app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Model prediction"}



@app.post('/text')
async def train(test:TextOut):
    '''output text'''
    # print(test)
    print(test)
    return {'text': test}


@app.post('/xout_list')
async def serial(data:XtestOut):
    'output second item of serias'
    print(data)
    # print(pd.DataFrame(json.loads(data.json())))
    # print(model.predict(pd.DataFrame(data)))
    return data

@app.post('/xout_dict')
async def xout_dict(data:XtestDout):
    print(data.json())
    # print(pd.DataFrame(json.load(data.json())))
    return data




@app.post("/predict")
async def predict(data:XtestOut):
    X_test_out = pd.DataFrame(json.loads(data.json()))
    predict_x_test_out = model.predict(X_test_out)
    # print(json.dumps(predict_x_test_out.tolist()))
    print(pd.Series(predict_x_test_out))
    # df_data = pd.DataFrame(json.loads(data.json()))
    # print(df_data)
    # return  json.dumps(predict_x_test_out.tolist()) # predictions # recive dataset, make prediction, return prediction,
    return  predict_x_test_out.tolist()



@app.get("/test")
async def test():
    return {"Hello": "Api Fast"}



@app.get("/train/{item}")
async def train(item):
    return {"Train": item}

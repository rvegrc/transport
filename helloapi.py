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


from typing import List, Dict, Any
import typing

pd.set_option('display.float_format', '{:.3f}'.format)

# constant
RAND_ST = 345

model = joblib.load('best_model.joblib')



class XInputDict(BaseModel):
    '''X input from to_dict orient dict'''
    distance: Dict[int, Any]
    own_container: Dict[int, Any]
    complect_send: Dict[int, Any]
    container_train: Dict[int, Any]
    transportation_type: Dict[int, Any]
    days: Dict[int, Any]


class XInputIndex(BaseModel):
    '''X input from to_dict orient dict'''
    distance: Dict
    own_container: Dict
    complect_send: Dict
    container_train: Dict
    transportation_type: Dict
    days: Dict



class XInputList(BaseModel):
    '''X input from to_dict orient list'''
    distance: List
    own_container: List
    complect_send: List
    container_train: List
    transportation_type: List
    days: List






app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Model prediction"}



def out(X_input):
    predict_x = model.predict(X_input)
    # print(json.dumps(predict_x_test_out.tolist()))
    # print(X_input.index)
    # print(pd.Series(predict_x_test_out))
    # df_data = pd.DataFrame(json.loads(data.json()))
    # print(df_data)
    # return  json.dumps(predict_x_test_out.tolist()) # predictions # recive dataset, make prediction, return prediction,
    return  pd.DataFrame(predict_x.tolist(), columns=['y_pred'], index=X_input.index)

    

@app.post("/dict_predict") # may be def for each endpoints
async def predict(data:XInputDict):
    X_input = pd.DataFrame(json.loads(data.json()))
    print(X_input)
    return out(X_input)
    
@app.post("/list_predict")  
async def predict(data:XInputList):
    X_input = pd.DataFrame(json.loads(data.json()))
    print(X_input)
    return out(X_input)

@app.post("/index_predict")  
async def predict(data:XInputIndex):
    # X_input = pd.DataFrame(json.loads(data.json()))
    print(data)
    # print(pd.DataFrame(json.loads(data.json())))  
    # return out(X_input)


@app.get("/test")
async def test():
    return {"Hello": "Api Fast"}



@app.get("/train/{item}")
async def train(item):
    return {"Train": item}

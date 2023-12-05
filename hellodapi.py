from fastapi import FastAPI

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

pd.set_option('display.float_format', '{:.3f}'.format)

# constant
RAND_ST = 345

model = joblib.load('model_test.joblib')


class PredOut(BaseModel):
    distance: float
    own_container: int
    complect_send: int
    container_train: int
    transportation_type: int
    days: int


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Model prediction"}


@app.post("/predict")
async def predict(data:PredOut):
    predictions = model.predict(data)
    return  predictions # recive dataset, make prediction, return prediction,

@app.get("/test")
async def test():
    return {"Hello": "Api Fast"}

@app.get("/train/{item}")
async def train(item):
    return {"Train": item}

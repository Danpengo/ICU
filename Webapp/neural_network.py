from venv import create
from pandas import *
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pip
import sys
import logging
import tensorflow as tf
from sklearn.utils import compute_class_weight
import numpy as np
import joblib

def create_baseline():
        # create model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(32, input_shape=(237,), activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

def nn_predictor(dataset, model):
    # load dataset
    dataset = dataset.values
    
    # split into input (X) and output (Y) variables
    X = dataset[:, :-1].astype(float)
    Y = dataset[:, -1]
    
    # load your model for further usage
    grid_result = joblib.load("model_file_name.pkl")

    ypred2 = grid_result.predict(X)
    ypred2_probability = grid_result.predict_proba(X)

    cr = classification_report(Y, ypred2)

    return grid_result, cr, Y, ypred2, ypred2_probability
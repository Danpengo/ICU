from pandas import *
from sklearn.model_selection import train_test_split
import pip
import sys
import logging
import tensorflow as tf
import joblib
import numpy as np
import lime.lime_tabular
import matplotlib.pyplot as plt

def LIME_explainer(model):

    dataframe = read_csv("ICU_dataset_death_knnimputed.csv")
    x = dataframe.iloc[:, :-1]

    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    X = dataset[:, :-1].astype(float)
    Y = dataset[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

    model = model.best_estimator_

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=list(x), class_names=[0, 1], mode='classification')

    return explainer


def LIME_sample(dataset, model, explainer, sample):

    predictors = len(dataset.columns) - 1
    
    dataset = dataset.values

    X_new = dataset[:, :-1].astype(float)

    exp = explainer.explain_instance(X_new[sample], model.predict_proba, num_features=predictors)

    exp.save_to_file('lime_test.html')


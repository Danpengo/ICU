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


# logging.getLogger("tensorflow").setLevel(logging.ERROR)

# pip.main(['install','scikeras'])

# load dataset
dataframe = read_csv("ICU_dataset_death_knnimputed.csv")
predictors = len(dataframe.columns) - 1

dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, :-1].astype(float)
Y = dataset[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)


# baseline model

def create_baseline():
    # create model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, input_shape=(237,), activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


batch_size = [20, 40, 60, 80, 100]
epochs = [20, 50, 100]
# best epochs = 5 and batch size = 60

param_grid = dict(clf__batch_size=batch_size, clf__epochs=epochs)
st = StandardScaler()

model = KerasClassifier(model=create_baseline)
pipeline = Pipeline(steps=[('scaler', st),
                               ('clf', model)])

kfold = StratifiedKFold(n_splits=10, shuffle=True)
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=kfold, scoring="f1")

grid_result = grid.fit(X_train, y_train, clf__class_weight=class_weights)
# grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Predictions

ypred = grid_result.predict(X_train)
print(classification_report(y_train, ypred))
print('######################')
ypred2 = grid_result.predict(X_test)
print(classification_report(y_test, ypred2))


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import random

#add different numbers of neurons and epochs
def going():

    
    df = pd.read_csv("ready_csvs/ready_for_polynomial.csv")

    df = df.drop(columns=['EvergreenBroadleafTrees', 'DeciduousBroadleafTrees', 'Shrubs', 'OpenWater', 'keys'])

    bins = [-np.inf, 1, 2, 4, np.inf]
    labels = [0, 1, 2, 3]

    mosq = pd.cut(df["Mosquito"], bins=bins, labels=labels)
    y = mosq
    X = df.drop(["Mosquito"], axis=1)

    ss = StandardScaler()
    X_scaled = ss.fit_transform(X)

    num_folds = 5

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)


    
    best = -np.inf
    for _ in range(30):
        model = keras.Sequential([
            keras.layers.Dense(48, input_shape=(31,), activation='relu'),
            keras.layers.Dense(46, activation='exponential'),
            keras.layers.Dense(4, activation="softmax")
        ])

        model.compile(optimizer='rmsprop', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]


            


            model.fit(X_train, y_train, epochs=5, verbose=0)

            loss, accuracy = model.evaluate(X_test, y_test)

            if accuracy > best:
                best = accuracy
                model.save("models/neural.h5")


    print(f"Accuracy {best}")


import timeit

x = timeit.timeit(going, number=2)

print(f"Time: {x}")
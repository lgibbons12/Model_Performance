#imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler


#add different numbers of neurons and epochs
def going():

    #download polynomial data and remove outliers
    df = pd.read_csv("ready_csvs/ready_for_polynomial.csv")
    df = df.drop(columns=['EvergreenBroadleafTrees', 'DeciduousBroadleafTrees', 'Shrubs', 'OpenWater', 'keys'])


    #first we have to create classification bins
    #these bins are created to be relatively equal amounts of mosquitos
    #the model will try to predict which bin a row goes into based on the data
    #These bins split the data into five categories: values less than or equal to 1, values between 1 and 2, values between 2 and 4, values between 4 and positive infinity.
    bins = [-np.inf, 1, 2, 4, np.inf]
    labels = [0, 1, 2, 3]

    #get the data ready for splitting
    mosq = pd.cut(df["Mosquito"], bins=bins, labels=labels)
    y = mosq
    X = df.drop(["Mosquito"], axis=1)

    ss = StandardScaler()
    X_scaled = ss.fit_transform(X)

    #create the K folds classifier which will be used to check each split with different trains and testing
    #so each split will be tested 5 times instead of once
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)


    
    best = -np.inf
    for _ in range(30):
        #create the model (this is done inside because otherwise each iteration will train the same model and make it better)
        #we have an input layer, one hidden layer, and then an output layer that outputs to the 4 bins
        #the number of neurons (48 and 46) and activations (relu and exponential) were chosen by testing tons of different combinations, and these were the best
        #the output layer has a softmax which has all 4 neurons weights add up to 1, and the classifier picks the largest neuron as the classification
        model = keras.Sequential([
            keras.layers.Dense(48, input_shape=(31,), activation='relu'),
            keras.layers.Dense(46, activation='exponential'),
            keras.layers.Dense(4, activation="softmax")
        ])

        #we compile the model, with the optimizer and loss functions chosen through hyperparameter testing
        model.compile(optimizer='rmsprop', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

        #looping through all combinations with the KFold train test split
        for train_index, test_index in kf.split(X_scaled):
            #split the data using the indexes from the KFold
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #fit the model
            model.fit(X_train, y_train, epochs=5, verbose=0)

            #evaluate
            loss, accuracy = model.evaluate(X_test, y_test)

            #if its the best model so far, save the accuracy and save the model
            if accuracy > best:
                best = accuracy
                model.save("models/neural.h5")


    print(f"Accuracy {best}")


import timeit

x = timeit.timeit(going, number=2)

print(f"Time: {x}")
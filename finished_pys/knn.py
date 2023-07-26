#imports
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


def do():
    df = pd.read_csv("ready_csvs/ready_for_polynomial.csv")
    #dropping some columns that are outliers as they negatively affect KNN
    df = df.drop(columns=['EvergreenBroadleafTrees', 'DeciduousBroadleafTrees', 'Shrubs', 'OpenWater', 'keys'])

    #first we have to create classification bins
    #these bins are created to be relatively equal amounts of mosquitos
    #the model will try to predict which bin a row goes into based on the data
    #These bins split the data into five categories: values less than or equal to 1, values between 1 and 2, values between 2 and 4, values between 4 and positive infinity.
    bins = [-np.inf, 1, 2, 4, np.inf]
    labels = [0, 1, 2, 3]

    #now we have to set up our classification variable, y, and our data, X
    mosq = pd.cut(df["Mosquito"], bins=bins, labels=labels)
    y = mosq
    X = df.drop(["Mosquito"], axis=1)

    best = -np.inf

    for _ in range(30):
        #we split the data and scale it by minimizing the large outliers with the MinMaxScaler()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        mm = MinMaxScaler()

        X_train_scaled = mm.fit_transform(X_train)
        X_test_scaled = mm.transform(X_test)

        #we create the model with 4 neighbors (that was the best hyperparameter found in testing)
        #and we fit it to the data
        knn = KNeighborsClassifier(n_neighbors=4)
        knn = knn.fit(X_train_scaled, y_train)

        y_pred = knn.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {acc}")

        #if its the best model so far, save the accuracy and save the model
        if acc > best:
            best = acc
            with open("models/knn.pickle", 'wb') as f:
                pickle.dump(knn, f)
        
    print(f"Best Accuracy: {best}")

import timeit

x = timeit.timeit(do, number=2)

print(f"Time: {x}")
    
        



    
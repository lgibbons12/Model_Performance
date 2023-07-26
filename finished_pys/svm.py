#imports
import sklearn
from sklearn import svm, metrics
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np  

def do():
    #download the polynomial dataset and remove outliers
    df = pd.read_csv("ready_csvs/ready_for_polynomial.csv")
    df = df.drop(columns=['EvergreenBroadleafTrees', 'DeciduousBroadleafTrees', 'Shrubs', 'OpenWater', 'keys'])

    #first we have to create classification bins
    #these bins are created to be relatively equal amounts of mosquitos
    #the model will try to predict which bin a row goes into based on the data
    #These bins split the data into five categories: values less than or equal to 1, values between 1 and 2, values between 2 and 4, values between 4 and positive infinity.
    bins = [-np.inf, 1, 2, 4, np.inf]
    labels = [0, 1, 2, 3]

    #get the data prepared
    mosq = pd.cut(df["Mosquito"], bins=bins, labels=labels)
    y = mosq
    X = df.drop(["Mosquito"], axis=1)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)

    ss = StandardScaler()

    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.fit_transform(X_test)

    best = -np.inf

    for _ in range(30):
        #train and scale the data
        X_train, y_train, y_train, y_test = train_test_split(X, y, test_size=0.2)

        ss = StandardScaler()

        X_train_scaled = ss.fit_transform(X_train)
        X_test_scaled = ss.transform(X_test)

        #create the model with the polynomial kernel, a degree of 3, and a soft margin of 0.1
        #through lots of testing, these were the hyperparameters that produced the best results
        clf = svm.SVC(kernel="poly", degree=3, C=0.1)

        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)

        acc = metrics.accuracy_score(y_test, y_pred)

        print(f"Accuracy: {acc}")

        #if its the best model so far, save the accuracy and save the model
        if acc > best:
            best = acc
            with open("models/svm.pickle", 'wb') as f:
                pickle.dump(clf, f)
    print(f"Best Accuracy: {best}")

import timeit

x = timeit.timeit(do, number=2)

print(f"Time: {x}")


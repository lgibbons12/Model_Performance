#imports
from sklearn.decomposition import PCA 
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import sklearn
import pickle

#main function used for timeit
def do():
    df = pd.read_csv("ready_csvs/ready_for_polynomial.csv")

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

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)

    #we must scale and normalize the data so that outliers and non_standard data don't skew our model
    ss = StandardScaler()

    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.fit_transform(X_test)


    #now we run a principal component analysis
    #this will help us limit the number of variables to keep the complexity of the tree low
    #while still maining much of the predictive power of our data
    pca_train = PCA(n_components=6)
    X_train_scaled_pca = pca_train.fit_transform(X_train_scaled)
    X_test_scaled_pca = pca_train.transform(X_test_scaled)



    best = -np.inf

    #loop through for best data
    for _ in range(30):

        #we must create new data, scale, and pca each time to truly find the best model
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)

        ss = StandardScaler()

        X_train_scaled = ss.fit_transform(X_train)
        X_test_scaled = ss.fit_transform(X_test)



        pca_train = PCA(n_components=6)


        X_train_scaled_pca = pca_train.fit_transform(X_train_scaled)
        X_test_scaled_pca = pca_train.transform(X_test_scaled)

        
        #we need to create a decision tree, fit it to the data, and make a prediction
        cool = DecisionTreeClassifier(max_depth=5)

        tree_fitted = cool.fit(X_train_scaled_pca, y_train)

        prediction = tree_fitted.predict(X_test_scaled_pca)

        acc = metrics.accuracy_score(y_test, prediction)

        print(f"Accuracy: {acc}")

        #if its the best model so far, save the accuracy and save the model
        if acc > best:
            
            best = acc
            with open("models/tree.pickle", 'wb') as f:
                pickle.dump(tree_fitted, f)

    print(f"Best Accuracy: {best}")

import timeit

x = timeit.timeit(do, number=2)

print(f"Time: {x}")

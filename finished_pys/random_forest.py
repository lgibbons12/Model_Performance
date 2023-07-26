#imports
from sklearn.decomposition import PCA 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import sklearn
import pickle

#each time we run the model, we want to dynamically generate parameters to put in the model
#this function uses RandomizedSearchCV to find the best depth and estimator parameters for each forest
def get_params(X_train_scaled_pca, y_train):
    param_dist = {'n_estimators': randint(50, 500),
                'max_depth': randint(1, 20)}

    rf = RandomForestClassifier()

    rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)

    rand_search.fit(X_train_scaled_pca, y_train)

    params = rand_search.best_params_

    return params['max_depth'], params['n_estimators']


def do():
    df = pd.read_csv("ready_csvs/ready_for_polynomial.csv")

    #they have outliers
    df = df.drop(columns=['EvergreenBroadleafTrees', 'DeciduousBroadleafTrees', 'Shrubs', 'OpenWater', 'keys'])


    #first we have to create classification bins
    #these bins are created to be relatively equal amounts of mosquitos
    #the model will try to predict which bin a row goes into based on the data
    #These bins split the data into five categories: values less than or equal to 1, values between 1 and 2, values between 2 and 4, values between 4 and positive infinity.
    bins = [-np.inf, 1, 2, 4, np.inf]
    labels = [0, 1, 2, 3]

    #split the data
    mosq = pd.cut(df["Mosquito"], bins=bins, labels=labels)
    y = mosq
    X = df.drop(["Mosquito"], axis=1)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)


    #scaling to normalize the data
    ss = StandardScaler()

    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.fit_transform(X_test)


    #we run a principal component analysis to reduce the number of variables
    #this reduces the complexity of the trees, preventing overfitting or the model taking forever to compile
    pca_train = PCA(n_components=6)


    X_train_scaled_pca = pca_train.fit_transform(X_train_scaled)
    
    X_test_scaled_pca = pca_train.transform(X_test_scaled)

    best = -np.inf
    for _ in range(30):

        #each time we must train, scale, and pca the data
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)

        ss = StandardScaler()

        X_train_scaled = ss.fit_transform(X_train)
        X_test_scaled = ss.fit_transform(X_test)



        pca_train = PCA(n_components=6)


        X_train_scaled_pca = pca_train.fit_transform(X_train_scaled)
        X_test_scaled_pca = pca_train.transform(X_test_scaled)

        #getting the parameters based on our data
        depth, estimators = get_params(X_train_scaled_pca, y_train)

        #create and fit our model
        cool = RandomForestClassifier(max_depth=depth, n_estimators=estimators)

        tree_fitted = cool.fit(X_train_scaled_pca, y_train)

        prediction = tree_fitted.predict(X_test_scaled_pca)

        acc = accuracy_score(y_test, prediction)

        print(f"Accuracy: {acc}")

        #if its the best model so far, save the accuracy and save the model
        if acc > best:
            
            best = acc
            with open("models/random_forest.pickle", 'wb') as f:
                pickle.dump(tree_fitted, f)

    print(f"Best Accuracy: {best}")

import timeit

x = timeit.timeit(do, number=2)

print(f"Time: {x}")

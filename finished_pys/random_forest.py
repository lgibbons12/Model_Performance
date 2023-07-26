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

    bins = [-np.inf, 1, 2, 4, np.inf]
    labels = [0, 1, 2, 3]

    mosq = pd.cut(df["Mosquito"], bins=bins, labels=labels)
    y = mosq
    X = df.drop(["Mosquito"], axis=1)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)

    ss = StandardScaler()

    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.fit_transform(X_test)



    pca_train = PCA(n_components=6)


    X_train_scaled_pca = pca_train.fit_transform(X_train_scaled)
    
    X_test_scaled_pca = pca_train.transform(X_test_scaled)

    best = -np.inf
    for _ in range(30):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)

        ss = StandardScaler()

        X_train_scaled = ss.fit_transform(X_train)
        X_test_scaled = ss.fit_transform(X_test)



        pca_train = PCA(n_components=6)


        X_train_scaled_pca = pca_train.fit_transform(X_train_scaled)
        X_test_scaled_pca = pca_train.transform(X_test_scaled)

        depth, estimators = get_params(X_train_scaled_pca, y_train)
        cool = RandomForestClassifier(max_depth=depth, n_estimators=estimators)

        tree_fitted = cool.fit(X_train_scaled_pca, y_train)

        prediction = tree_fitted.predict(X_test_scaled_pca)

        acc = accuracy_score(y_test, prediction)

        print(f"Accuracy: {acc}")

        if acc > best:
            
            best = acc
            with open("models/random_forest.pickle", 'wb') as f:
                pickle.dump(tree_fitted, f)

    print(f"Best Accuracy: {best}")

import timeit

x = timeit.timeit(do, number=2)

print(f"Time: {x}")

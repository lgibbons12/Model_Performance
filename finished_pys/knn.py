
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
    df = df.drop(columns=['EvergreenBroadleafTrees', 'DeciduousBroadleafTrees', 'Shrubs', 'OpenWater', 'keys'])
    bins = [-np.inf, 1, 2, 4, np.inf]
    labels = [0, 1, 2, 3]

    mosq = pd.cut(df["Mosquito"], bins=bins, labels=labels)
    y = mosq
    X = df.drop(["Mosquito"], axis=1)

    best = -np.inf

    for _ in range(30):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        mm = MinMaxScaler()

        X_train_scaled = mm.fit_transform(X_train)
        X_test_scaled = mm.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=4)
        knn = knn.fit(X_train_scaled, y_train)

        y_pred = knn.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {acc}")

        if acc > best:
            best = acc
            with open("models/knn.pickle", 'wb') as f:
                pickle.dump(knn, f)
        
    print(f"Best Accuracy: {best}")

import timeit

x = timeit.timeit(do, number=2)

print(f"Time: {x}")
    
        



    
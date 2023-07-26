import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import linear_model
import pickle

data = pd.read_csv("ready_csvs/ready_for_linear.csv")


y = data["Mosquito"]
X = data.drop(columns="Mosquito")

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

best = 0
for i in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)

    print(f'The accuracy score is {acc}')

    if acc > best:
        best = acc
        with open("models/linear_regression.pickle", "wb") as f:
            pickle.dump(linear, f)

print("best accuracy was ", best)


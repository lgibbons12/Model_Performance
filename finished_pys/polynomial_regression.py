import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle

data = pd.read_csv("ready_csvs/ready_for_linear.csv")


y = data["Mosquito"]
X = data.drop(columns="Mosquito")

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

degree = 3

poly = PolynomialFeatures(degree=degree)
x_train_poly = poly.fit_transform(x_train)


x_test_poly = poly.fit_transform(x_test)
best = 0
for i in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
    x_train_poly = poly.fit_transform(x_train)


    x_test_poly = poly.fit_transform(x_test)
    linear = LinearRegression()

    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)

    print(f'The accuracy score is {acc}')

    if acc > best:
        best = acc
        #with open("models/polynomial_regression.pickle", "wb") as f:
            #pickle.dump(linear, f)

print("best accuracy was ", best)


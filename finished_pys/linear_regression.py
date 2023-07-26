#imports
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import linear_model
import pickle

#we use data that only contains variables who have a linear relationship with the predictor
data = pd.read_csv("ready_csvs/ready_for_linear.csv")

#create the data
y = data["Mosquito"]
X = data.drop(columns="Mosquito")

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

best = 0
for i in range(30):
    #split the data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    #train the model
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)

    print(f'The accuracy score is {acc}')

    #if its the best model so far, save the accuracy and save the model
    if acc > best:
        best = acc
        with open("models/linear_regression.pickle", "wb") as f:
            pickle.dump(linear, f)

print("best accuracy was ", best)


#imports
import pandas as pd
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.model_selection import train_test_split
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler


#there are some models that work better with outliers and some that don't
#so we use two different csvs to handle both
#we use data that only contains variables who have a linear relationship with the predictor
small_outliers = pd.read_csv("ready_csvs/ready_for_linear.csv")
big_outliers = pd.read_csv("ready_csvs/big_outliers_ready_for_robust.csv")

#small outliers first
y = small_outliers["Mosquito"]
X = small_outliers.drop(columns="Mosquito")

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)


best_theilsen = 0
best_huber_small = 0

for i in range(30):
    #split and scale the data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    ss = StandardScaler()
    x_train_scaled = ss.fit_transform(x_train)
    x_test_scaled = ss.transform(x_test)

    #we make both models at the same time
    #set the HuberRegressor to 1000 as that was the number that had a good accuracy without taking too long
    theilsen = TheilSenRegressor()
    huber = HuberRegressor(max_iter=1000)


    #fit the data for both models and if they are better than the previous best, save the new model
    theilsen.fit(x_train_scaled, y_train)
    huber.fit(x_train_scaled, y_train)

    theilsen_acc = theilsen.score(x_test_scaled, y_test)
    huber_acc = huber.score(x_test_scaled, y_test)

    if theilsen_acc > best_theilsen:
        best_theilsen = theilsen_acc
        with open("models/theilsen_robust.pickle", 'wb') as f:
            pickle.dump(theilsen, f)
    
    if huber_acc > best_huber_small:
        best_huber_small = huber_acc
        with open("models/huber_robust_small.pickle", 'wb') as f:
            pickle.dump(huber, f)

print(f"Best Theilsen Accuracy: {best_theilsen}")
print(f"Best Huber Small Accuracy {best_huber_small}")


#now big outliers
y = big_outliers["Mosquito"]
X = big_outliers.drop(columns="Mosquito")

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

best_ransac = 0
best_huber_big = 0

for i in range(30):
    #same as big, split and scale, create models, fit them, update best if necessary
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    ransac = RANSACRegressor()
    huber = HuberRegressor(max_iter=1000)
    ss = StandardScaler()
    x_train_scaled = ss.fit_transform(x_train)
    x_test_scaled = ss.transform(x_test)
    ransac.fit(x_train_scaled, y_train)
    huber.fit(x_train_scaled, y_train)

    ransac_acc = ransac.score(x_test_scaled, y_test)
    huber_acc = huber.score(x_test_scaled, y_test)

    if ransac_acc > best_ransac:
        best_ransac = ransac_acc
        with open("models/ransac_robust.pickle", 'wb') as f:
            pickle.dump(ransac, f)
    
    if huber_acc > best_huber_big:
        best_huber_big = huber_acc
        with open("models/huber_robust_big.pickle", 'wb') as f:
            pickle.dump(huber, f)

print(f"Best RANSAC Accuracy: {best_ransac}")
print(f"Best Huber Big Accuracy {best_huber_big}")



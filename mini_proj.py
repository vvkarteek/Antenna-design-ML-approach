
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv('/Users/vvkar/OneDrive/Desktop/Mini project/Antenna-dataset.csv')
dataset.head()
print(dataset.isnull().sum())
print(dataset.shape)

import warnings
warnings.filterwarnings('ignore')
from scipy.stats import norm
plt.figure(figsize = (9, 6))
print("\n")
sns.distplot(dataset['s11(dB)'], color = 'red', fit = norm) 
plt.show()
plt.figure(figsize = (12, 8))
X = dataset.iloc[:, :-1]
Y = dataset['s11(dB)']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#LINEAR REGRESSION ALGORITHM
from sklearn.linear_model import LinearRegression
lin_regn = LinearRegression()

#BUILDING A TRAINING MODEL
lin_regn.fit(x_train, y_train)
y_pred = lin_regn.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
mserr = mean_squared_error(y_test, y_pred)
print("Mean Squared Error(Linear Regression):", mserr)
rmserr = np.sqrt(mserr)
print("RMSE(Linear Regression):", rmserr)
r_squared = r2_score(y_test, y_pred)
print("R-squared(Linear Regression):", r_squared)
print("\n")
plt.scatter(y_test, y_test, c='r', label='Actual')
plt.scatter(y_test, y_pred, c='b', label='Predicted')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted(Linear Regression)")
plt.legend()
plt.show()

#DECISION TREE ALGORITHM
from sklearn.tree import DecisionTreeRegressor
desc_tree = DecisionTreeRegressor()

#BUILDING A TRAINING MODEL
desc_tree.fit(x_train, y_train)
y_pred = desc_tree.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
mserr = mean_squared_error(y_test, y_pred)
print("Mean Squared Error(Decision Tree Regression):", mserr)
rmserr = np.sqrt(mserr)
print("RMSE(Decision Tree Regression):", rmserr)
r_squared = r2_score(y_test, y_pred)
print("R-squared(Decision Tree Regression):", r_squared)
print("\n")
plt.scatter(y_test, y_test, c='r', label='Actual')
plt.scatter(y_test, y_pred, c='b', label='Predicted')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted(Decision Tree Regression)")
plt.legend()
plt.show()

#Random Forest ALGORITHM
from sklearn.ensemble import RandomForestRegressor
rand_forest = RandomForestRegressor()

#BUILDING A TRAINING MODEL
rand_forest.fit(x_train, y_train)
y_pred = rand_forest.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
mserr = mean_squared_error(y_test, y_pred)
print("Mean Squared Error(Random Forest Regression):", mserr)
rmserr = np.sqrt(mserr)
print("RMSE(Random Forest Regression):", rmserr)
r_squared = r2_score(y_test, y_pred)
print("R-squared(Random Forest Regression):", r_squared)
print("\n")
plt.scatter(y_test, y_test, c='r', label='Actual')
plt.scatter(y_test, y_pred, c='b', label='Predicted')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted(Random Forest Regression)")
plt.legend()
plt.show()

#LINEAR REGRESSION ALGORITHM
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()

#BUILDING A TRAINING MODEL
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
mserr = mean_squared_error(y_test, y_pred)
print("Mean Squared Error(KNN Regression):", mserr)
rmserr = np.sqrt(mserr)
print("RMSE(KNN Regression):", rmserr)
r_squared = r2_score(y_test, y_pred)
print("R-squared(KNN Regression):", r_squared)
print("\n")
data_table = pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})
print(data_table)
plt.scatter(y_test, y_test, c='r', label='Actual')
plt.scatter(y_test, y_pred, c='b', label='Predicted')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted(KNN Regression)")
plt.legend()
plt.show()

#GRADIENT BOOSTING ALGORITHM
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()

#BUILDING A TRAINING MODEL
gbr.fit(x_train, y_train)
y_pred = gbr.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
mserr = mean_squared_error(y_test, y_pred)
print("Mean Squared Error(Gradient Boosting Regression):", mserr)
rmserr = np.sqrt(mserr)
print("RMSE(Gradient Boosting Regression):", rmserr)
r_squared = r2_score(y_test, y_pred)
print("R-squared(Gradient Boosting Regression):", r_squared)
print("\n")
plt.scatter(y_test, y_test, c='r', label='Actual')
plt.scatter(y_test, y_pred, c='b', label='Predicted')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted(Gradient Boosting Regression)")
plt.legend()
plt.show()

#XGBOOST REGRESSION ALGORITHM
from xgboost import XGBRegressor
xgb = XGBRegressor()

#BUILDING A TRAINING MODEL
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
mserr = mean_squared_error(y_test, y_pred)
print("Mean Squared Error(XGBoost Regression):", mserr)
rmserr = np.sqrt(mserr)
print("RMSE(XGBoost Regression):", rmserr)
r_squared = r2_score(y_test, y_pred)
print("R-squared(XGBoost Regression):", r_squared)
print("\n")
plt.scatter(y_test, y_test, c='r', label='Actual')
plt.scatter(y_test, y_pred, c='b', label='Predicted')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted(XGBoost Regression)")
plt.legend()
plt.show()

#KMEANS REGRESSION ALGORITHM
from sklearn.cluster import KMeans
km_regn = KMeans(n_clusters = 1)

#BUILDING A TRAINING MODEL
km_regn.fit(x_train, y_train)
y_pred = km_regn.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
mserr = mean_squared_error(y_test, y_pred)
print("Mean Squared Error(KMeans Regression):", mserr)
rmserr = np.sqrt(mserr)
print("RMSE(KMeans Regression):", rmserr)
r_squared = r2_score(y_test, y_pred)
print("R-squared(KMeans Regression):", r_squared)
print("\n")
plt.scatter(y_test, y_test, c='r', label='Actual')
plt.scatter(y_test, y_pred, c='b', label='Predicted')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted(KMeans Regression)")
plt.legend()
plt.show()

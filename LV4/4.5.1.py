import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 


data = pd.read_csv('data_C02_emission (1).csv')

X = data[['Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)', 'Engine Size (L)', 'Cylinders']].to_numpy()
y = data['CO2 Emissions (g/km)'].to_numpy()

X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


plt.figure()
plt.scatter(X_train[:,0],y_train,color='b')
plt.scatter(X_test[:,0],y_test,color='r')
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()

sc= MinMaxScaler()
X_train_n=sc.fit_transform(X_train)
X_test_n=sc.transform(X_test)

plt.figure()
plt.subplot(1,2,1)
plt.hist(X_train[:,0],bins=20,color='b')
plt.subplot(1,2,2)
plt.hist(X_train_n[:,0],bins=20,color='r')
plt.show()

lr= LinearRegression()
lr.fit(X_train_n,y_train)
print(lr.coef_)


y_test_p= lr.predict(X_test_n)
plt.figure()
plt.scatter(y_test, y_test_p,color='b', alpha=0.4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

print(f'MSE: {mean_squared_error(y_test, y_test_p)}')
print(f'RMSE: {root_mean_squared_error(y_test, y_test_p)}')
print(f'MAE: {mean_absolute_error(y_test, y_test_p)}')
print(f'MAPE: {mean_absolute_percentage_error(y_test, y_test_p)}%')
print(f'R2: {r2_score(y_test, y_test_p)}')
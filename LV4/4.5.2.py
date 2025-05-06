import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn . preprocessing import OneHotEncoder
from sklearn.metrics import max_error
from sklearn.model_selection import train_test_split


data = pd.read_csv('data_C02_emission (1).csv')

ohe = OneHotEncoder()
X_encoded = pd.DataFrame(ohe.fit_transform(data[['Fuel Type']]).toarray())
data = data.join(X_encoded)

data.columns = ['Make','Model','Vehicle Class','Engine Size (L)','Cylinders','Transmission','Fuel Type','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','CO2 Emissions (g/km)','Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']

y = data['CO2 Emissions (g/km)'].copy()
X = data.drop('CO2 Emissions (g/km)', axis=1)

X_train_all, X_test_all, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train = X_train_all[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']]
X_test = X_test_all[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']]

linearModel = LinearRegression()
linearModel.fit(X_train, y_train)
y_test_p = linearModel.predict(X_test)

plt.scatter(X_test['Fuel Consumption City (L/100km)'], y_test, c='b', alpha=0.4)
plt.scatter(X_test['Fuel Consumption City (L/100km)'], y_test_p, c='r', alpha=0.4)
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()
plt.show()

max_Error = max_error(y_test, y_test_p)
print(f'Maksimalna pogreska {max_Error}')
print('Model vozila sa maksimalnom pogreskom: ', X_test_all[abs(y_test - y_test_p == max_Error)]['Model'].iloc[0])
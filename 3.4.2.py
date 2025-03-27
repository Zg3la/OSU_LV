import pandas as pd
import numpy as np
import matplotlib . pyplot as plt

data = pd.read_csv('data_C02_emission.csv')
#a
plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist')
plt.show()

#b
fuel_colors = {
    'D': 'black',  
    'X': 'red',  
    'Z': 'green',  
    'E': 'blue', 
    'Other': 'purple'  
}


plt.figure()

for fuel, color in fuel_colors.items():
    subset = data[data['Fuel Type'] == fuel]  
    
    if not subset.empty:  
        plt.scatter(subset['Fuel Consumption City (L/100km)'], subset['CO2 Emissions (g/km)'], 
                    label=fuel, color=color)

plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("Emissions (g/km)")
plt.legend(title="Fuel Type")
plt.show()

#c
data.boxplot(column='Fuel Consumption Hwy (L/100km)', by='Fuel Type')
plt.show()

#d
plt.figure()
data_by_fuel = data.groupby(by="Fuel Type").size()
data_by_fuel.plot(kind="bar")
plt.show()

#e 
plt.figure()
bycylinder= data.groupby('Cylinders')
bycylinder['CO2 Emissions (g/km)'].mean().plot(kind='bar')
plt.show()




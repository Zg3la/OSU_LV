import pandas as pd
import matplotlib . pyplot as plt

data = pd.read_csv('data_C02_emission.csv')
#a
print(f"Ukupno mjerenja: {len(data)}")
print(f"Data types: {data.dtypes} ")
print (data.isnull().sum())
data.dropna(axis=0)
data.dropna(axis=1)
duplicates= data.duplicated().any()
print("Duplikati?:",duplicates)
obj_cols= ["Make","Model","Vehicle Class","Transmission","Fuel Type"]
for column in obj_cols:
    data[column]=data[column].astype('category')

print(f"Data types: {data.dtypes} ")

#b
sorted_data = data.sort_values('Fuel Consumption City (L/100km)',ascending=False)
print(sorted_data[['Make','Model','Fuel Consumption City (L/100km)']].head(3))
print(sorted_data[['Make','Model','Fuel Consumption City (L/100km)']].tail(3))

#c
new_data = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
print(f"Ukupan broj auta sa zapreminom izmedu 2.5 i 3.5 litara: {len(new_data)}")
print(f"Prosjecna CO2 gramaza ovih vozila: {new_data['CO2 Emissions (g/km)'].mean()}")


#d
audis= data[(data['Make']=="Audi")]
print(f"Ukupan broj Audija: {len(audis)}")
audi4cly = audis[(audis['Cylinders']==4)]
print(f"Prosjecna CO2 gramaza Audija s 4 cilindra :{audi4cly['CO2 Emissions (g/km)'].mean()}")


#e
print(f"Broj vozila s 4 clindra {len(data[(data['Cylinders']==4)])}")
print(f"Broj vozila s 5 clindra {len(data[(data['Cylinders']==5)])}")
print(f"Broj vozila s 6 clindra {len(data[(data['Cylinders']==6)])}")
print(f"Broj vozila s 8 clindra {len(data[(data['Cylinders']==8)])}")
print(f"Broj vozila s 10 clindra {len(data[(data['Cylinders']==10)])}")
print(f"Broj vozila s 12 clindra {len(data[(data['Cylinders']==12)])}")
print(f"Broj vozila s 16 clindra {len(data[(data['Cylinders']==16)])}")

print(f"Prosjecna CO2 gramaza s 4 cilindra  {data[(data['Cylinders']==4)]['CO2 Emissions (g/km)'].mean()}")
print(f"Prosjecna CO2 gramaza s 5 cilindra  {data[(data['Cylinders']==5)]['CO2 Emissions (g/km)'].mean()}")
print(f"Prosjecna CO2 gramaza s 6 cilindra  {data[(data['Cylinders']==6)]['CO2 Emissions (g/km)'].mean()}")
print(f"Prosjecna CO2 gramaza s 8 cilindra  {data[(data['Cylinders']==8)]['CO2 Emissions (g/km)'].mean()}")
print(f"Prosjecna CO2 gramaza s 10 cilindra  {data[(data['Cylinders']==10)]['CO2 Emissions (g/km)'].mean()}")
print(f"Prosjecna CO2 gramaza s 12 cilindra  {data[(data['Cylinders']==12)]['CO2 Emissions (g/km)'].mean()}")
print(f"Prosjecna CO2 gramaza s 16 cilindra  {data[(data['Cylinders']==16)]['CO2 Emissions (g/km)'].mean()}")

#f

dizelasi = data[(data['Fuel Type']=='X')]
benzinci = data[(data['Fuel Type']=='Z')]

print(f"Prosjecna gradska potrosnja dizel auta: {dizelasi['Fuel Consumption City (L/100km)'].mean()}")
print(f"Prosjecna gradska potrosnja benzin auta: {benzinci['Fuel Consumption City (L/100km)'].mean()}")
print(f"Median za dizelase: {dizelasi['Fuel Consumption City (L/100km)'].median()}")
print(f"Median za benzince: {benzinci['Fuel Consumption City (L/100km)'].median()}")

#g
cilindra4 = data[(data['Cylinders']==4)]
dizelass4= cilindra4[(cilindra4['Fuel Type']=='X')]
index= dizelass4['Fuel Consumption City (L/100km)'].idxmax()
najveci_potrosac = dizelass4.loc[index]
print(najveci_potrosac[['Make','Model','Fuel Consumption City (L/100km)']])

#h
print(f"Broj auta s manualnim mjenjacem: {len(data[data['Transmission'].str.contains('M','AM')])}")

#i
print(data.corr(numeric_only=True))
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.csv",delimiter=",",skiprows=1)

duljina = len(data)
print(duljina)

height = data[:, 1]
weight = data[:, 2]

plt.scatter(height, weight, color="red")
plt.xlabel(height)
plt.ylabel(weight)
plt.show()


plt.scatter(height[::50],weight[::50],color="green")
plt.xlabel(height)
plt.ylabel(weight)
plt.show()


max= np.max(height)
min= np.min(height)
avg = np.average(height)

print(max)
print(min)
print(avg)

muski=[]
zenske=[]

for row in data:
    if row[0]==1.0:
        muski.append(row[1])
    else:
        zenske.append(row[1])

max_muski= np.max(muski)
max_zenski = np.max(zenske)
min_muski = np.min(muski)
min_zenske = np.min(zenske)
print(f"Najveca muska visina:{max_muski}")
print(f"Najveca muska visina:{max_zenski}")
print(f"Najmanja muska visina:{min_muski}")
print(f"Najmanja zenska visina:{min_zenske}")
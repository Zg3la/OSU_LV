import numpy as np
import matplotlib.pyplot as plt

x= np.array([1,2,3,3,1])
y= np.array([1,2,2,1,1])
plt.plot(x,y,'r',linewidth=1)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
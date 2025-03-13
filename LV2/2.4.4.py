import numpy as np
import matplotlib.pyplot as plt

black_square = np.zeros((50, 50), dtype=np.uint8)   
white_square = np.ones((50, 50), dtype=np.uint8) * 255  


top= np.hstack((white_square,black_square))
bottom= np.hstack((black_square,white_square))

chess=np.vstack((top,bottom))
print(chess)
plt.figure()
plt.imshow(chess,cmap="BuGn")
plt.show()
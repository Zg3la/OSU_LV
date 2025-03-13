import numpy as np
import matplotlib.pyplot as plt

img=plt.imread("road.jpg")
image_array = np.array(img)
brightened = np.clip(img + 50, 0, 255)

height, width, _ = image_array.shape
second_quarter = image_array[:, width // 4:width // 2]

rotated = np.rot90(image_array, k=-1)

mirrored = np.fliplr(image_array)

fig, axs = plt.subplots(1, 4, figsize=(15, 5))

axs[0].imshow(brightened)
axs[0].set_title("Posvijetljena")

axs[1].imshow(second_quarter)
axs[1].set_title("Druga četvrtina")

axs[2].imshow(rotated)
axs[2].set_title("Rotirano 90°")

axs[3].imshow(mirrored)
axs[3].set_title("Zrcaljeno")

for ax in axs:
    ax.axis("off")

plt.show()
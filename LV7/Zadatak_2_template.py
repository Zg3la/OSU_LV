import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\imgs\\test_1.jpg")
# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()
# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255
# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
# rezultatna slika
img_array_aprox = img_array.copy()
#7.5.2
#1
unique_colors= np.unique(img_array,axis=0)
print(f"Broj različitih boja: {len(unique_colors)}")
#2
km= KMeans(n_clusters=5,random_state=0,init='random')
km.fit(img_array_aprox)
centers= km.cluster_centers_
labels=km.predict(img_array_aprox)
j=km.inertia_ 
img_array_aprox=centers[labels]
for i in range(len(img_array_aprox)):
    img_array_aprox[i]= centers[labels[i]]

plt.figure()
plt.title("slika rekonstruirana")
plt.imshow(np.reshape(img_array_aprox,(w,h,d)))
plt.tight_layout()
plt.show()

#2 SLIKA
# ucitaj sliku
img = Image.imread("imgs\\imgs\\test_2.jpg")
# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()
# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255
# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
# rezultatna slika
img_array_aprox = img_array.copy()
#7.5.2
#1
unique_colors= np.unique(img_array,axis=0)
print(f"Broj različitih boja: {len(unique_colors)}")
#2
km= KMeans(n_clusters=7,random_state=0,init='random')
km.fit(img_array_aprox)
centers= km.cluster_centers_
labels=km.predict(img_array_aprox)
j=km.inertia_ 
img_array_aprox=centers[labels]
for i in range(len(img_array_aprox)):
    img_array_aprox[i]= centers[labels[i]]

plt.figure()
plt.title("slika rekonstruirana")
plt.imshow(np.reshape(img_array_aprox,(w,h,d)))
plt.tight_layout()
plt.show()


#3 SLIKA
# ucitaj sliku
img = Image.imread("imgs\\imgs\\test_3.jpg")
# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()
# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255
# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
# rezultatna slika
img_array_aprox = img_array.copy()
#7.5.2
#1
unique_colors= np.unique(img_array,axis=0)
print(f"Broj različitih boja: {len(unique_colors)}")
#2
km= KMeans(n_clusters=5,random_state=0,init='random')
km.fit(img_array_aprox)
centers= km.cluster_centers_
labels=km.predict(img_array_aprox)
j=km.inertia_ 
img_array_aprox=centers[labels]
for i in range(len(img_array_aprox)):
    img_array_aprox[i]= centers[labels[i]]

plt.figure()
plt.title("slika rekonstruirana")
plt.imshow(np.reshape(img_array_aprox,(w,h,d)))
plt.tight_layout()
plt.show()

#4 SLIKA
# ucitaj sliku
img = Image.imread("imgs\\imgs\\test_4.jpg")
# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()
# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255
# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
# rezultatna slika
img_array_aprox = img_array.copy()
#7.5.2
#1
unique_colors= np.unique(img_array,axis=0)
print(f"Broj različitih boja: {len(unique_colors)}")
#2
km= KMeans(n_clusters=6,random_state=0,init='random')
km.fit(img_array_aprox)
centers= km.cluster_centers_
labels=km.predict(img_array_aprox)
j=km.inertia_ 
img_array_aprox=centers[labels]
for i in range(len(img_array_aprox)):
    img_array_aprox[i]= centers[labels[i]]

plt.figure()
plt.title("slika rekonstruirana")
plt.imshow(np.reshape(img_array_aprox,(w,h,d)))
plt.tight_layout()
plt.show()


#5 SLIKA
# ucitaj sliku
img = Image.imread("imgs\\imgs\\test_5.jpg")
# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()
# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255
# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
# rezultatna slika
img_array_aprox = img_array.copy()
#7.5.2
#1
unique_colors= np.unique(img_array,axis=0)
print(f"Broj različitih boja: {len(unique_colors)}")
#2
km= KMeans(n_clusters=8,random_state=0,init='random')
km.fit(img_array_aprox)
centers= km.cluster_centers_
labels=km.predict(img_array_aprox)
j=km.inertia_ 
img_array_aprox=centers[labels]
for i in range(len(img_array_aprox)):
    img_array_aprox[i]= centers[labels[i]]

plt.figure()
plt.title("slika rekonstruirana")
plt.imshow(np.reshape(img_array_aprox,(w,h,d)))
plt.tight_layout()
plt.show()


#6 SLIKA
# ucitaj sliku
img = Image.imread("imgs\\imgs\\test_6.jpg")
# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()
# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255
# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
# rezultatna slika
img_array_aprox = img_array.copy()
unique_colors= np.unique(img_array,axis=0)
print(f"Broj različitih boja: {len(unique_colors)}")
km= KMeans(n_clusters=9,random_state=0,init='random')
km.fit(img_array_aprox)
centers= km.cluster_centers_
labels=km.predict(img_array_aprox)
j=km.inertia_ 
img_array_aprox=centers[labels]
for i in range(len(img_array_aprox)):
    img_array_aprox[i]= centers[labels[i]]

plt.figure()
plt.title("slika rekonstruirana")
plt.imshow(np.reshape(img_array_aprox,(w,h,d)))
plt.tight_layout()
plt.show()


#2.6
inercije=[]
k_values= list(range(2,20))
for k in k_values:
    kmeans= KMeans(n_clusters=k,init='random',random_state=0)
    kmeans.fit(img_array)
    inercije.append(kmeans.inertia_)

plt.figure()
plt.plot(k_values,inercije)
plt.title("Lakat za 6 sliku")
plt.tight_layout()
plt.show()







img = Image.imread("imgs\\imgs\\test_3.jpg")
img = img.astype(np.float64) / 255
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
img_array_aprox = img_array.copy()
km.fit(img_array_aprox)
centers= km.cluster_centers_
labels=km.predict(img_array_aprox)
j=km.inertia_ 
target = 4
binary_img_array = np.ones_like(img_array_aprox)  
for i in range(len(labels)):
    if labels[i] == target:
        binary_img_array[i] = [0, 0, 0]  
    else:
        binary_img_array[i] = [1, 1, 1]  

plt.figure()
plt.title(f"Binarna slika grupa {target}")
plt.imshow(np.reshape(binary_img_array, (w, h, d)))
plt.tight_layout()
plt.show()





import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X = generate_data(500,5)
X1 = generate_data(500,1)
X2 = generate_data(500,2)
X3 = generate_data(500,3)
X4 = generate_data(500,4)
X5 = generate_data(500,5)


# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()


#7.5.1
#Kada je flag=1, na grafu imaju 3 grupe. Kada je flag=2, imaju 3 grupe. 
#Kada je flag=3, rekao bih da imaju 4 grupe. Kada je flag=4, imaju 2 grupe.
#Naposljetku kada je flag=5, imaju 2 grupe.
#7.5.2 i 7.5.3
km= KMeans(n_clusters=3,init='random',n_init=10,random_state=0)
km.fit(X1)
labels=km.predict(X1)
plt.figure()
plt.scatter(X1[:,0],X1[:,1],c=labels)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('K=3')
plt.show()

km.fit(X2)
labels=km.predict(X2)
plt.figure()
plt.scatter(X2[:,0],X2[:,1],c=labels)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('K=3')
plt.show()

km= KMeans(n_clusters=4,init='random',n_init=10,random_state=0)
km.fit(X3)
labels=km.predict(X3)
plt.figure()
plt.scatter(X3[:,0],X3[:,1],c=labels)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('K=4')
plt.show()

km= KMeans(n_clusters=2,init='random',n_init=50,random_state=0)
km.fit(X4)
labels=km.predict(X4)
plt.figure()
plt.scatter(X4[:,0],X4[:,1],c=labels)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('K=2, mijenjan init')
plt.show()

km.fit(X5)
labels=km.predict(X5)
plt.figure()
plt.scatter(X5[:,0],X5[:,1],c=labels)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('K=2, mijenjan init')
plt.show()

#Grupe se mijenjaju u ovisnosti o K ali i n_init odnosno broju ponavljanja.
#Ono što je čudno je da algoritam dijeli grupe da drukčiji način na koji bih ja.

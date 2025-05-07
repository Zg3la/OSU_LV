import numpy as np
import matplotlib.pyplot as plt
from sklearn . linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn . metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn.metrics import classification_report


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='Reds', edgecolor='k')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='Grays', edgecolor='k',marker='x' )
plt.show()

lr= LogisticRegression()
lr.fit(X_train,y_train)


t0=lr.intercept_[0]
t1,t2=lr.coef_[0]
x=np.linspace(X[:,0].min(),X[:,0].max(),100)
y= -(t0+t1*x)/t2
plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='Reds', edgecolor='k')
plt.plot(x,y)
plt.show()


y_test_p=lr.predict(X_test)
cm=ConfusionMatrixDisplay(confusion_matrix(y_test,y_test_p))
cm.plot()
plt.show()
print(classification_report(y_test , y_test_p))


correct=(y_test==y_test_p)
false=(y_test!=y_test_p)

false_values=[]

for i in range(len(y_test)):
    if y_test[i]!=y_test_p[i]:
        false_values.append([X_test[i,0],X_test[i,1]])

false_values=np.array(false_values)

plt.scatter(X_test[:,0],X_test[:,1],c='green')
plt.scatter(false_values[:,0],false_values[:,1],c='black')
plt.show()
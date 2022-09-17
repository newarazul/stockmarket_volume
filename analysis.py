import numpy as np
import pandas as pd
import csv


import matplotlib.pyplot as plt

import sklearn 
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
#from mp1_toolkits.mplot3d import Axes3D
#from sklearn.preprocessing import scale

#import sklearn.metrics as sm
#from sklearn import datasets 
#from sklearn.metrics import confusion_matrix,classification_report

#matplotlib inline
#rcParams['figure.figsize'] = 7,4

#iris=datasets.load_iris()
#X=scale(iris.data)



data=pd.read_csv("MarketData.csv")





#create the data set
X,y =make_blobs(n_samples=3000, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

print(type(X))
print(X)
#plt.scatter(X[:,0], X[:,1], c="white", marker="o", edgecolor="black",s=50)
#plt.show()

km=KMeans(n_clusters=3, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km=km.fit_predict(X)


plt.scatter(X[y_km==0,0], X[y_km==0,1], s=50, c="lightgreen",marker="s", edgecolor="black", label="cluster 1")
plt.scatter(X[y_km==1,0], X[y_km==1,1], s=50, c="orange",marker="o", edgecolor="black", label="cluster 2")
plt.scatter(X[y_km==2,0], X[y_km==2,1], s=50, c="lightblue",marker="v", edgecolor="black", label="cluster 3")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], s=250, c="red",marker="*", edgecolor="black", label="centers")



plt.legend(scatterpoints=1)
plt.grid()
plt.show()


#distortions=[]
#for i in range(1,11):
#    km=KMeans(n_clusters=i, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=0)
#    km.fit(X)
#    distortions.append(km.inertia_)


#plt.plot(range(1,11), distortions, marker="o")
#plt.xlabel("Number of clusters")
#plt.ylabel("Distortion")
#plt.show()



#X=np.array([data.Open])
#centers,indices = kmeans_plusplus(X, n_clusters=2, random_state=0)

#plot.figure(1)

#plt.scatter(centers

#overview=pd.DataFrame(data)
#plt.figure();
#overview.plot();



#print(data)
#print(data.Open)


#here the fractional values are calculated and stored in lists
#fractional_change=[]
#fractional_close=[]
#fractional_volume=[]

#for value in data.Open:
#    fractional_open=[(y/x) for x,y in zip(data.Open, data.Open[1:])]
#    print(fractional_open)
#    fractional_close=[(y/x) for x,y in zip(data.Open, data.Close)]
#    fractional_volume=[(y/x) for x,y in zip(data.Volume, data.Volume[1:])]



#Open = pd.Series(data=fractional_volume, index=data.Date[1:])
#Open.plot()
#plt.show()

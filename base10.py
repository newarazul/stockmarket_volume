import numpy as np
import pandas as pd
import csv
import statistics


import matplotlib.pyplot as plt

import sklearn 
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin
from statistics import mean
#from mp1_toolkits.mplot3d import Axes3D
#from sklearn.preprocessing import scale

#import sklearn.metrics as sm
#from sklearn import datasets 
#from sklearn.metrics import confusion_matrix,classification_report

#matplotlib inline
#rcParams['figure.figsize'] = 7,4

#iris=datasets.load_iris()
#X=scale(iris.data)
def day30gain(data):
    change = []
    Z=data.Open.values
    for i in Z:
        change.append(Z[i+90]/Z[i])
        print(change)
    return change

def findnumber(X):
    distortions=[]
    for i in range (1,11):
        km=KMeans(n_clusters=i, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)
    
    return distortions 
#    plt.plot(range(1,11),distortions,marker="o")
#    plt.xlabel("Number of clusters")
#    plt.ylabel("Distortion")
#    plot.show()
    


def find_clusters(X, n_clusters, rseed=2):
    # Randomly chose clusters
    rng=np.random.RandomState(rseed)
    i=rng.permutation(X.shape[0])[:n_clusters]
    centers=X[i]

    while True:
        #assign labels based on closest center
        labels=pairwise_distances_argmin(X, centers)
        
        #find new centers from means of points
        new_centers=np.array([X[labels==i].mean(0) for i in range(n_clusters)])

        #check for convergence
        if np.all(centers==new_centers):
            break
        centers=new_centers
       
    return centers,labels       


def normalize(data):
    X=[]
    Y=[]
    C=data.Volume
    for index, row in data.iterrows():
        if index>=90:
            D=C[index-90:index]
            median=mean(D)
            X.append(index)
            Y.append(row["Volume"]/median)

    return X,Y



#X=[]
#y=[]






data=pd.read_csv("MarketData.csv")
result = []
Open1=data.Open.values
Open2=Open1[91:]
Open1=Open1[1:len(Open1)-90]
#print(Open1)
#print(Open2)
for val1, val2 in zip(Open1, Open2):
    result.append((val1/val2)-1)
#print(result)


averagegain=(Open2[-1]/Open2[1])/len(Open2)
print("average_gain", averagegain)




result=result[90:]
print(len(result))

#for index,value in enumerate(Z):
#    change.append(
#print(day30gain(data))
#index=data.index.values
#print(index)
#X=[]
#for index, row in data.iterrows():
#    print(index,row["Volume"])
#    X.append([[index,row]])

#numpy=data["Volume"].index.to_numpy()
X,Y=normalize(data)
#X=data.index.values
#print(type(X))
#print(X)
#Y=data["Volume"].values
#print(Y)
Y=Y[:len(Y)-91]
print(type(Y))
print(len(Y))
#C=np.vstack((X,Y)).T
#print(C)
#distortions=findnumber(C)
C=np.vstack((Y,result)).T
print(C)
print(len(C))
distortions=findnumber(C)
plt.plot(range(1,11),distortions,marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")
plt.show()
    
#X=data["Volume"].index.to_numpy()
#Y=data["Volume"].values
#C=np.array(X,Y)


#X=[[data["Volume"].index.values,data["Volume"].values]]
#X=X.values
#print(type(C))
#print(C)
centers, labels=find_clusters(C,5)
print(centers)
plt.scatter(C[:,0], C[:,1], c=labels, s=10, cmap="viridis")
plt.xlabel("Fractional Volume increase")
plt.ylabel("Fractional 90 day return")

plt.show()
#plt.savefig("S&P.png")


#def normalie90dayreturn:
    





#create the data set
#X,y =make_blobs(n_samples=3000, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
#print(y)

#print(data.Volume)
#X=data.Volume
#print(X)
#y=data.Date.tolist()
#plt.scatter(X[:,0], X[:,1], c="white", marker="o", edgecolor="black",s=50)
#plt.show()

#km=KMeans(n_clusters=3, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=0)
#y_km=km.fit_predict(X)


#plt.scatter(X[y_km==0,0], X[y_km==0,1], s=50, c="lightgreen",marker="s", edgecolor="black", label="cluster 1")
#plt.scatter(X[y_km==1,0], X[y_km==1,1], s=50, c="orange",marker="o", edgecolor="black", label="cluster 2")
#plt.scatter(X[y_km==2,0], X[y_km==2,1], s=50, c="lightblue",marker="v", edgecolor="black", label="cluster 3")
#plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], s=250, c="red",marker="*", edgecolor="black", label="centers")



#plt.legend(scatterpoints=1)
#plt.grid()
#plt.show()


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

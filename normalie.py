import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import statistics
from statistics import mean
import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin
from statistics import mean



data=pd.read_csv("MarketData.csv")


def printvolumedata(data):
    for index, row in data.iterrows:
        print(index,row["Volume"])
        print(row["Volume"].iloc[i:])

X=[]
Y=[]
#C=data.Volume
#print(mean(C))
#median=mean(C)
#for index, row in data.iterrows():
#    if index >= 300:
#        D=C[index-300:index]
#        median=mean(D)
#        print(index,row["Volume"]/median)
#        X.append(index)
#        Y.append(row["Volume"]/median)
    
#print(len(X))
#print(len(Y))
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

def findnumber(X):
    D=np.vstack((Y,X)).T
    distortions=[]
    for i in range (1,11):
        km=KMeans(n_clusters=i, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=0)
        km.fit(D)
        distortions.append(km.inertia_)

    return distortions


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

def plotkpoints(distortions):
    distortions=findnumber(Y)
    plt.plot(range(1,11),distortions,marker="o")
    plt.show()


def dayreturn(data):
    Open2=[]
    Open1=[]
    change=[]
    volume=[]
    volume_mean=[]
    volume_last90=[]
    volume_out=[]
    Open1=data.Open.values[91:-90]
    Open2=data.Open.values[181:]
    volume=data.Volume[1:-90]
    for i in range(1,len(volume)):
        for s in range(0,90):
            print(volume[s])
            volume_last90.append(volume[i+s])
        volume_mean.append(mean(volume_last90))        
        print(mean(volume_last90))
    volume=volume[91:]
    volume_mean=volume_mean[:-90]
    for d,t in zip(volume,volume_mean):    
        volume_out.append(d/t)
    for val1,val2 in zip(Open1, Open2):
            change.append((Open2/Open1)-1)
#            print(change)
    return change,volume_out





X,Y=dayreturn(data)
centers=[]
labels=[]
plotkpoints(findnumber(X))
D=np.vstack((Y,X)).T
centers, labels=find_clusters(D,4)
print(centers)


plt.scatter(D[:,0], D[:,1], c=labels, s=10, cmap="viridis")
plt.xlabel("Volume of the day vs average 90 day volume")
plt.ylabel("90 days change in open")
#plt.scatter(X,Y)
plt.show()





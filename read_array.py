import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv("MarketData.csv")
print(data.Volume)
#Volumes=data.Volume
#Array=Volumes.values
#print(Array)
#y=data.Date
#y=y.values
#print(y)
#c=np.concatenate((Array,y),axis=0)
#print(c)


Open=pd.Series(data.Volume)
Open.plot()
plt.show()
#plt.scatter(Array,y)
#plt.show()

#CSVData=open("MarketData.csv")

#Array2d_result=np.loadtxt(CSVData, delimiter=",")

#print(Array2d_result)


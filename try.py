import pandas as pd

poke=pd.read_csv("MarketData.csv")
print(poke.columns)
print(poke[["Date","Volume"]])

data=poke[["Date","Volume"]].values
print(data)


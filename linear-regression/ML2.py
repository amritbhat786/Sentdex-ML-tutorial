import pandas as pd 
import quandl
import math

df =  quandl.get('WIKI/GOOGL')	
print(type(df))
#print(df.head())
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
high = df['Adj. High']
close = df['Adj. Close']
Open = df['Adj. Open'] 
#print(df)	

df['hl_pct'] = (high - close) / close 
df['pct_change'] = (close - Open) / Open 

df = df[['Adj. Close','hl_pct','pct_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999,inplace = True) #NA/NAN(Pandas): Not available/Not a Number

print(len(df))
forecast_out = int(math.ceil(0.005*len(df)))


df['label'] = df[forecast_col].shift(-forecast_out)
df. dropna(inplace = True)
print(df.tail())
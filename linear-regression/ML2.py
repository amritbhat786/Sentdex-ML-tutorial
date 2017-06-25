import pandas as pd 
import quandl

df =  quandl.get('WIKI/GOOGL')	

#print(df.head())
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
high = df['Adj. High']
close = df['Adj. Close']
Open = df['Adj. Open'] 
print( df )
df['hl_pct'] = {high - close} / close 
df['pct_change'] = {close - Open} / Open 

df = df[['Adj. Close','hl_pct','pct_change','Adj. Volume']]
print(df.head())


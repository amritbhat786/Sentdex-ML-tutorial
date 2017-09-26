import pandas as pd 
import quandl,math
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression

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
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)


df['label'] = df[forecast_col].shift(-forecast_out)
df. dropna(inplace = True)
#print(df.tail())

X = np.array(df.drop(['label'],1))
y = np.array(df['label']) 

X = preprocessing.scale(X)

#X = X[:-forecast_out+1]
#df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = LinearRegression(n_jobs = -1)
#clf = svm.SVR(kernel = 'poly')
clf.fit(X_train,y_train) # Train the classifier

accuracy = clf.score(X_test,y_test)	 #Test the classifier
print(df.tail())
print(accuracy)
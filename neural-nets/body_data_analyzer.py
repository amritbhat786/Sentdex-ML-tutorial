import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#Download the data from website 
df_all =  pd.read_csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv')
num_rows = df_all.shape[0]
print(num_rows) 


#Clean the data
#Count number of missing elements (NaN) in each coloumn

count_nan = df_all.isnull().sum()
count_wo_nan = count_nan[count_nan==0] 	
#remove the coloumns with missing elements 
df_all = df_all[count_wo_nan.keys()]

#Remove the first 7 coloumns which contain no discriminative information
df_all = df_all.ix[:,7:]


x = df_all.ix[:,:-1].values
standard_scalar = StandardScaler()
x_std = standard_scalar.fit_transform(x)

#t distributed stochastic neighbouring embedding (t-SNE) visualization
tsne = 	TSNE(n_components=2,random_state = 0)
x_test_2d = tsne.fit_transform(x_std)

#scatter plot the sample points among 5 classes 
markers = ('s','d','o','^','v')
color_map = {0:'red',1:'blue',2:'lightgreen',3:'purple',4:'cyan'}
plt.figure()
for idx,cl in enumerate(np.unique(x_test_2d)):
	plt.scatter(x=x_test_2d[cl,0],y = x_test_2d[cl,1],c=color_map[idx],marker = markers[idx],label =cl)

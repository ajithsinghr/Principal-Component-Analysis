# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 17:11:21 2022

@author: ramav
"""
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#===============================================================================
#===============================================================================

# importing csv file
df = pd.read_csv("D:\\Assignments\\PCA\\wine.csv")

df.head()
df.shape
df.isnull().sum()
df.describe()

#===============================================================================
#===============================================================================

df["Type"].value_counts().plot(kind="bar")

df = df.drop("Type",axis=1)

x = df.iloc[:,0:]
x.shape
x.head()

# Data transformation

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(x)

scaled_x = ss.fit_transform(x)

# Applying Principal component ananlysis(PCA) for the given data.

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(scaled_x)

x_pca = pca.fit_transform(scaled_x)
x_pca = pd.DataFrame(x_pca,columns= ["Pc1","Pc2","Pc3","Pc4","Pc5","Pc6","Pc7","Pc8","Pc9","Pc10","Pc11","Pc12"
                                     ,"Pc13"])
x_pca.head()
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)

x_pca.shape
x_pca.var()

x_pca["Pc1"].var()
x_pca["Pc2"].var()
x_pca["Pc3"].var()
x_pca["Pc4"].var()

# cummulative variance
df1 = pd.DataFrame({'var':pca.explained_variance_ratio_,
                  'x_pca':["Pc1","Pc2","Pc3","Pc4","Pc5","Pc6","Pc7","Pc8","Pc9","Pc10","Pc11","Pc12"
                                                       ,"Pc13"]})
sns.barplot(x='x_pca',y="var", data=df1, color="c");
plt.plot(pca.explained_variance_ratio_)

'''
# cummulative variance
var = np.cumsum(np.round(pca.explained_variance_ratio_,5)*100)

plt.plot(var, color="red")
'''

# updated data
# selecting only first three columns after PCA which contains more information for predicting

ss_x = x_pca.drop(x_pca.iloc[:,3:],axis=1)
ss_x.head()

sns.scatterplot(data=ss_x)
sns.scatterplot(data=ss_x,x='Pc1',y='Pc2')
sns.scatterplot(data=ss_x,x='Pc2',y='Pc3')


# hireachial / aglomerativve clustering
# single linkage method
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10,7))
plt.title("dendogram")
dend = shc.dendrogram(shc.linkage(ss_x,method="single"))


from sklearn.cluster import AgglomerativeClustering
agc1 = AgglomerativeClustering(n_clusters=3,linkage="single",affinity="euclidean")
y = agc1.fit_predict(ss_x)
y = pd.DataFrame(y,columns=["clusters"])
y.value_counts() # using single linkage more no of data formed into one cluster only
#=======================

#Complete linkage method

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10,7))
plt.title("dendogram")
dend = shc.dendrogram(shc.linkage(ss_x,method="complete"))


from sklearn.cluster import AgglomerativeClustering
agc2 = AgglomerativeClustering(n_clusters=4,linkage="complete",affinity="euclidean")
y = agc2.fit_predict(ss_x)
y = pd.DataFrame(y,columns=["clusters"])
y.value_counts()

#=====================
# ward linkage method

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(ss_x, method='ward')) 

from sklearn.cluster import AgglomerativeClustering
agc3 = AgglomerativeClustering(n_clusters=2,linkage="ward",affinity="euclidean")
y = agc3.fit_predict(ss_x)
y = pd.DataFrame(y,columns=["clusters"])
y.value_counts()

agc2.labels_
winedata = df.copy()
winedata["cluster_no"] = agc2.labels_
winedata.cluster_no.value_counts().plot(kind="bar")

'''
for this wine data complete linkage is more familiar for good results as it forming nearly equal 
no of clusters comparing to single and ward linkage method
 
'''

##################=======================================########################

# K  means clustering

from sklearn.cluster import KMeans
kmeans = KMeans()
kmeans.fit(ss_x)

l1 = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,n_init=20)
    kmeans.fit(ss_x)
    l1.append(kmeans.inertia_)
    
print(l1)

# EDA 
pd.DataFrame(range(1,11))        
pd.DataFrame(l1)
    
pd.concat([pd.DataFrame(range(1,11)),pd.DataFrame(l1)], axis=1)

plt.scatter(range(1,11),l1)
plt.show()    

# elbow plot(used to find optimium no of clusters)

plt.plot(range(1,11),l1)
plt.title("elbow graph")
plt.xlabel("k value")
plt.ylabel("wcss value")
plt.show()

# elbow plot(used to find optimium no of clusters)
#pip install yellowbrick
from yellowbrick.cluster import KElbowVisualizer

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans,k=(1,11))
elbow.fit(ss_x)
elbow.poof()
plt.show()

# final model (building clusters by using k = 3)

from sklearn.cluster import KMeans
Kmeans = KMeans(n_clusters=3,n_init=20)
Kmeans.fit(ss_x)

kclus = kmeans.inertia_
# Getting the cluster centers
C = kmeans.cluster_centers_


winedata2 = df.copy()
winedata2["cluster_name"]=Kmeans.labels_
winedata2.cluster_name.value_counts().plot(kind="bar")




























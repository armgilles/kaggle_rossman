# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:21:11 2015

@author: babou
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg") #Needed to save figures
import matplotlib.pyplot as plt


# Gather some features
def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
#    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly

    # add some more with a bit of preprocessing
#    features.append('SchoolHoliday')
#    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)

    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    
    PromoInterval_dict = {0 : 0, 'Jan,Apr,Jul,Oct' : 1, 'Feb,May,Aug,Nov' : 2, 'Mar,Jun,Sept,Dec' : 3}
    store['PromoInterval'] = store.PromoInterval.map(PromoInterval_dict)
#    features.extend(['DayOfWeek', 'month', 'day', 'year'])
#    data['year'] = data.Date.dt.year
#    data['month'] = data.Date.dt.month
#    data['day'] = data.Date.dt.day
#    data['DayOfWeek'] = data.Date.dt.dayofweek
    return data
    
    
def clustering(data, n_components):
    features = data.columns[1:] # No store id
    X = data[features].values
    pca = PCA(n_components=n_components).fit(X)
    X_pca = pca.transform(X)
    print "Explained variance sum : "  + str(pca.explained_variance_ratio_.sum())
    return (X_pca, pca)
     



types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(float),
         'PromoInterval': np.dtype(str)}
train = pd.read_csv("input/train.csv", parse_dates=[2], dtype=types)
test = pd.read_csv("input/test.csv", parse_dates=[3], dtype=types)
store = pd.read_csv("input/store.csv")

print("Assume store open, if not provided")
train.fillna(1, inplace=True)
test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")
train = train[train["Open"] != 0]
print("Use only Sales bigger then zero. Simplifies calculation of rmspe")
train = train[train["Sales"] > 0]

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("augment features train")
store = build_features([], store)
print(features)

print "OK"


customers_store = train.groupby('Store')['Customers'].agg({'mean' : np.mean,
                                                           'median' : np.median,
                                                           'std': np.std}).reset_index()

 

print("Create features with Customers")
store_enrich = store.merge(customers_store, on='Store')

print("Standardization")
sc = StandardScaler()
data_sc = sc.fit_transform(store_enrich)
store_enrich_sc = pd.DataFrame(data_sc, columns=store_enrich.columns)

print "PCA cacul" 
#X_pca, pca = clustering(store, 2)
X_pca, pca = clustering(store_enrich_sc, 6)
#plt.scatter(X_pca[:, 0], X_pca[:, 1], marker="o", alpha=0.4)


print "Kmean"
k_means = KMeans(init='k-means++', n_clusters=6, n_init=10).fit(X_pca)

k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)

data = pd.DataFrame({'pca_1' : X_pca[:,0],
                     'pca_2' : X_pca[:,1],
                     'cluster' : pd.Series(k_means_labels),
                     'Store' : pd.Series(store_enrich.Store)})
                     

dico_color = {0: '#8b0000',
              1: '#d84765',
              2: '#fea0ac',
              3: '#ffffe0',
              4: '#9edba4',
              5: '#5aaf8c',
              6: '#008080',
#              7: '#c7f0ba',
#              8: '#9edba4',
#              9: '#7ac696',
#              10: '#5aaf8c',
#              11: '#399785',
#              12: '#008080'
              }
              

center_color = [col for col in dico_color.values()]

color = [col for col in dico_color.values()]

data.color = data.cluster.map(dico_color)

for k, col in zip(range(k_means_labels_unique.argmax() + 1), color):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.scatter(data.pca_1, data.pca_2, color=data.color,
             marker='.', alpha=0.6)
    plt.plot(cluster_center[0], cluster_center[1], 
         '*', markerfacecolor=col, markersize=15)
                     
plt.title('%d types de store' % k_means.n_clusters)
plt.xlabel('First PCA direction')
plt.ylabel('Seconde PCA direction')

data[['Store', 'cluster']].to_csv("cluster_kmean.csv", index=False)




 

     
     
 
 

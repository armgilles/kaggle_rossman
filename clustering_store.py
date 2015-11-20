# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:21:11 2015

@author: babou
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn import metrics
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


print "DBSCAN"
db = DBSCAN(eps=2, min_samples=100, metric='euclidean').fit(X_pca)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
unique_labels = set(labels)
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

data = pd.DataFrame({'pca_1' : X_pca[:,0],
                     'pca_2' : X_pca[:,1],
                     'cluster' : pd.Series(labels),
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

#for k, col in zip(range(k_means_labels_unique.argmax() + 1), color):
#    my_members = k_means_labels == k
#    cluster_center = k_means_cluster_centers[k]
#    plt.scatter(data.pca_1, data.pca_2, color=data.color,
#             marker='.', alpha=0.6)
#    plt.plot(cluster_center[0], cluster_center[1], 
#         '*', markerfacecolor=col, markersize=15)

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X_pca, labels))

         
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
    class_member_mask = (labels == k)    
    xy= X_pca[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], xy, 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14, alpha=0.4)
    xy = X_pca[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6, alpha=0.3)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
                     
#plt.title('%d types de store' % k_means.n_clusters)
plt.xlabel('First PCA direction')
plt.ylabel('Seconde PCA direction')

data[['Store', 'cluster']].to_csv("cluster_DBCAN.csv", index=False)




 

     
     
 
 

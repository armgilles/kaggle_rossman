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



# Gather some features for Train / Test
def build_features(data):
    # remove NaNs
    data.fillna(0, inplace=True)
#    data.loc[data.Open.isnull(), 'Open'] = 1  We need Open = 0 for our groupby
    
    # add some more with a bit of preprocessing
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)

    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.weekofyear
    
    # CompetionOpen en PromoOpen from https://www.kaggle.com/ananya77041/rossmann-store-sales/randomforestpython/code
    # Calculate time competition open time in months
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
                            (data.Month - data.CompetitionOpenSinceMonth)
    
    # Promo open time in months
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
                        (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    # If No Promo2SinceYear = 0, PromoOpen =0
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0
    
    # Indicate that sales on that day are in promo interval
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Okt', 11:'Nov', 12:'Dec'}
    data['monthStr'] = data.Month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1
    
    # Transform PromoInterval 
    PromoInterval_dict = {'' : 0, 'Jan,Apr,Jul,Oct' : 1, 'Feb,May,Aug,Nov' : 2, 'Mar,Jun,Sept,Dec' : 3}
    data['PromoInterval'] = data.PromoInterval.map(PromoInterval_dict)
    
    return data


# Gather some features for Stores
def build_features_store(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)

    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    
    PromoInterval_dict = {0 : 0, 'Jan,Apr,Jul,Oct' : 1, 'Feb,May,Aug,Nov' : 2, 'Mar,Jun,Sept,Dec' : 3}
    store['PromoInterval'] = store.PromoInterval.map(PromoInterval_dict)
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

## We keep "Open" = 0 for our groupby operation
#print("Consider only open stores for training. Closed stores wont count into the score.")
#train = train[train["Open"] != 0]
#print("Use only Sales bigger then zero. Simplifies calculation of rmspe")
#train = train[train["Sales"] > 0]

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')


train = build_features(train)

features = []

print("augment features train")
store = build_features_store([], store)
print(features)

print "OK"


cust_sales_store = train.groupby('Store')['Customers', 'Sales'].mean().reset_index()
open_store = train.groupby('Store')['Open'].sum().reset_index()

 

print("Create features with Customers")
store_enrich = store.merge(cust_sales_store, on='Store')
store_enrich = store_enrich.merge(open_store, on='Store')

print("Standardization")
sc = StandardScaler()
data_sc = sc.fit_transform(store_enrich.ix[:, 1:])
store_enrich_sc = pd.DataFrame(data_sc, columns=store_enrich.columns[1:])
store_enrich_sc['Store'] = store.Store

print "PCA cacul" 

pca_features = store_enrich_sc.columns.tolist()
pca_features.remove('Store')
#X_pca, pca = clustering(store, 2)
X_pca, pca = clustering(store_enrich_sc[pca_features], 8)
#plt.scatter(X_pca[:, 0], X_pca[:, 1], marker="o", alpha=0.4)


print "Kmean"
k_means = KMeans(init='k-means++', n_clusters=5, n_init=10).fit(X_pca)

k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)

data = pd.DataFrame({'pca_1' : X_pca[:,0],
                     'pca_2' : X_pca[:,1],
                     'cluster' : pd.Series(k_means_labels),
                     'Store' : pd.Series(store_enrich.Store)})
                     

dico_color = {0: '#7cfc00',
              1: '#d2c62d',
              2: '#e99449',
              3: '#e56451',
              4: '#d13a48',
              5: '#b2152f',
              6: '#8b0000',
#              7: '#c7f0ba',
#              8: '#9edba4',
#              9: '#7ac696',
#              10: '#5aaf8c',
#              11: '#399785',
#              12: '#008080'
              }
# http://gka.github.io/palettes/#colors=SpringGreen,SkyBlue,Orange|steps=6|bez=0|coL=0             
# '#7cfc00','#d2c62d','#e99449','#e56451','#d13a48','#b2152f','#8b0000'

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
                     
plt.title("Number of Store's cluster %d" % k_means.n_clusters)
plt.xlabel('First PCA direction (%s)' % round(pca.explained_variance_ratio_[0], 2))
plt.ylabel('Seconde PCA direction (%s)' % round(pca.explained_variance_ratio_[1], 2))

data[['Store', 'cluster']].to_csv("cluster_kmean.csv", index=False)




 

     
     
 
 

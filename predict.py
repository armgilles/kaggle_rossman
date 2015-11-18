# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:48:04 2015

@author: babou
"""

import pandas as pd
import numpy as np

import sklearn.cross_validation as cv
import sklearn.feature_extraction as fe
import sklearn.preprocessing as preprocessing
import sklearn.ensemble as es



store = pd.read_csv('input/store.csv')
train = pd.read_csv('input/train.csv',low_memory=False)
test = pd.read_csv('input/test.csv')

print("Reading CSV Done.")


df_train = train.copy()
df_test = test.copy()

print("Assume store open, if not provided")
df_test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")
df_train = df_train[df_train["Open"] != 0]


df_train = pd.merge(train,store,on='Store')
df_test = pd.merge(test,store,on='Store')


#df_train = df_train.sample(100000)


# ## FEATURE ENGINEERIN


sale_means = train.groupby('Store').mean().Sales
sale_means.name = 'Sales_Means'

df_train = df_train.join(sale_means,on='Store')
df_test = df_test.join(sale_means,on='Store')




# ## Transform dataframe to Matrix

y = df_train.Sales.tolist()

df_train_ = df_train.drop(['Date','Sales','Store','Customers'],axis=1).fillna(0)



train_dic = df_train_.fillna(0).to_dict('records')


test_dic = df_test.drop(["Date","Store","Id"],axis=1).fillna(0).to_dict('records')


#transfrom dataframe to matrix by dict vectorizer
dv = fe.DictVectorizer()
X = dv.fit_transform(train_dic)
Xo = dv.transform(test_dic)


#MIN_MAX SCALER
maxmin = preprocessing.MinMaxScaler()
X = maxmin.fit_transform(X.toarray())
Xo = maxmin.transform(Xo.toarray())


Xtrain,Xtest,Ytrain,Ytest = cv.train_test_split(X,y)



# ###### MODEL SELECTION

clf = es.RandomForestRegressor(n_estimators=25)
clf.verbose = True
clf.n_jobs = 8
clf


clf.fit(Xtrain,Ytrain)
print ("Training Score :" + str(clf.score(Xtrain,Ytrain)))
print ("Test Score : " + str(clf.score(Xtest,Ytest)) )


q = [i for i in zip(dv.feature_names_,clf.feature_importances_) ]

q = pd.DataFrame(q,columns = ['Feature_Names','Importance'],index=dv.feature_names_)

q_chart = q.sort('Importance').plot(kind='barh',layout='Feature_Names')

fig_q = q_chart.get_figure()
fig_q.savefig('feature_impartance.png')


Yresult = clf.predict(Xtest)
Yresult = np.array(Yresult)

Ytest = np.array(Ytest)

np.abs((Yresult - Ytest)).sum() / len(Yresult)


# PREDICTION


result = clf.predict(Xo)


output = pd.DataFrame(df_test.Id).join(pd.DataFrame(result,columns=['Sales']))

output.to_csv('submission.csv',index=False)

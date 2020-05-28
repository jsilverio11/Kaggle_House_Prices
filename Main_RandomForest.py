# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:02:28 2020
Project: Kaggle:House Prices
Script: Main_RandomForest.py
@author: JSilverio11
"""

## 1 - Import
## 1.1 Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, normalize
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import randint
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

## 1.2 Import Data
practice_data   = pd.read_csv('data/train.csv', index_col='Id')
evaluation_data = pd.read_csv('data/test.csv', index_col='Id')

## 2 - Preprocessing
## 2.1 Append practice and evaluation data
full_data = pd.concat([practice_data,evaluation_data])

## 2.2 Deal with NAs
## 2.2.1 Numeric Columns: Replace NAs with mean from entire column
numeric_columns = full_data.loc[:,(full_data.dtypes==np.int64) | (full_data.dtypes==np.float64)].columns.drop('SalePrice')
fill_numeric_na = lambda x: x.fillna(np.mean(x))
full_data[numeric_columns] = full_data[numeric_columns].apply(fill_numeric_na, axis=0)

## 2.2.2 Categorical Columns: Replace NAs with None
object_columns = full_data.loc[:,full_data.dtypes==object].columns
fill_object_na  = lambda x: x.fillna('None')
full_data[object_columns]  = full_data[object_columns].apply(fill_object_na, axis=0)

## 2.3 Split data back into practice and evaluation data
practice_processed   = full_data[~pd.isna(full_data['SalePrice'])]
evaluation_processed = full_data[pd.isna(full_data['SalePrice'])].drop('SalePrice', axis=1)

## 2.3 Split data into X_train, X_test, y_train, y_test
SEED = 123

X = practice_processed.drop('SalePrice', axis=1)
y = practice_processed['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

## 2.3 Feature Selection
## 2.3.1 Numeric Features
## 2.3.1.1 Correlation Matrix
price_correlations = pd.DataFrame(practice_processed.corr())['SalePrice'].abs().sort_values()
price_correlations.plot.bar()
plt.show()

## 2.3.1.2 Select columns with absolute value of correlation >= k
k = 0.2
numeric_cols_keep = price_correlations[(price_correlations >= k) & (price_correlations < 1)].index
X_full_numeric = X[numeric_cols_keep]

## 2.3.2 Categorical Features
## 2.3.2.1 Encode Categorical values to integers
X_train_object = X_train[object_columns]
X_test_object  = X_test[object_columns]
X_full_object = pd.concat([X_train_object,X_test_object], axis=0)

oe = OrdinalEncoder()
oe.fit(X_full_object)
X_train_enc = oe.transform(X_train_object)
X_test_enc  = oe.transform(X_test_object)

## 2.3.2.2 Select best categorical features according to chi squared test
fs = SelectKBest(score_func=chi2, k='all')
fs.fit(X_train_enc, y_train)
# for i in range(len(fs.scores_)):
#	print('Feature %d: %f' % (i, fs.scores_[i]))

# plt.bar([i for i in range(len(fs.scores_))], fs.scores_/max(fs.scores_))
# plt.show()
feature_scores = pd.Series(fs.scores_/max(fs.scores_), index=object_columns)
threshold = 0.05
category_cols_keep = feature_scores[feature_scores>threshold].index
X_full_object_trunc = X_full_object[category_cols_keep]

## 2.3.2.3 Categorize features with type object and create dummy variables
to_category = lambda x: x.astype('category')
X_full_category = X_full_object_trunc.apply(to_category, axis=0)
X_full_dummies = pd.get_dummies(X_full_category, prefix_sep='_')

## 2.3.3 Concatenate Numeric and categorical data
X_full_processed = pd.concat([X_full_numeric,X_full_dummies], axis=1)
X_train_processed = X_full_processed.loc[X_train.index,:]
X_test_processed  = X_full_processed.loc[X_test.index,:]

## 3 - Machine Learning Pipeline
## 3.1 Hyperparameter tunning
## 3.1.1 Initialize possible parameters
params = {'n_estimators': [200,250,300,350],
          'max_depth':[18,19,20,23,25,30],
		  'min_samples_leaf':[0.09,0.1,0.11,0.12],
          'max_features':['auto','sqrt','log2'],
          'bootstrap':[False,True],
          'oob_score':[False,True]}

## 3.1.2 Initialize model
tree = RandomForestRegressor(random_state=SEED)

## 3.1.3 Initialize GridSearchAlgorithm
tree_cv = RandomizedSearchCV(tree, params, cv=5)

## 3.1.4 Fit GridSearchAlgorithm to training data
tree_cv.fit(X_train_processed,y_train)

## 3.1.5 Get and evaluate best model parameters
print('\n\nBest parameters:', tree_cv.best_params_)
print('Best Score:', tree_cv.best_score_)

## 3.2 Build model
## 3.2.1 Initialize model with best parameters found
final_tree = RandomForestRegressor(n_estimators     = tree_cv.best_params_['n_estimators'],
                                   max_depth        = tree_cv.best_params_['max_depth'],
                                   min_samples_leaf = tree_cv.best_params_['min_samples_leaf'],
                                   max_features     = tree_cv.best_params_['max_features'],
                                   bootstrap        = tree_cv.best_params_['bootstrap'],
                                   oob_score        = tree_cv.best_params_['oob_score'],
                                   random_state     = SEED)

## 3.2.2 Fit model
final_tree.fit(X_train_processed,y_train)

## 3.2.3 Predict outputs for X_train e X_test
y_train_pred = final_tree.predict(X_train_processed)
y_test_pred  = final_tree.predict(X_test_processed)

## 3.2.4 Evaluate results
train_score_result = np.sqrt(mean_squared_error(np.log10(y_train),np.log10(y_train_pred)))
test_score_result  = np.sqrt(mean_squared_error(np.log10(y_test),np.log10(y_test_pred)))

print('\n\nRoot Mean Squared Log Error - train: ', train_score_result)
print('Root Mean Squared Log Error -  test: ', test_score_result)




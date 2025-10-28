import os
import time
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
import xgboost as xgb

from nltk import pos_tag
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer

import readabilityTransformer
import removePos
import customVectorizer
from load_BERT_embeddings import *
from stratifiedGroupSplitter import *


start_time = time.time()

# read in data
data_dir = 'data_readinglevel'
x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

# read in BERT data
xBERT_train_NH = load_arr_from_npz(os.path.join(data_dir, 'x_train_BERT_embeddings.npz'))
xBERT_test_NH = load_arr_from_npz(os.path.join(data_dir, 'x_test_BERT_embeddings.npz'))



# remove text based info as we will assume we do not need that to determine difficulty
# x_train_df = x_train_df.iloc[:,4:]

# clean training labels to be simply 0 or 1
y_tr_N = []
zeroCount = 0
oneCount = 0
for label in y_train_df['Coarse Label']:
	if label == "Key Stage 2-3":
		y_tr_N.append(0)
		zeroCount += 1
	else:
		y_tr_N.append(1)
		oneCount += 1

# split data using normal or new split method to place authors in one set only and ensure the balance of data exists
#X_train, X_val, y_train, y_val = train_test_split(x_train_df, y_tr_N, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = stratified_group_split(x_train_df, y_tr_N, group_col="author", test_size=0.2, random_state=42)
train_index = X_train.index
val_index = X_val.index

# taking the indicies from the split above, split BERT for similar purpose
X_train_BERT = xBERT_train_NH[train_index]
X_val_BERT = xBERT_train_NH[val_index]

svd = TruncatedSVD(n_components=256, random_state=42)
X_train_BERT_reduced = svd.fit_transform(X_train_BERT)
X_val_BERT_reduced = svd.transform(X_val_BERT)
X_test_BERT_reduced = svd.transform(xBERT_test_NH)


pipeline = Pipeline([
	('scaler', StandardScaler()),
	('clf', MLPClassifier(max_iter=1000, random_state=1, solver='adam', early_stopping=True))
])

neurons = [128,256]
layer_sizes = []

for n_layers in [1,2]:
    for combination in itertools.product(neurons, repeat=n_layers):
        layer_sizes.append(combination)

param_grid = {
	'clf__solver': ['adam'],
	'clf__alpha': np.logspace(-3, 1, 6),
	'clf__hidden_layer_sizes': layer_sizes
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# comment out one of the grids, top for in depth takes long time, bottom is for quick searches
grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='roc_auc',  cv=cv, n_jobs=-1)
# grid = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=200, cv=cv, scoring='roc_auc', n_jobs=-1)

grid.fit(X_train_BERT_reduced, y_train)

y_val_pred = grid.predict(X_val_BERT_reduced)
roc = roc_auc_score(y_val, y_val_pred)

print("Best parameters:", grid.best_params_)
print("Best cross-validated ROC AUC:", grid.best_score_)
print("Validation ROC AUC:", roc)

print("CM for Training: ")
best_model = grid.best_estimator_
y_hat = best_model.predict(X_train_BERT_reduced)
cm = confusion_matrix(y_train, y_hat)
print(cm)

print("CM for Validation: ")
best_model = grid.best_estimator_
y_hat = best_model.predict(X_val_BERT_reduced)
cm = confusion_matrix(y_val, y_hat)
print(cm)

yproba2_test = best_model.predict_proba(X_test_BERT_reduced)
with open("yproba2_test.txt", "w") as f:
	for entry in yproba2_test[:,1]:
		f.write(f"{entry}\n")

end_time = time.time() 
print(f"\n⏳ Total time elapsed: {end_time - start_time:.2f} seconds")
import numpy as np
import pandas as pd
import os
import nltk
from nltk import pos_tag
import textwrap
import sklearn.linear_model
import sklearn.model_selection
import readabilityTransformer
import removePos
import customVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import time
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPClassifier
import itertools
from sklearn.model_selection import RandomizedSearchCV

start_time = time.time()

# read in data
data_dir = 'data_readinglevel'
x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

# remove text based info as we will assume we do not need that to determine difficulty
x_train_df = x_train_df.iloc[:,4:]


# clean training labels to be simply 0 or 1
y_tr_N = []
for label in y_train_df['Coarse Label']:
	if label == "Key Stage 2-3":
		y_tr_N.append(0)
	else:
		y_tr_N.append(1)
		
X_train, X_val, y_train, y_val = train_test_split(x_train_df, y_tr_N, test_size=0.2, random_state=42)

pipeline = Pipeline([
	('scaler', StandardScaler()),
	('clf', MLPClassifier(max_iter=1000, random_state=1))
])

neurons = [3,4,5,6,7]
layer_sizes = []

for n_layers in [3]:
    for combination in itertools.product(neurons, repeat=n_layers):
        layer_sizes.append(combination)

param_grid = {
	'clf__solver': ['lbfgs'],
	'clf__alpha': np.logspace(-6, -1, 6),
	'clf__hidden_layer_sizes': layer_sizes
}

# comment out one of the grids, top for in depth takes long time, bottom is for quick searches
grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='roc_auc',  cv=5,)
# grid = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=200, cv=5, scoring='roc_auc')

grid.fit(X_train, y_train)


y_val_pred = grid.predict(X_val)
roc = roc_auc_score(y_val, y_val_pred)

print("Best parameters:", grid.best_params_)
print("Best cross-validated ROC AUC:", grid.best_score_)
print("Validation ROC AUC:", roc)

best_model = grid.best_estimator_
y_hat = best_model.predict(x_train_df.iloc[:,4:])
cm = confusion_matrix(y_tr_N, y_hat)
print(cm)

yproba2_test = best_model.predict_proba(x_test_df.iloc[:,4:])
with open("yproba2_test.txt", "w") as f:
	for entry in yproba2_test[:,1]:
		f.write(f"{entry}\n")

end_time = time.time()  # ⏱ End timer

print(f"\n⏳ Total time elapsed: {end_time - start_time:.2f} seconds")
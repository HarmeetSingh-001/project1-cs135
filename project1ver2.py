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
nltk.download('punkt_tab')

# read in data
data_dir = 'data_readinglevel'
x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

# clean training labels to be simply 0 or 1
y_tr_N = []
for label in y_train_df['Coarse Label']:
	if label == "Key Stage 2-3":
		y_tr_N.append(0)
	else:
		y_tr_N.append(1)

param_grid = {
	'features__tfidf__preprocessor': [removePos.remove_pos], #, removePos.remove_pos
	'features__tfidf__min_df': [1], #token frequency
	'features__tfidf__max_df': [2],
	'features__tfidf__min_count': [1],
	'clf__C': [1.0],	  # Regularization strength
	'clf__penalty': ['l2'],				 
	'clf__solver': ['lbfgs'], #,'sag','saga'
	'clf__max_iter': [500]
}

readability_pipeline = Pipeline([
	('readability', readabilityTransformer.ReadabilityTransformer()),
	('scale', StandardScaler())
])

#Minimum number of tokens gets built into a custom vectorizer
custom_tfidf = customVectorizer.CustomFilteredTfidfVectorizer()

# Combine both pipelines
full_pipeline = Pipeline([
	#Preprocessing pipeline stuff
	('features', FeatureUnion([
		('tfidf', custom_tfidf)
	])),
	('clf', sklearn.linear_model.LogisticRegression(max_iter=1000))
])
		
grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train_df['text'], y_tr_N)

print(" Best Params: " + str(grid_search.best_params_) + 
	" Best CV Accuracy: " + str(grid_search.best_score_))
		
#Use the best model we got to show confusion matrix and auroc
best_model = grid_search.best_estimator_
y_hat = best_model.predict(x_train_df['text'])
cm = confusion_matrix(y_tr_N, y_hat)
print(cm)

yproba1_test = best_model.predict_proba(x_test_df['text'])

with open("yproba1_test.txt", "w") as f:
	for entry in yproba1_test[:,1]:
		f.write(f"{entry}\n")

y_scores = best_model.predict_proba(x_train_df['text'])[:, 1]

fpr, tpr, thresholds = roc_curve(y_tr_N, y_scores)
auc_score = roc_auc_score(y_tr_N, y_scores)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Best Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
		

		








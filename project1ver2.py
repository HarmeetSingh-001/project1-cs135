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
#nltk.download('punkt_tab')
start_time = time.time()
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

X_train, X_val, y_train, y_val = train_test_split(
    x_train_df['text'], y_tr_N, test_size=0.2, random_state=42)

param_grid = {
	'features__tfidf__tfidf__preprocessor': [None,removePos.remove_pos], #, removePos.remove_pos
	'features__tfidf__tfidf__min_df': [10], #token frequency
	'features__tfidf__tfidf__min_count': [3],
	'features__tfidf__tfidf__ngram_range': [
        (1, 1),   # unigrams only
        #(1, 2),   # unigrams + bigrams
        #(1, 3),   # unigrams + bigrams + trigrams
        #(2, 2),   # bigrams only
        #(2, 3),   # bigrams + trigrams
    ],
	'clf__C': [0.1],	  # Regularization strength
	'clf__penalty': ['l2'],				 
	'clf__solver': ['lbfgs'], #,'sag','saga', 'lbfgs'
	'clf__max_iter': [500],
	'features__tfidf__selectk__k': [2500],
}

tfidf_pipeline = Pipeline([
    ('tfidf', customVectorizer.CustomFilteredTfidfVectorizer()),
    ('selectk', SelectKBest(chi2))  # keep top 2000 features
])

#Minimum number of tokens gets built into a custom vectorizer
custom_tfidf = customVectorizer.CustomFilteredTfidfVectorizer()

# Combine both pipelines
full_pipeline = Pipeline([
	#Preprocessing pipeline stuff
	('features', FeatureUnion([
		#('readability', readability_pipeline),
		('tfidf', tfidf_pipeline)
	])),
	('clf', sklearn.linear_model.LogisticRegression(max_iter=1000))
])
		
grid_search = GridSearchCV(full_pipeline, param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
#grid_search.fit(x_train_df['text'], y_tr_N)


		
grid_search.fit(X_train, y_train)

y_pred_val = grid_search.predict(X_val)
y_proba_val = grid_search.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_proba_val)

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

#y_scores = best_model.predict_proba(x_train_df['text'])[:, 1]

#fpr, tpr, thresholds = roc_curve(y_tr_N, y_hat)
#auc_score = roc_auc_score(y_tr_N, y_hat)
#plt.figure(figsize=(8,6))
#plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
#plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve for Best Model')
#plt.legend(loc='lower right')
#plt.grid(True)
#plt.show()
		
end_time = time.time()  # ⏱ End timer

print(f"\n⏳ Total time elapsed: {end_time - start_time:.2f} seconds")
		








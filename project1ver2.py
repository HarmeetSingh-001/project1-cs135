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
from sklearn.model_selection import GridSearchCV,GroupKFold,cross_val_score,GroupShuffleSplit
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
from utility import grouped_train_val_split_ids,plot_hyperparam_vs_score

data_dir = 'data_readinglevel'
x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

param_grid = {
	'features__tfidf__tfidf__preprocessor': [None], #, removePos.remove_pos
	#'features__tfidf__tfidf__min_df': [1,3,5,7,9], #token frequency, has no effect
	#'features__tfidf__tfidf__max_df': [10,30,50,70,90],
	#'features__tfidf__tfidf__min_count': [0,2,4,6,8],
	#'features__tfidf__tfidf__max_count': [300,600,900,1200,1500],
	#'features__tfidf__tfidf__min_length': [1,2,3,4,5],
	#'features__tfidf__tfidf__use_stemming': [True,False],
	'clf__class_weight':['balanced'],
	'clf__C': [ 1],	  # Regularization strength
	'clf__penalty': ['l2'],				 
	'clf__solver': ['lbfgs'], #,'sag','saga', 'lbfgs'
	'clf__max_iter': [250],
	'features__tfidf__selectk__k': [1000],
}

tfidf_pipeline = Pipeline([
    ('tfidf', customVectorizer.CustomFilteredTfidfVectorizer()),
    ('selectk', SelectKBest(chi2)) 
])

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

train_idx, val_idx = grouped_train_val_split_ids(
    x_train_df, y_train_df, test_size=0.2, random_state=42
)

y_tr_N = np.array([0 if label == "Key Stage 2-3" else 1 for label in y_train_df['Coarse Label']])
X_train, X_val = x_train_df.loc[train_idx, 'text'], x_train_df.loc[val_idx, 'text']
y_train, y_val = y_tr_N[train_idx], y_tr_N[val_idx]

groups = x_train_df.loc[X_train.index, 'author'] if isinstance(X_train, pd.Series) else x_train_df['author'].iloc[train_idx]
group_kfold = GroupKFold(n_splits=10)


grid_search = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid,
    cv=group_kfold,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train, groups=groups)
print("\nBest Params:", grid_search.best_params_)
print(f"Best CV AUROC: {grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_
y_hat = best_model.predict(X_val)
auc_score = roc_auc_score(y_val, y_hat)

print(f"heldout score:{auc_score}")

results = pd.DataFrame(grid_search.cv_results_)

plot_hyperparam_vs_score(results, 'param_features__tfidf__selectk__k')

cm = confusion_matrix(y_val, y_hat)
print(cm)

val_df = pd.DataFrame({
    'text': X_val,
    'true_label': y_val,
    'pred_label': y_hat,
    'correct': [int(t == p) for t, p in zip(y_val, y_hat)]
})

val_df['text_length'] = val_df['text'].apply(lambda x: len(str(x).split()))
val_df['char_length'] = val_df['text'].apply(lambda x: len(str(x)))
val_df['num_sentences'] = val_df['text'].apply(lambda x: len(str(x).split('.')))

accuracy = val_df['correct'].mean()
print(f"Held-out Accuracy: {accuracy:.3f}")
print(val_df['correct'].value_counts(normalize=True))

median_len = val_df['text_length'].median()
val_df['length_bucket'] = np.where(val_df['text_length'] > median_len, 'long', 'short')

length_perf = val_df.groupby('length_bucket')['correct'].mean()
print("\n📏 Accuracy by text length bucket:")
print(length_perf)

plt.figure(figsize=(6,4))
plt.bar(length_perf.index, length_perf.values, color=['#77a', '#a77'])
plt.ylabel("Held-out Accuracy")
plt.title("Model Accuracy by Text Length Bucket")
plt.show()

for _, row in val_df[val_df['correct'] == 0].sample(5, random_state=42).iterrows():
    print(f"\nText (len={row['text_length']}): {textwrap.shorten(row['text'], width=150)}")
    print(f"True: {row['true_label']} | Predicted: {row['pred_label']}")
    
class_perf = val_df.groupby('true_label')['correct'].mean()
print("\n📚 Accuracy by true label:")
print(class_perf)

plt.figure(figsize=(5,4))
plt.bar(class_perf.index.astype(str), class_perf.values, color=['#8fa', '#fa8'])
plt.ylabel("Accuracy")
plt.xlabel("True Label")
plt.title("Per-Class Accuracy on Held-Out Data")
plt.show()

fpr, tpr, thresholds = roc_curve(y_val, y_hat)

#plot auroc
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Best Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
		
# Generate report file
yproba1_test = best_model.predict_proba(x_test_df['text'])

with open("yproba1_test.txt", "w") as f:
	for entry in yproba1_test[:,1]:
		f.write(f"{entry}\n")
		
		







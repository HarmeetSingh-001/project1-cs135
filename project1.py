import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# === 1. Read and prepare data ===
data_dir = 'data_readinglevel'
x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

# Binary labels: 0 = simpler, 1 = more complex
y_tr_N = [0 if label == "Key Stage 2-3" else 1 for label in y_train_df['Coarse Label']]

# Make sure there’s an author column in your data
if 'author' not in x_train_df.columns:
    raise ValueError("x_train.csv must contain an 'author' column for grouped splitting.")

authors = x_train_df['author']

# === 2. Grouped Train/Validation Split ===
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(x_train_df['text'], y_tr_N, groups=authors))

X_train, X_val = x_train_df['text'].iloc[train_idx], x_train_df['text'].iloc[val_idx]
y_train, y_val = np.array(y_tr_N)[train_idx], np.array(y_tr_N)[val_idx]

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
print(f"Unique authors in train: {len(set(authors.iloc[train_idx]))}, val: {len(set(authors.iloc[val_idx]))}")

# === 3. Define pipeline ===
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english', ngram_range=(1, 1), min_df=1)),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(max_iter=1000, solver='lbfgs'))
])

# === 4. Parameter grid ===
param_grid = {
    'vectorizer__max_features': [2500],
    'vectorizer__min_df': [3, 5, 7, 9, 11, 13],
    'clf__C': [0.001,0.01, 0.1]
}

# === 5. Grouped CV inside grid search ===
grouped_cv = GroupShuffleSplit(n_splits=10, test_size=0.1, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=grouped_cv,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train, y_train, groups=authors.iloc[train_idx])

print("\nBest Params:", grid_search.best_params_)
print(f"Best CV AUROC: {grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_

# === 6. Evaluate on held-out validation set ===
def evaluateHeldout(best_model, X_val, y_val):
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)
    print(f"Validation AUROC: {val_auc:.3f}")
    return val_auc

val_auc = evaluateHeldout(best_model, X_val, y_val)

# === 7. Export test predictions ===
def exportPredictions(best_model, x_test_df):
    yproba1_test = best_model.predict_proba(x_test_df['text'])
    with open("yproba1_test.txt", "w") as f:
        for entry in yproba1_test[:, 1]:
            f.write(f"{entry}\n")



# === 8. Generic plotting function for hyperparameter vs score (always include C) ===
def plot_hyperparam_vs_score(results_df, hyperparam):
    """
    Plots AUROC vs one hyperparameter, with different lines for clf__C values.
    Example: plot_hyperparam_vs_score(results_df, 'param_vectorizer__min_df')
    """
    if hyperparam not in results_df.columns:
        print(f"⚠️ {hyperparam} not found in results DataFrame columns.")
        print("Available columns:\n", list(results_df.columns))
        return
    
    if 'param_clf__C' not in results_df.columns:
        print("⚠️ Could not find 'param_clf__C' in results DataFrame.")
        return
    
    plt.figure(figsize=(8,6))
    
    # Plot AUROC vs hyperparameter, color-coded by C
    for c_val in sorted(results_df['param_clf__C'].unique()):
        subset = results_df[results_df['param_clf__C'] == c_val]
        plt.plot(
            subset[hyperparam],
            subset['mean_test_score'],
            marker='o',
            label=f'C={c_val}'
        )

    plt.xlabel(hyperparam.replace('param_', '').replace('__', ' '))
    plt.ylabel('Mean CV AUROC')
    plt.title(f'AUROC vs {hyperparam.replace("param_", "")} (colored by C)')
    plt.legend(title='Regularization Strength (C)')
    plt.grid(True)
    plt.show()


# === 9. Plot hyperparameter relationships ===
results_df = pd.DataFrame(grid_search.cv_results_)
# Example: visualize how AUROC changes with min_df
plot_hyperparam_vs_score(results_df, 'param_vectorizer__min_df')
# You can also plot for other hyperparams by changing the string
# e.g., plot_hyperparam_vs_score(results_df, 'param_clf__C')

exportPredictions(best_model, x_test_df)
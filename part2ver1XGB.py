import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from itertools import product
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, roc_auc_score
from stratifiedGroupSplitter import *
from load_BERT_embeddings import *

start_time = time.time()

# load in normal data
data_dir = 'data_readinglevel'
x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

# load in BERT data
xBERT_train_NH = load_arr_from_npz(os.path.join(data_dir, 'x_train_BERT_embeddings.npz'))
xBERT_test_NH = load_arr_from_npz(os.path.join(data_dir, 'x_test_BERT_embeddings.npz'))

# update y values to binary
y_tr_N = [0 if label == "Key Stage 2-3" else 1 for label in y_train_df['Coarse Label']]

# split data using new splitter, keeps authors grouped and keeps score balanced
X_train, X_val, y_train, y_val = stratified_group_split(x_train_df, y_tr_N, group_col="author", test_size=0.2, random_state=42)
train_index = X_train.index
val_index = X_val.index

# split BERT using the index from normal splits
X_train_BERT = xBERT_train_NH[train_index]
X_val_BERT = xBERT_train_NH[val_index]

# make BERT smaller, optimal value seems to be 31
svd = TruncatedSVD(n_components=31, random_state=42)
X_train_BERT_reduced = svd.fit_transform(X_train_BERT)
X_val_BERT_reduced = svd.transform(X_val_BERT)
X_test_BERT_reduced = svd.transform(xBERT_test_NH)

# --- Convert to DMatrix ---
D_train = xgb.DMatrix(X_train_BERT_reduced, label=y_train)
D_val = xgb.DMatrix(X_val_BERT_reduced, label=y_val)
D_test = xgb.DMatrix(X_test_BERT_reduced)

# grid search values for XGB
param_grid = {
    'learning_rate': [0.01, 0.05],
    'max_depth': [8, 10],
    'subsample': [0.8],
    'colsample_bytree': [0.6, 0.8],
    'gamma': [0, 0.1],
    'min_child_weight': [1, 5],
    'reg_alpha': [0.01, 0.1],
    'reg_lambda': [1]
}

num_boost_round = 1000
early_stopping_rounds = 50
evals = [(D_train, 'Train'), (D_val, 'Valid')]

best_auc = 0
best_params = None
best_ntree = None


# loop all combinations to find best parameters
for lr in param_grid['learning_rate']:
    for depth in param_grid['max_depth']:
        for csbt in param_grid['colsample_bytree']:
            for gamma in param_grid['gamma']:
                for mcw in param_grid['min_child_weight']:
                    for alpha in param_grid['reg_alpha']:
                        params = {
                            'objective': 'binary:logistic',
                            'eval_metric': 'auc',
                            'learning_rate': lr,
                            'max_depth': depth,
                            'subsample': 0.8,
                            'colsample_bytree': csbt,
                            'gamma': gamma,
                            'min_child_weight': mcw,
                            'reg_alpha': alpha,
                            'reg_lambda': 1,
                            'seed': 42
                        }

                        bst = xgb.train(
                            params,
                            D_train,
                            num_boost_round=num_boost_round,
                            evals=evals,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=False
                        )

                        val_proba = bst.predict(D_val) 
                        auc = roc_auc_score(y_val, val_proba)

                        if auc > best_auc:
                            best_auc = auc
                            best_params = params
                            best_ntree = bst.best_iteration

print("=== Best Parameters ===")
print(best_params)
print("Best Validation AUC:", best_auc)

# use the best params to train a model
bst_final = xgb.train(
    best_params,
    D_train,
    num_boost_round=best_ntree,
    evals=evals,
    verbose_eval=True
)

# use trained model to make predictions on all sets
y_train_proba = bst_final.predict(D_train)  
y_val_proba = bst_final.predict(D_val)
y_test_proba = bst_final.predict(D_test)

y_train_pred = (y_train_proba > 0.5).astype(int)
y_val_pred = (y_val_proba > 0.5).astype(int)

# print metrics, AUROC and CM for training and validation
print("Training ROC AUC:", roc_auc_score(y_train, y_train_proba))
print("Training Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
print("Validation ROC AUC:", roc_auc_score(y_val, y_val_proba))
print("Validation Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

# create file with our test predictions to submit
with open("yproba2_test.txt", "w") as f:
    for p in y_test_proba:
        f.write(f"{p}\n")

end_time = time.time()
print(f"\n⏳ Total time elapsed: {end_time - start_time:.2f} seconds")

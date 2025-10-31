import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def grouped_train_val_split_ids(x_train_df, y_train_df, test_size=0.2, random_state=42):

    y_tr_N = [0 if label == "Key Stage 2-3" else 1 for label in y_train_df['Coarse Label']]
    y_tr_N = np.array(y_tr_N)

    df = x_train_df.copy()
    df['label'] = y_tr_N

    rng = np.random.default_rng(random_state)

    author_stats = df.groupby('author').agg(
        n_texts=('text', 'size'),
        mean_label=('label', 'mean')
    ).reset_index()

    author_stats['majority_label'] = (author_stats['mean_label'] >= 0.5).astype(int)

    train_authors, val_authors = train_test_split(
        author_stats['author'],
        test_size=test_size,
        stratify=author_stats['majority_label'],
        random_state=random_state
    )

    train_idx = df.index[df['author'].isin(train_authors)].to_numpy()
    val_idx = df.index[df['author'].isin(val_authors)].to_numpy()

    return train_idx, val_idx

def plot_hyperparam_vs_score(results_df, hyperparam):
	
	plt.figure(figsize=(8,6))
	
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
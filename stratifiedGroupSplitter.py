import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_group_split(X, y, group_col, test_size=0.2, random_state=42):
    """
    Split data into train/validation sets such that:
    - No group (e.g., author) appears in both sets
    - Label distribution is roughly preserved
    """
    rng = np.random.RandomState(random_state)

    df = X.copy()
    df["label"] = y
    df["group"] = X[group_col]

    # Compute label proportions per group
    group_stats = (
        df.groupby("group")["label"]
        .agg(["count", "mean"])
        .rename(columns={"count": "n", "mean": "pos_frac"})
    ).reset_index()

    # Shuffle groups
    groups = group_stats["group"].sample(frac=1, random_state=random_state).tolist()

    # Initialize lists
    train_groups, val_groups = [], []
    train_pos, val_pos, train_total, val_total = 0, 0, 0, 0
    total_pos_rate = df["label"].mean()

    # Greedy allocation of groups
    for g in groups:
        g_rows = group_stats[group_stats["group"] == g]
        g_n = g_rows["n"].values[0]
        g_pos = g_rows["pos_frac"].values[0] * g_n

        # Estimate if adding to val set keeps the global pos rate closer to total_pos_rate
        if (len(val_groups) < len(groups) * test_size and
            abs((val_pos + g_pos) / (val_total + g_n + 1e-6) - total_pos_rate)
            < abs((train_pos + g_pos) / (train_total + g_n + 1e-6) - total_pos_rate)):
            val_groups.append(g)
            val_pos += g_pos
            val_total += g_n
        else:
            train_groups.append(g)
            train_pos += g_pos
            train_total += g_n

    # Build masks
    train_mask = X[group_col].isin(train_groups)
    val_mask = X[group_col].isin(val_groups)

    X_train, X_val = X[train_mask].copy(), X[val_mask].copy()
    y_train = df.loc[train_mask, "label"].values
    y_val = df.loc[val_mask, "label"].values

    return X_train, X_val, y_train, y_val
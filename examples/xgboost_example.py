import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ============================
# CONFIGURATION
# ============================

problem = "transfusion"       # base name of the CSV file, e.g. "transfusion" -> "transfusion.csv"
target_col = "Target"         # name of the target column in the CSV
csv_path = problem + ".csv"   # full path to dataset

n_splits = 10                 # stratified K-fold
n_mc_runs = 20                # independent runs per fold
n_trees = 50                  # number of trees in the ensemble

# ============================
# LOAD & PREPARE DATA
# ============================

data = pd.read_csv(csv_path, sep=",")

# Optional cleaning similar to your SMC script
data = data.replace(r'\?', np.nan, regex=True)
data = data.replace('y', 1)
data = data.replace('n', 0)
data = data.dropna()

# If target is categorical, encode it numerically
if data[target_col].dtype == 'object':
    classes = data[target_col].unique()
    print("Classes:", classes)
    class_map = {c: i for i, c in enumerate(classes)}
    data[target_col].replace(class_map, inplace=True)

X = data.drop(columns=[target_col]).to_numpy()
y = data[target_col].to_numpy()

n_classes = len(np.unique(y))
print(f"Dataset: {problem}")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Number of classes: {n_classes}")

# Choose objective consistent with classification
if n_classes == 2:
    xgb_objective = "binary:logistic"
    extra_params = {}
else:
    xgb_objective = "multi:softprob"
    extra_params = {"num_class": n_classes}

# ============================
# STORAGE FOR RESULTS
# ============================

# Per-fold means
XGB_ACC_MEAN = []
XGB_TREE_SIZE_MEAN = []

# For SE across *all* runs
XGB_ACC_ALL = []
XGB_TREE_SIZE_ALL = []

# ============================
# STRATIFIED K-FOLD + MULTIPLE RUNS
# ============================

kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    print(f"\n=== Fold {fold_idx} ===")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    fold_accs = []
    fold_tree_sizes = []

    for run_idx in range(n_mc_runs):
        seed = 1000 * fold_idx + run_idx

        # XGBoost classifier with mostly defaults, but:
        # - n_estimators = 50
        # - subsample = 0.75
        # - proper classification objective
        xgb_clf = XGBClassifier(
            n_estimators=n_trees,
            subsample=0.75,
            objective=xgb_objective,
            random_state=seed,
            n_jobs=-1,
            **extra_params
        )

        xgb_clf.fit(X_train, y_train)
        y_pred = xgb_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Compute average tree size (number of non-terminal nodes) via booster dump
        booster = xgb_clf.get_booster()
        df_trees = booster.trees_to_dataframe()

        # Feature != 'Leaf' are internal nodes
        internal_counts = (
            df_trees
            .assign(is_internal=lambda df: df["Feature"] != "Leaf")
            .groupby("Tree")["is_internal"]
            .sum()
            .values
        )

        avg_tree_size = float(np.mean(internal_counts))

        print(f"  Run {run_idx}: acc = {acc:.4f}, avg tree size = {avg_tree_size:.2f}")

        fold_accs.append(acc)
        fold_tree_sizes.append(avg_tree_size)

        XGB_ACC_ALL.append(acc)
        XGB_TREE_SIZE_ALL.append(avg_tree_size)

    # Per-fold means over the 20 runs
    XGB_ACC_MEAN.append(np.mean(fold_accs))
    XGB_TREE_SIZE_MEAN.append(np.mean(fold_tree_sizes))

# ============================
# AGGREGATE: OVERALL MEAN + SE
# ============================

XGB_ACC_ALL = np.array(XGB_ACC_ALL)
XGB_TREE_SIZE_ALL = np.array(XGB_TREE_SIZE_ALL)

n_runs = len(XGB_ACC_ALL)   # should be n_splits * n_mc_runs

xgb_acc_overall_mean = np.mean(XGB_ACC_ALL)
xgb_tree_overall_mean = np.mean(XGB_TREE_SIZE_ALL)

xgb_acc_se = np.std(XGB_ACC_ALL, ddof=1) / np.sqrt(n_runs)
xgb_tree_se = np.std(XGB_TREE_SIZE_ALL, ddof=1) / np.sqrt(n_runs)

print("\n=== Overall results across all runs ===")
print(f"Mean XGB ACC      = {xgb_acc_overall_mean:.4f} ± {xgb_acc_se:.44f} (SE)")
print(f"Mean XGB TREE SIZE= {xgb_tree_overall_mean:.2f} ± {xgb_tree_se:.2f} (SE)")

# ============================
# BUILD RESULTS DATAFRAME
# ============================

results = pd.DataFrame()

# Rows: 0..n_splits-1, 'avg', 'se'
split_labels = list(range(n_splits)) + ['avg', 'se']
results['SPLIT'] = split_labels

results['XGB ACC'] = XGB_ACC_MEAN + [xgb_acc_overall_mean, xgb_acc_se]
results['XGB TREE SIZE'] = XGB_TREE_SIZE_MEAN + [xgb_tree_overall_mean, xgb_tree_se]

out_path = f"{problem}_XGB_results.csv"
results.to_csv(out_path, index=False)

print(f"\nResults saved to: {out_path}")

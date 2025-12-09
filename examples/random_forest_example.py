import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ============================
# CONFIGURATION
# ============================

problem = "Raisin_Dataset"       # base name of the CSV file, e.g. "transfusion" -> "transfusion.csv"
target_col = "Target"         # name of the target column in the CSV
target_col = "Class"         # name of the target column in the CSV
csv_path = problem + ".csv"   # full path to dataset

n_splits = 10                 # stratified K-fold
n_mc_runs = 20                # independent runs per fold
n_trees = 50                  # number of trees in the forest

# ============================
# LOAD & PREPARE DATA
# ============================

data = pd.read_csv(csv_path, sep=",")

# Optional cleaning similar to your SMC script
data = data.replace(r'\?', np.nan, regex=True)
data = data.replace('y', 1)
data = data.replace('n', 0)
data = data.dropna()

# If target is categorical, you can encode it numerically
if data[target_col].dtype == 'object':
    classes = data[target_col].unique()
    print("Classes:", classes)
    class_map = {c: i for i, c in enumerate(classes)}
    data[target_col].replace(class_map, inplace=True)

X = data.drop(columns=[target_col]).to_numpy()
y = data[target_col].to_numpy()

print(f"Dataset: {problem}")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# ============================
# STORAGE FOR RESULTS
# ============================

# Per-fold means
RF_ACC_MEAN = []
RF_TREE_SIZE_MEAN = []

# For SE across *all* runs (10 * 20 = 200)
RF_ACC_ALL = []
RF_TREE_SIZE_ALL = []

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
        # Different random_state per run for diversity, but reproducible
        seed = 1000 * fold_idx + run_idx

        rf = RandomForestClassifier(
            n_estimators=n_trees,
            random_state=seed,  # controls bootstrap & feature sampling
            n_jobs=-1           # use all cores if you want
        )

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Compute average tree size (number of non-terminal nodes)
        tree_sizes = []
        for est in rf.estimators_:
            tree = est.tree_
            is_leaf = (tree.children_left == -1)
            non_leaf_count = np.sum(~is_leaf)  # total nodes - leaf nodes
            tree_sizes.append(non_leaf_count)

        avg_tree_size = float(np.mean(tree_sizes))

        print(f"  Run {run_idx}: acc = {acc:.4f}, avg tree size = {avg_tree_size:.2f}")

        fold_accs.append(acc)
        fold_tree_sizes.append(avg_tree_size)

        RF_ACC_ALL.append(acc)
        RF_TREE_SIZE_ALL.append(avg_tree_size)

    # Per-fold means over the 20 runs
    RF_ACC_MEAN.append(np.mean(fold_accs))
    RF_TREE_SIZE_MEAN.append(np.mean(fold_tree_sizes))

# ============================
# AGGREGATE: OVERALL MEAN + SE
# ============================

RF_ACC_ALL = np.array(RF_ACC_ALL)
RF_TREE_SIZE_ALL = np.array(RF_TREE_SIZE_ALL)

n_runs = len(RF_ACC_ALL)   # should be n_splits * n_mc_runs

rf_acc_overall_mean = np.mean(RF_ACC_ALL)
rf_tree_overall_mean = np.mean(RF_TREE_SIZE_ALL)

rf_acc_se = np.std(RF_ACC_ALL, ddof=1) / np.sqrt(n_runs)
rf_tree_se = np.std(RF_TREE_SIZE_ALL, ddof=1) / np.sqrt(n_runs)

print("\n=== Overall results across all runs ===")
print(f"Mean RF ACC      = {rf_acc_overall_mean:.4f} ± {rf_acc_se:.4f} (SE)")
print(f"Mean RF TREE SIZE= {rf_tree_overall_mean:.2f} ± {rf_tree_se:.2f} (SE)")

# ============================
# BUILD RESULTS DATAFRAME
# ============================

results = pd.DataFrame()

# Rows: 0..n_splits-1, 'avg', 'se'
split_labels = list(range(n_splits)) + ['avg', 'se']
results['SPLIT'] = split_labels

results['RF ACC'] = RF_ACC_MEAN + [rf_acc_overall_mean, rf_acc_se]
results['RF TREE SIZE'] = RF_TREE_SIZE_MEAN + [rf_tree_overall_mean, rf_tree_se]

out_path = f"{problem}_RF_results.csv"
results.to_csv(out_path, index=False)

print(f"\nResults saved to: {out_path}")

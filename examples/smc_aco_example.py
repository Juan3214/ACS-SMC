import numpy
from numpy.ma.core import append
from sklearn.model_selection import train_test_split
from discretesampling.base.algorithms import DiscreteVariableSMC_ACO
from discretesampling.base.executor.executor import Executor
import discretesampling.domain.decision_tree as dt
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd
import numpy as np
from mpi4py import MPI
from collections import defaultdict


def assign_mpi_tasks(rank: int, P: int, n_splits: int, n_test: int):
    """
    Assign (fold_idx, test_idx) pairs to a given MPI rank.

    Assumptions:
      - P = total number of MPI ranks (comm.Get_size()).
      - n_splits = number of CV folds.
      - n_test = number of Monte Carlo repetitions per fold.
      - We only allow P such that P divides n_test or n_test divides P.
        (You said you've already enforced this in your main script.)

    Returns:
      fold_indices: list[int]
      test_indices: list[int]
      such that for each k, (fold_indices[k], test_indices[k]) is a task
      to be executed by this rank.
    """
    # Safety check (can be removed if you already do it outside)
    if (P % n_test != 0) and (n_test % P != 0):
        raise ValueError(
            "P must be either a divisor or a multiple of n_test."
        )

    fold_indices = []
    test_indices = []

    # Case 1: P <= n_test  → parallelise over tests, all ranks do all folds
    if P <= n_test:
        # Tests assigned to this rank: j = rank, rank + P, rank + 2P, ...
        local_tests = list(range(rank, n_test, P))

        for fold_idx in range(n_splits):
            for j in local_tests:
                fold_indices.append(fold_idx)
                test_indices.append(j)

    else:
        # Case 2: P > n_test and P is a multiple of n_test
        # We both split over folds AND tests.
        # Let m = number of ranks per test index
        m = P // n_test

        # Each rank gets a fixed test index j:
        #   j = rank % n_test
        # And belongs to a subgroup for that test:
        #   group = rank // n_test in {0, ..., m-1}
        j = rank % n_test
        group = rank // n_test

        # Folds for this (group, j) are: fold_idx = group, group + m, group + 2m, ...
        local_folds = list(range(group, n_splits, m))

        for fold_idx in local_folds:
            fold_indices.append(fold_idx)
            test_indices.append(j)

    return fold_indices, test_indices


### Dataset definition
problem = "Breast_Cancer"
data = pd.read_csv(problem+".csv",sep=',')
target = "Target"
#target = "Class"
data=data.replace('\?',np.nan,regex = True)
data=data.replace('y',1)
data=data.replace('n',0)
data=data.dropna()
classes=data[target].unique()
print("N classes ",len(classes))
num_classes=[i for i in range(len(classes))]
data[target].replace(classes,num_classes, inplace=True)
X=data.loc[:, data.columns != target]
y=data[target]
X=np.array(X)
y=np.array(y)

### Test parameters
n_splits=10
n_test=20
kf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state=1)
#kf.get_n_splits(X,y);

### SMC hyperparameters
alpha = 1.0
beta = 3.0
heuristic = 0
Tsmc = 200  # number of SMC iterations
N = 50    # number of samples
a = 1

### MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
P = comm.Get_size()

if (P % n_test != 0) and (n_test % P != 0):
    raise ValueError(
        "The number of MPI ranks must be either an integer multiple of n_test "
        "and n_test must be divisible by P"
    )

fold_indices, test_indices = assign_mpi_tasks(rank, P, n_splits, n_test)

# Build mapping: for each fold, which tests j this rank should do
tasks_by_fold = defaultdict(list)
for f, j in zip(fold_indices, test_indices):
    tasks_by_fold[f].append(j)

# Precompute splits once
splits = list(kf.split(X, y))

# --- local accumulators per fold (for proper CV stats) ---
local_sum_acc      = np.zeros(n_splits, dtype=float)
local_sum_tree     = np.zeros(n_splits, dtype=float)
local_sum_best_acc = np.zeros(n_splits, dtype=float)
local_count        = np.zeros(n_splits, dtype=int)

# --- local per-iteration accumulator across ALL runs this rank performs ---
local_sum_acc_per_iter = np.zeros(Tsmc, dtype=float)
local_count_per_iter   = 0  # number of (fold, j) runs this rank executed

try:
    for fold_idx in range(n_splits):
        if fold_idx not in tasks_by_fold:
            continue  # this rank has no tasks for this fold

        train_index, test_index = splits[fold_idx]
        X_test = X[test_index]
        X_train = X[train_index]
        y_test = y[test_index]
        y_train = y[train_index]

        target = dt.TreeTarget(a)

        initialProposal = dt.TreeInitialProposal(X_train, y_train)
        dtSMC_ACO = DiscreteVariableSMC_ACO(dt.Tree, target, initialProposal, alpha, beta, use_optimal_L=False, exec=Executor())

        for j in tasks_by_fold[fold_idx]:
            # if rank == 3:
            #     print(f"rank = {rank} Split: {fold_idx} Test: {j}")

            treeSamples, trees_per_it, weight_it = dtSMC_ACO.sample(Tsmc, N, heuristic, j)

            avg_tree_sizes = np.average([len(tree.tree) for tree in treeSamples])
            smc_maj = dt.stats(treeSamples, X_test).predict(X_test, use_majority=True)
            smc_maj_acc = dt.accuracy(y_test, smc_maj)

            smc_maj_acc_k = np.zeros(Tsmc)
            for k in range(Tsmc):
                smc_maj_k = dt.stats(trees_per_it[k], X_test).predict(X_test, use_majority=True)
                smc_maj_acc_k[k] = dt.accuracy(y_test, smc_maj_k)

            smc_best_tree = [treeSamples[np.argmax(weight_it[0][-1])]]
            best_tree_prediction = dt.stats(smc_best_tree, X_test).predict(X_test, use_majority=True)
            best_tree_prediction_acc = dt.accuracy(y_test, best_tree_prediction)

            # --- accumulate per-fold quantities locally ---
            local_sum_acc[fold_idx]      += smc_maj_acc
            local_sum_tree[fold_idx]     += avg_tree_sizes
            local_sum_best_acc[fold_idx] += best_tree_prediction_acc
            local_count[fold_idx]        += 1

            # --- accumulate per-iteration accuracy across ALL runs ---
            local_sum_acc_per_iter += smc_maj_acc_k
            local_count_per_iter   += 1

    # ======== MPI reductions to rank 0 ========

    global_sum_acc      = comm.reduce(local_sum_acc,      op=MPI.SUM, root=0)
    global_sum_tree     = comm.reduce(local_sum_tree,     op=MPI.SUM, root=0)
    global_sum_best_acc = comm.reduce(local_sum_best_acc, op=MPI.SUM, root=0)
    global_count        = comm.reduce(local_count,        op=MPI.SUM, root=0)

    global_sum_acc_per_iter = comm.reduce(local_sum_acc_per_iter, op=MPI.SUM, root=0)
    global_count_per_iter = comm.reduce(local_count_per_iter, op=MPI.SUM, root=0)

    # ======== Rank 0: compute means, SE, and write CSVs ========
    if rank == 0:
        SMC_ACC_MEAN       = []
        SMC_TREE_SIZE_MEAN = []
        BEST_TREE_ACC_MEAN = []

        for fold_idx in range(n_splits):
            if global_count[fold_idx] == 0:
                # should not happen, but be robust
                SMC_ACC_MEAN.append(np.nan)
                SMC_TREE_SIZE_MEAN.append(np.nan)
                BEST_TREE_ACC_MEAN.append(np.nan)
            else:
                SMC_ACC_MEAN.append(global_sum_acc[fold_idx] / global_count[fold_idx])
                SMC_TREE_SIZE_MEAN.append(global_sum_tree[fold_idx] / global_count[fold_idx])
                BEST_TREE_ACC_MEAN.append(global_sum_best_acc[fold_idx] / global_count[fold_idx])

        # Convert to arrays for stats
        fold_means_acc      = np.array(SMC_ACC_MEAN, dtype=float)
        fold_means_tree     = np.array(SMC_TREE_SIZE_MEAN, dtype=float)
        fold_means_best_acc = np.array(BEST_TREE_ACC_MEAN, dtype=float)

        # mask out NaNs if any
        #valid_mask = ~np.isnan(fold_means_acc)
        #n_eff_folds = np.sum(valid_mask)  # normally = n_splits
        n_runs = n_splits * n_test

        # Overall means (same as averaging all runs, but via folds)
        smc_acc_overall_mean   = np.nanmean(fold_means_acc)
        smc_tree_overall_mean  = np.nanmean(fold_means_tree)
        best_tree_overall_mean = np.nanmean(fold_means_best_acc)

        # Standard errors ACROSS FOLDS (correct CV SE)
        smc_acc_se   = np.nanstd(fold_means_acc,  ddof=1) / np.sqrt(n_runs)
        smc_tree_se  = np.nanstd(fold_means_tree, ddof=1) / np.sqrt(n_runs)
        best_tree_se = np.nanstd(fold_means_best_acc, ddof=1) / np.sqrt(n_runs)

        # ======== Per-iteration average accuracy across ALL runs ========
        if global_count_per_iter > 0:
            SMC_ACC_AT_k_OVERALL_MEAN = (global_sum_acc_per_iter / float(global_count_per_iter)).tolist()
        else:
            SMC_ACC_AT_k_OVERALL_MEAN = [np.nan] * Tsmc

        # ======== Build results DataFrames ========
        results = pd.DataFrame()
        results_per_iteration = pd.DataFrame()

        # Rows: 0..n_splits-1, 'avg', 'se'
        splits_labels = [x for x in range(0, n_splits)] + ['avg', 'se']
        results['SPLIT'] = splits_labels

        # Per-fold means + overall mean + SE
        results['SMC ACC'] = SMC_ACC_MEAN + [smc_acc_overall_mean, smc_acc_se]
        results['SMC TREE SIZE'] = SMC_TREE_SIZE_MEAN + [smc_tree_overall_mean, smc_tree_se]
        results['BEST TREE ACC'] = BEST_TREE_ACC_MEAN + [best_tree_overall_mean, best_tree_se]

        # Per-iteration overall mean accuracy
        results_per_iteration['k'] = [x for x in range(0, Tsmc)]
        results_per_iteration['AVG ACC at iteration k'] = (SMC_ACC_AT_k_OVERALL_MEAN)

        results.to_csv(problem + '_' + str(heuristic) + '_mp.csv', index=False)
        results_per_iteration.to_csv(problem + '_' + str(heuristic) + '_mp_per_iteration.csv', index=False)

except ZeroDivisionError:
    print("SMC sampling failed due to division by zero on rank", rank)

# We are better at training, testing is another story.

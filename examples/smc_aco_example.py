import numpy
from sklearn.model_selection import train_test_split
from discretesampling.base.algorithms import DiscreteVariableSMC_ACO
from discretesampling.base.executor.executor import Executor
import discretesampling.domain.decision_tree as dt
import pandas as pd
import numpy as np

problem = "transfusion"
data = pd.read_csv(problem+".csv",sep=',')
target = "Target"
X=data.loc[:, data.columns != target]
y=data[target]
X=np.array(X)
y=np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
                                                    random_state=5)
alpha = 1.0;
beta = 3.0
heuristic =  3

a = 1

target = dt.TreeTarget(a)
initialProposal = dt.TreeInitialProposal(X_train, y_train)

dtSMC_ACO = DiscreteVariableSMC_ACO(dt.Tree, target, initialProposal,
                            alpha,beta,
                            use_optimal_L=False, exec=Executor())

try:
    treeSamples,trees_per_it,weight_it = dtSMC_ACO.sample(10, 50,heuristic)
    smcLabels = dt.stats(treeSamples, X_test).predict(X_test, use_majority=True)
    smc_acc = dt.accuracy(y_test, smcLabels)
    print(numpy.mean(smc_acc))
except ZeroDivisionError:
    print("SMC sampling failed due to division by zero")

import numpy
from sklearn.model_selection import train_test_split
from discretesampling.base.algorithms import DiscreteVariableSMC_ACO
from discretesampling.base.executor.executor import Executor
import discretesampling.domain.decision_tree as dt
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd
import numpy as np

problem = "transfusion"
data = pd.read_csv(problem+".csv",sep=',')
target = "Target"
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
n_splits=10
n_test=20
kf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state=1)
kf.get_n_splits(X,y);
alpha = 1.0;
beta = 3.0
heuristic =  3
#Heuristic = 3 SMC-ACS
#Heuristic = 0 SMC

a = 1

SMC_ACC_MEAN=[]
try:
    for k,(train_index,test_index) in enumerate(kf.split(X,y)):
        X_test = X[test_index]
        X_train= X[train_index]
        y_test = y[test_index]
        y_train= y[train_index]
        target = dt.TreeTarget(a)
        SMC_ACC=[]
        initialProposal = dt.TreeInitialProposal(X_train, y_train)
        dtSMC_ACO = DiscreteVariableSMC_ACO(dt.Tree, target, initialProposal,
                            alpha,beta,
                            use_optimal_L=False, exec=Executor())
        for j in range(20):
            treeSamples,trees_per_it,weight_it = dtSMC_ACO.sample(200, 50,heuristic,j)
            smc_maj =dt.stats(treeSamples, X_test).predict(X_test, use_majority=True)
            smc_maj_acc = dt.accuracy(y_test, smc_maj)
            print(smc_maj_acc)
            SMC_ACC.append(smc_maj_acc)
        SMC_ACC_MEAN.append(np.mean(SMC_ACC))
    SMC_ACC_MEAN = SMC_ACC_MEAN+[np.mean(SMC_ACC_MEAN)]
    resultados=pd.DataFrame()
    resultados['SPLIT']=[x for x in range(0,10)] + ['avg']
    resultados['SMC ACC']=SMC_ACC_MEAN
    resultados.to_csv(problem+'_'+str(heuristic)+'_mp.csv',index=False)
except ZeroDivisionError:
    print("SMC sampling failed due to division by zero")

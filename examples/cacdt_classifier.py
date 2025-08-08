
#!/usr/bin/env python3
"""cACDT – Continuous-attribute Ant Colony Decision Tree (with StratifiedKFold evaluation)."""

from __future__ import annotations
import argparse, math, random, statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any

import numpy as np, pandas as pd
from scipy.io import arff  # type: ignore
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import os

# ---------------- Dataset loader ----------------
def load_dataset(path:str,target:Optional[str|int]=None)->Tuple[pd.DataFrame,np.ndarray]:
    if path.lower().endswith('.csv'):
        df=pd.read_csv(path)
    elif path.lower().endswith('.arff'):
        data,_=arff.loadarff(path)
        df=pd.DataFrame({c:(data[c].str.decode() if data[c].dtype.kind=='S' else data[c])
                         for c in data.dtype.names})
    else:
        raise ValueError('Unsupported file type')
    if isinstance(target,int): target=df.columns[target]
    elif target is None: target=df.columns[-1]
    y=df[target].astype('category').cat.codes.to_numpy(np.int32)
    X=df.drop(columns=[target])
    return X,y

# ---------------- Split & Node ----------------
@dataclass
class Split: attr:str; value:Any; is_numeric:bool
@dataclass
class Node:
    split:Optional[Split]=None; left:Optional['Node']=None; right:Optional['Node']=None; label:Optional[int]=None
    def is_leaf(self): return self.split is None

# ---------------- Twoing heuristic ----------------
def twoing(y_left,y_right)->float:
    if not y_left or not y_right: return 0.0
    n=len(y_left)+len(y_right)
    pl=len(y_left)/n; pr=1-pl
    classes=set(y_left)|set(y_right)
    diff=sum(abs(y_left.count(c)/len(y_left)-y_right.count(c)/len(y_right)) for c in classes)
    return (pl*pr/4.0)*diff*diff

# ---------------- cACDT core (condensed) ----------------
class cACDTClassifier:
    def __init__(self,*,max_depth=None,min_samples_split=2,num_ants=10,num_iter=25,
                 alpha=1.0,beta=3.0,gamma=0.1,random_state=None):
        self.max_depth=max_depth; self.min_samples_split=max(2,min_samples_split)
        self.num_ants=num_ants; self.num_iter=num_iter
        self.alpha=alpha; self.beta=beta; self.gamma=gamma
        self.rng=random.Random(random_state)
        self.root:Optional[Node]=None; self.pher:Dict[Tuple[str,Any,bool],float]={}

    def _majority(self,y): return max(set(y),key=y.count)

    def _candidate_splits(self,X:pd.DataFrame,y:Sequence[int])->List[Split]:
        cands=[]
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                uniq=np.unique(X[col].values)
                thr=(uniq[:-1]+uniq[1:])/2 if uniq.size>1 else []
                for t in thr: cands.append(Split(col,t,True))
            else:
                for v in X[col].unique(): cands.append(Split(col,v,False))
        return cands

    def _build_tree_ant(self,X:pd.DataFrame,y:List[int],depth:int=0)->Node:
        if len(set(y))==1 or len(y)<self.min_samples_split or (self.max_depth and depth>=self.max_depth):
            return Node(label=self._majority(y))
        cands=self._candidate_splits(X,y)
        if not cands: return Node(label=self._majority(y))
        scores=[]
        for sp in cands:
            mask=X[sp.attr]<=sp.value if sp.is_numeric else X[sp.attr]==sp.value
            y_left=[y[i] for i,m in enumerate(mask) if m]
            y_right=[y[i] for i,m in enumerate(mask) if not m]
            h=twoing(y_left,y_right)+1e-9
            tau=self.pher.get((sp.attr,sp.value,sp.is_numeric),1.0)
            scores.append((tau**self.alpha)*(h**self.beta))
        probs=np.array(scores); probs/=probs.sum()
        sp=self.rng.choices(cands,weights=probs,k=1)[0]
        mask=X[sp.attr]<=sp.value if sp.is_numeric else X[sp.attr]==sp.value
        if mask.sum()==0 or mask.sum()==len(y): return Node(label=self._majority(y))
        left=self._build_tree_ant(X[mask].reset_index(drop=True),[y[i] for i,m in enumerate(mask) if m],depth+1)
        right=self._build_tree_ant(X[~mask].reset_index(drop=True),[y[i] for i,m in enumerate(mask) if not m],depth+1)
        return Node(split=sp,left=left,right=right)

    def fit(self,X:pd.DataFrame,y:Sequence[int]):
        y=list(y)
        best_root=None; best_acc=-math.inf
        for _ in range(self.num_iter):
            iter_best=None; iter_acc=-math.inf
            for _ in range(self.num_ants):
                root=self._build_tree_ant(X,y)
                acc=self._score_tree(root,X,y)
                if acc>iter_acc: iter_best,iter_acc=root,acc
            # pheromone update
            for k in self.pher: self.pher[k]*=(1-self.gamma)
            def collect(node):
                if node.split:
                    k=(node.split.attr,node.split.value,node.split.is_numeric)
                    self.pher[k]=self.pher.get(k,1.0)+iter_acc
                    collect(node.left); collect(node.right)
            collect(iter_best)
            if iter_acc>best_acc: best_root,best_acc=iter_best,iter_acc
        self.root=best_root; return self

    def _predict_one(self,node,row):
        if node.is_leaf(): return node.label
        branch=node.left if ((row[node.split.attr]<=node.split.value) if node.split.is_numeric else (row[node.split.attr]==node.split.value)) else node.right
        return self._predict_one(branch,row)

    def _predict_with(self, root: Node, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._predict_one(root, row) for _, row in X.iterrows()])

    def predict(self,X): return np.array([self._predict_one(self.root,row) for _,row in X.iterrows()])
    def _score_tree(self, root: Node, X: pd.DataFrame, y) -> float:
        return (self._predict_with(root, X) == y).mean()
    def score(self,X,y): return (self.predict(X)==y).mean()

# ------------- Stratified CV evaluation -------------
def evaluate_stratified_cv(X:pd.DataFrame,y:np.ndarray,folds:int=5,**clf_kw)->Tuple[float,float]:
    skf=StratifiedKFold(n_splits=folds,shuffle=True,random_state=0)
    accs=[]
    i = 0
    for train_idx,test_idx in skf.split(X,y):
        X_tr,X_te=X.iloc[train_idx],X.iloc[test_idx]
        y_tr,y_te=y[train_idx],y[test_idx]
        clf=cACDTClassifier(**clf_kw).fit(X_tr,y_tr)
        accs.append(clf.score(X_te,y_te))
        print(i)
        i = i+1
    return float(np.mean(accs)), float(np.std(accs)/math.sqrt(folds))

# -------------------- CLI --------------------
def _cli():
    ap=argparse.ArgumentParser(description='cACDT with StratifiedKFold eval')
    ap.add_argument('dataset'); ap.add_argument('--target')
    ap.add_argument('--folds',type=int,default=5); ap.add_argument('--ants',type=int,default=10)
    ap.add_argument('--iters',type=int,default=25); args=ap.parse_args()
    X,y=load_dataset(args.dataset,args.target)
    #print(ap.add_argument('dataset'))
    mean,se=evaluate_stratified_cv(X,y,folds=args.folds,num_ants=args.ants,num_iter=args.iters)
    dataset_name = os.path.basename(args.dataset)
    print(
        f"{dataset_name}: stratified {args.folds}-fold accuracy "
        f"{mean:.4f} ± {se:.4f}"
    )
if __name__=='__main__': _cli()

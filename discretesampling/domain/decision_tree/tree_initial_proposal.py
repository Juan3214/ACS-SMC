import math
import numpy as np
from discretesampling.base.types import DiscreteVariableInitialProposal
from discretesampling.base.random import RNG
from discretesampling.domain.decision_tree import Tree


class TreeInitialProposal(DiscreteVariableInitialProposal):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        # self.rng = rng

    # def sample(self, rng):
    #     leafs = [1, 2]

    #     feature = rng.randomInt(0, len(self.X_train[0])-1)
    #     threshold = rng.randomInt(0, len(self.X_train)-1)
    #     tree = [[0, 1, 2, feature, self.X_train[threshold, feature],0]]
    #     return Tree(self.X_train, self.y_train, tree, leafs)

    def sample_ACO(self, rng=RNG(), target=None):
        '''
        Build an initial tree using the heuristic twoing criterion.
        '''
        leafs = [1, 2]
        #modified for ACO algorithm matrix 
        X_data_in_node = [i for i in range(len(self.X_train))]
        deterministic= 0
        num_classes_in_node = len(np.unique(self.y_train[X_data_in_node]))
        count_X_data_in_node=len(X_data_in_node)
        prob_thresh={}
        #count_in_1={}
        #count_in_2={}
        sum_twoing=0.0
        sum_twoing_feat={}
         
        c=list(set(self.y_train))
        class_data_feat= np.array(self.y_train[X_data_in_node])
        for feat in range(len(self.X_train[0])):
            array=np.array(list(set(self.X_train[:, feat])))
            if len(array) < 100:
                set_of_thresh = array
            else:
                set_of_thresh=[np.percentile(array,i) for i in range(1,101)]
            # divide all the threshold in percentile, and use it for selecting the thresholds
            data_feat=np.array(self.X_train[X_data_in_node,feat])
            sum_twoing_feat[feat]=0.0
            for thresh in set_of_thresh:
                mask = data_feat > thresh
                count_in_1 = np.bincount(class_data_feat[mask],minlength=max(c)+1)
                count_in_2 = np.bincount(class_data_feat[~mask],minlength=max(c)+1)
                count_1 = len(class_data_feat[mask])
                count_2 = len(class_data_feat[~mask])
                twoing=0.0;cumulative_sum=0.0
                if count_1!=0 and count_2!=0:
                    prob_1=count_1/count_X_data_in_node
                    prob_2=count_2/count_X_data_in_node
                    for clas in c:
                        cumulative_sum+=abs(((count_in_1[clas]/count_1)-(count_in_2[clas]/count_2)))
                        twoing=(prob_1*prob_2/4)*(cumulative_sum**2)
                prob_thresh[(feat,thresh)]=twoing
                sum_twoing+=twoing
                sum_twoing_feat[feat]+=twoing
        if sum_twoing!=0.0:
            if deterministic==0:
                # next two dictionaries are for the eval function
                prob_feat_eval={}
                # probability of selecting a particular feature
                prob_thresh_eval={}
                # probability of selecting a particular threshold of the feature
                for feat in range(len(self.X_train[0])): 
                    sum_prob_feat_eval=0.0
                    array=np.array(list(set(self.X_train[:, feat])))
                    if len(array) < 100:
                        set_of_thresh = array
                    else:
                        set_of_thresh=[np.percentile(array,i) for i in range(1,101)]
                    for thresh in set_of_thresh:
                        if sum_twoing_feat[feat]!=0.0:
                            prob_thresh_eval[(feat,thresh)] = (prob_thresh[(feat,thresh)]/sum_twoing_feat[feat])
                        else:
                            prob_thresh_eval[(feat,thresh)] = 0.0
                        prob_thresh[(feat,thresh)]/=sum_twoing
                        sum_prob_feat_eval+=prob_thresh[(feat,thresh)]
                    prob_feat_eval[feat]=sum_prob_feat_eval 
                feat_thresh=rng.randomChoices(list(prob_thresh.keys()),weights=list(prob_thresh.values()))
                threshold=feat_thresh[0][1]
                feature=int(feat_thresh[0][0])
                #save the probability of the selected feature-threshold for eval
                
            else:
                best_key=np.argmax(list(prob_thresh.values()))
                feat_thresh=list(prob_thresh.keys())[best_key]
                new_threshold=feat_thresh[1]
                new_feature=int(feat_thresh[0])
 
        tree = [[0, 1, 2, feature, threshold, 0,-1]]
        init_tree = Tree(self.X_train, self.y_train, tree, leafs)

        if target is None:
            return init_tree

        i = 0
        while i < len(leafs):
            u = rng.uniform()
            prior = math.exp(target.evaluatePrior(init_tree))
            # print("tree before: ", init_tree)
            if u < prior:
                init_tree = init_tree.grow_leaf_ACO(leafs.index(leafs[i]), rng)
                leafs = init_tree.leafs
            else:
                i += 1
        #    # print("tree after: ", init_tree)
        return init_tree
    def sample(self, rng=RNG(), target=None):
        leafs = [1, 2]
        #modified for ACO algorithm matrix 
        feature = rng.randomInt(0, len(self.X_train[0])-1)
        array=np.array(list(set(self.X_train[:, feature])))
        if len(array) < 100:
            set_of_thresh = array
        else:
            set_of_thresh=[np.percentile(array,i) for i in range(1,101)]
        threshold = rng.randomInt(0, len(set_of_thresh)-1)
        threshold = set_of_thresh[threshold]
        tree = [[0, 1, 2, feature, threshold, 0,-1]]
        init_tree = Tree(self.X_train, self.y_train, tree, leafs)

        if target is None:
            return init_tree

        i = 0
        while i < len(leafs):
            u = rng.uniform()
            prior = math.exp(target.evaluatePrior(init_tree))
            # print("tree before: ", init_tree)
            if u < prior:
                init_tree = init_tree.grow_leaf(leafs.index(leafs[i]), rng)
                leafs = init_tree.leafs
            else:
                i += 1
            # print("tree after: ", init_tree)
        return init_tree

    def eval(self, x, target=None):
        num_features = len(self.X_train[0])
        num_thresholds = len(self.X_train)
        if target is None:
            return -math.log(num_features) - math.log(num_thresholds)
        else:
            return -math.log(num_features) - math.log(num_thresholds) + target.evaluatePrior(x)

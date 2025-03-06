import copy
import numpy as np
from discretesampling.base.random import RNG
from discretesampling.base import types
from discretesampling.domain.decision_tree.util import encode_move, decode_move, extract_tree, extract_leafs
from discretesampling.domain.decision_tree.tree_distribution import TreeProposal
from discretesampling.domain.decision_tree.tree_target import TreeTarget
from discretesampling.domain.decision_tree.metrics import stats, accuracy,calculate_leaf_occurences

class Tree(types.DiscreteVariable):
    def __init__(self, X_train, y_train, tree, leafs, lastAction=""):
        self.X_train = X_train
        self.y_train = y_train
        self.tree = tree
        self.leafs = leafs
        self.small_float = 4.9406564584124654e-324
        self.prune_reverse = [self.small_float,self.small_float] #reverse f t probability
        self.change_reverse = [self.small_float,self.small_float] # reverse f t probability
        self.lastAction = lastAction
        self.data_in_node={}
        self.data_in_leaf={}
        self.parents_leafs={1:0,2:0}
        self.node_last_modification=0
        self.c=list(set(y_train))
        self.last_move_prob_f_t=(1.0/len(self.X_train[0])),1.0/len(self.X_train[:])
    def __eq__(self, x) -> bool:
        return (x.X_train == self.X_train).all() and\
            (x.y_train == self.y_train).all() and\
            x.tree == self.tree and x.leafs == self.leafs

    def __str__(self):
        return str(self.tree)

    def __copy__(self):
        # Custom __copy__ to ensure tree and leaf structure are deep copied
        new_tree = Tree(
            self.X_train,
            self.y_train,
            copy.deepcopy(self.tree),
            copy.deepcopy(self.leafs)
        )
        new_tree.data_in_node=self.data_in_node
        new_tree.data_in_leaf=self.data_in_leaf
        new_tree.parents_leafs=self.parents_leafs.copy()
        return new_tree

    @classmethod
    def getProposalType(self):
        return TreeProposal

    @classmethod
    def getTargetType(self):
        return TreeTarget

    @classmethod
    def encode(cls, x):
        tree = np.array(x.tree).flatten()
        leafs = np.array(x.leafs).flatten()
        last_action = encode_move(x.lastAction)
        tree_dim = len(tree)
        leaf_dim = len(leafs)

        x_new = np.hstack(
            (np.array([tree_dim, leaf_dim, last_action]), tree, leafs)
        )

        return x_new

    @classmethod
    def decode(cls, x, particle):
        tree_dim = x[0].astype(int)
        leaf_dim = x[1].astype(int)
        last_action = decode_move(x[2].astype(int))

        return Tree(
            particle.X_train,
            particle.y_train,
            extract_tree(x[3:(3+tree_dim)]),
            extract_leafs(x[(3+tree_dim):(3+tree_dim+leaf_dim)]),
            last_action
        )
    def prunable_node_indices(self):
        candidates = []
        for i in range(1, len(self.tree)): # cannot prune the root
            node_to_prune = self.tree[i]
            if ((node_to_prune[1] in self.leafs) and (node_to_prune[2] in self.leafs)):
                candidates.append(i)
        return(candidates)
    def depth_of_leaf(self, leaf):
        depth = 0
        for node in self.tree:
            if node[1] == leaf or node[2] == leaf:
                depth = node[5]+1

        return depth

    def grow_leaf(self, index, rng=RNG()):
        action = "grow"
        self.lastAction = action
        '''
        grow tree by just simply creating the individual nodes. each node
        holds their node index, the left and right leaf index, the node
        feature and threshold
        '''
        random_index = index
        leaf_to_grow = self.leafs[random_index]

        # generating a random feature
        feature = rng.randomInt(0, len(self.X_train[0])-1)
        # generating a random threshold
        threshold = rng.randomInt(0, len(self.X_train)-1)
        threshold = (self.X_train[threshold, feature])
        depth = self.depth_of_leaf(leaf_to_grow)
        node = [leaf_to_grow, max(self.leafs)+1, max(self.leafs)+2, feature,
                threshold, depth, self.parents_leafs[leaf_to_grow]]

        # add the new leafs on the leafs array
        leaf_1=max(self.leafs)+1
        self.leafs.append(leaf_1)
        leaf_2=max(self.leafs)+1
        self.leafs.append(leaf_2)
        # save the parents of the leafs
        self.parents_leafs[leaf_1]=leaf_to_grow
        self.parents_leafs[leaf_2]=leaf_to_grow
        # delete from leafs the new node
        self.leafs.remove(leaf_to_grow)
        del self.parents_leafs[leaf_to_grow]
        self.tree.append(node)
        return self

    def grow(self, rng=RNG(),min_data=2):
        action = "grow"
        self.lastAction = action
        '''
        grow tree by just simply creating the individual nodes. each node
        holds their node index, the left and right leaf index, the node
        feature and threshold
        '''
        random_index = rng.randomInt(0, len(self.leafs)-1)
        leaf_to_grow = self.leafs[random_index]
        X_data_in_leaf=self.data_in_leaf[leaf_to_grow]
        count_X_data_in_leaf=len(X_data_in_leaf)
        num_class = len(set(self.y_train[X_data_in_leaf]))
        # generating a random feature
        if (count_X_data_in_leaf<=min_data or num_class <2 ):
            return self.change(rng=rng)
        
        feature = rng.randomInt(0, len(self.X_train[0])-1)
        # generating a random threshold
        threshold = rng.randomInt(0, len(self.X_train)-1)
        threshold = (self.X_train[threshold, feature])
        depth = self.depth_of_leaf(leaf_to_grow)
        node = [leaf_to_grow, max(self.leafs)+1, max(self.leafs)+2, feature,
                threshold, depth, self.parents_leafs[leaf_to_grow]]

        # add the new leafs on the leafs array
        leaf_1=max(self.leafs)+1
        self.leafs.append(leaf_1)
        leaf_2=max(self.leafs)+1
        self.leafs.append(leaf_2)
        self.parents_leafs[leaf_1]=leaf_to_grow
        self.parents_leafs[leaf_2]=leaf_to_grow
        del self.parents_leafs[leaf_to_grow]
        # delete from leafs the new node
        self.leafs.remove(leaf_to_grow)
        self.tree.append(node)
        
        return self

    def prune(self, rng=RNG()):
        action = "prune"
        self.lastAction = action
        #print("i am here")
        '''
        For example when we have nodes 0,1,2 and leafs 3,4,5,6 when we prune
        we take the leafs 6 and 5 out, and the
        node 2, now becomes a leaf.
        '''
        new_parents_leafs={}
        candidates = self.prunable_node_indices()
        nc = len(candidates)
        random_index = rng.randomInt(0, nc-1)
        index_to_prune = candidates[random_index]
        node_to_prune = self.tree[index_to_prune]
        # Need to compute the prob of grow for the reverse, 
        parent_node = node_to_prune[-1]
        feature_node_to_prune = node_to_prune[3]
        threshold_node_to_prune = node_to_prune[4]
        # while random_index == 0:
        #     random_index = rng.randomInt(0, len(self.tree)-1)
        #     node_to_prune = self.tree[random_index]

        # if (node_to_prune[1] in self.leafs) and\
        #         (node_to_prune[2] in self.leafs):
        #     print("here")
            # remove the pruned leafs from leafs list and add the node as a
            # leaf
        self.leafs.append(node_to_prune[0])
        self.leafs.remove(node_to_prune[1])
        self.leafs.remove(node_to_prune[2])
        del self.parents_leafs[node_to_prune[1]]
        del self.parents_leafs[node_to_prune[2]]
        self.parents_leafs[node_to_prune[0]] = node_to_prune[6]
        # delete the specific node from the node lists
        del self.tree[index_to_prune]
            
        # else:

        #     delete_node_indices = []
        #     i = 0
        #     for node in self.tree:
        #         if node_to_prune[1] == node[0] or node_to_prune[2] == node[0]:
        #             delete_node_indices.append(node)
        #         i += 1
        #     self.tree.remove(node_to_prune)
        #     for node in delete_node_indices:
        #         self.tree.remove(node)

        #     for i in range(len(self.tree)):
        #         for p in range(1, len(self.tree)):
        #             count = 0
        #             for k in range(len(self.tree)-1):
        #                 if self.tree[p][0] == self.tree[k][1] or\
        #                         self.tree[p][0] == self.tree[k][2]:
        #                     count = 1
        #             if count == 0:
        #                 self.tree.remove(self.tree[p])
        #                 break

        new_leafs = []
        for node in self.tree:
            count1 = 0
            count2 = 0
            for check_node in self.tree:
                if node[1] == check_node[0]:
                    count1 = 1
                if node[2] == check_node[0]:
                    count2 = 1

            if count1 == 0:
                new_leafs.append(node[1])
                new_parents_leafs[node[1]] = node[0]

            if count2 == 0:
                new_leafs.append(node[2])
                new_parents_leafs[node[2]] = node[0]
        
        
        self.leafs[:] = new_leafs[:]
        self.parents_leafs = new_parents_leafs.copy()
        # print(self.leafs)
        # # print(self.tree)
        # print("***********")
        return self

    def change(self, rng=RNG()):
        action = "change"
        self.lastAction = action
        '''
        we need to choose a new feature at first
        we then need to choose a new threshold base on the feature we have
        chosen and then pick unoformly a node and change their features and
        thresholds
        '''
        #max_depth = max([sublist[5] for sublist in self.tree])
 
        # Define the initial value
        #initial_value = 0.5
 
        # Define the number of times to divide
        #num_divisions = max_depth
        #temporary_list = [initial_value / (2 ** i) for i in range(1, num_divisions + 1)]
 
        #flip the list
        #change_per_depth_possibilities = temporary_list[::-1]
        #compute cumsum
        #cumulative_sum = np.cumsum(change_per_depth_possibilities)
 
        #random_number = rng.uniform()
        #random_number = random.random()
 
        #find the index in the cumsum list for the depth we need to change
        #index_found = np.searchsorted(cumulative_sum, random_number)
 
        #find the list of lists we can change
        #indices_with_index_found = [i for i, sublist in enumerate(self.tree) if sublist[5] == index_found]
 
        # Randomly select an index from the list of indices we can change
        #random_index = rng.randomChoice(indices_with_index_found)
        # random_index = random.choice(indices_with_index_found)
 
        # Retrieve the sublist corresponding to the randomly selected index
        #random_sublist = self.tree[random_index]
 
        #findex the index of this random sublist
        #random_sublist_index = self.tree.index(random_sublist)
        #random_index = rng.randomInt(0, len(self.tree)-1)
        #node_to_change = self.tree[random_sublist_index]
        random_index = rng.randomInt(0, len(self.tree)-1)
        node_to_change = self.tree[random_index]
        new_feature = rng.randomInt(0, len(self.X_train[0])-1)
        new_threshold = rng.randomInt(0, len(self.X_train)-1)
        node_to_change[3] = new_feature
        node_to_change[4] = self.X_train[new_threshold, new_feature]
        
        return self

    def swap(self, rng=RNG()):
        action = "swap"
        self.lastAction = action
        '''
        need to swap the features and the threshold among the 2 nodes
        '''
        random_index_1 = rng.randomInt(0, len(self.tree)-1)
        random_index_2 = rng.randomInt(0, len(self.tree)-1)
        node_to_swap1 = self.tree[random_index_1]
        node_to_swap2 = self.tree[random_index_2]

        # in case we choose the same node
        while node_to_swap1 == node_to_swap2:
            random_index_2 = rng.randomInt(0, len(self.tree)-1)
            node_to_swap2 = self.tree[random_index_2]

        temporary_feature = node_to_swap1[3]
        temporary_threshold = node_to_swap1[4]

        node_to_swap1[3] = node_to_swap2[3]
        node_to_swap1[4] = node_to_swap2[4]

        node_to_swap2[3] = temporary_feature
        node_to_swap2[4] = temporary_threshold

        return self
    def grow_H_threshold(self, rng=RNG(),deterministic=0,min_data=2):
        action = "grow"
        self.lastAction = action
        '''
        grow tree by just simply creating the individual nodes. each node
        holds their node index, the left and right leaf index, the node
        feature and threshold
        '''
        random_index = rng.randomInt(0, len(self.leafs)-1)
        leaf_to_grow = self.leafs[random_index]

        # generating a random feature
        feature = rng.randomInt(0, len(self.X_train[0])-1)
         
        X_data_in_leaf=self.data_in_leaf[leaf_to_grow]
        num_class = len(set(self.y_train[X_data_in_leaf]))
        count_X_data_in_leaf=len(X_data_in_leaf)
        if (count_X_data_in_leaf<=min_data or num_class < 2):
            return self.change_H_feature(rng=rng)
        if count_X_data_in_leaf!=0 and num_class >= 2:
            prob_thresh={}
            count_in_1={}
            count_in_2={}
            array=np.array(list(set(self.X_train[:, feature]))) # 
            if len(array) < 100:
                set_of_thresh = array
            else:
                set_of_thresh=[np.percentile(array,i) for i in range(1,101)]
            # divide all the threshold in percentile, and use it for selecting the thresholds
            sum_twoing=0
            for thresh in set_of_thresh:
                #for every threshold compute the twoing coeficient using a random feature
                for clas in self.c:
                    count_in_1[clas]=0
                    count_in_2[clas]=0
                count_1=0
                count_2=0
                for datum in X_data_in_leaf:
                    if self.X_train[datum,feature]>thresh:
                        count_in_1[self.y_train[datum]]+=1
                        count_1+=1
                    else:
                        count_in_2[self.y_train[datum]]+=1
                        count_2+=1
                twoing=0.0;cumulative_sum=0.0
                if count_1!=0 and count_2!=0:
                    prob_1=count_1/count_X_data_in_leaf
                    prob_2=count_2/count_X_data_in_leaf
                    for clas in self.c:
                        cumulative_sum+=abs(((count_in_1[clas]/count_1)-(count_in_2[clas]/count_2)))
                    twoing=(prob_1*prob_2/4)*(cumulative_sum**2)
                prob_thresh[thresh]=twoing
                sum_twoing+=twoing
                
            if sum_twoing!=0:
                if deterministic==0:
                    for thresh in set_of_thresh:
                        prob_thresh[thresh]/=sum_twoing
                    threshold=rng.randomChoices(list(prob_thresh.keys()),weights=list(prob_thresh.values()))
                    
                    self.last_move_prob_f_t=(1/len(self.X_train[0]),prob_thresh[threshold[0]])
                    #save the probability of the selected feature-threshold for eval
                else:
                    best_key=np.argmax(list(prob_thresh.values()))
                    threshold=list(prob_thresh.keys())[best_key]

            else:
                threshold = rng.randomInt(0, len(self.X_train)-1)
                threshold = (self.X_train[threshold, feature])
                self.last_move_prob_f_t=(1/len(self.X_train[0]),1/len(self.X_train[:]))
        else:
            threshold = rng.randomInt(0, len(self.X_train)-1)
            threshold = (self.X_train[threshold, feature])
            self.last_move_prob_f_t=(1/len(self.X_train[0]),1/len(self.X_train[:]))
        depth = self.depth_of_leaf(leaf_to_grow)
        node = [leaf_to_grow, max(self.leafs)+1, max(self.leafs)+2, feature,
                threshold, depth,self.parents_leafs[leaf_to_grow]]

        # add the new leafs on the leafs array
        leaf_1=max(self.leafs)+1
        self.leafs.append(leaf_1)
        leaf_2=max(self.leafs)+1
        self.leafs.append(leaf_2)
        # save the parents of the leafs
        self.parents_leafs[leaf_1]=leaf_to_grow
        self.parents_leafs[leaf_2]=leaf_to_grow
        del self.parents_leafs[leaf_to_grow]
        # delete from leafs the new node
        self.leafs.remove(leaf_to_grow)
        self.tree.append(node)
        return self
    def change_H_threshold(self, rng=RNG(),deterministic=0,min_data=2):
        action = "change"
        self.lastAction = action
        '''
        we need to choose a new feature at first
        we then need to choose a new threshold base on the feature we have
        chosen and then pick unoformly a node and change their features and
        thresholds
        '''
        #max_depth = max([sublist[5] for sublist in self.tree])
 
        # Define the initial value
        #initial_value = 0.5
 
        # Define the number of times to divide
        #num_divisions = max_depth
        #temporary_list = [initial_value / (2 ** i) for i in range(1, num_divisions + 1)]
 
        #flip the list
        #change_per_depth_possibilities = temporary_list[::-1]
        #compute cumsum
        #cumulative_sum = np.cumsum(change_per_depth_possibilities)
 
        #random_number = rng.uniform()
        #random_number = random.random()
 
        #find the index in the cumsum list for the depth we need to change
        #index_found = np.searchsorted(cumulative_sum, random_number)
 
        #find the list of lists we can change
        #indices_with_index_found = [i for i, sublist in enumerate(self.tree) if sublist[5] == index_found]
 
        # Randomly select an index from the list of indices we can change
        #random_index = rng.randomChoice(indices_with_index_found)
        # random_index = random.choice(indices_with_index_found)
 
        # Retrieve the sublist corresponding to the randomly selected index
        #random_sublist = self.tree[random_index]
 
        #findex the index of this random sublist
        #random_sublist_index = self.tree.index(random_sublist)
        random_index = rng.randomInt(0, len(self.tree)-1)
        node_to_change = self.tree[random_index]
        new_feature = rng.randomInt(0, len(self.X_train[0])-1)
        X_data_in_node=self.data_in_node[node_to_change[0]]
        count_X_data_in_node=len(X_data_in_node)
        num_class = len(set(self.y_train[X_data_in_node]))
        if count_X_data_in_node!=0 and num_class >= 2:
            prob_thresh={}
            count_in_1={}
            count_in_2={}
            array=np.array(list(set(self.X_train[:, new_feature])))
            if len(array) < 100:
                set_of_thresh = array
            else:
                set_of_thresh=[np.percentile(array,i) for i in range(1,101)]
            # divide all the threshold in percentile, and use it for selecting the thresholds
            sum_twoing=0
            for thresh in set_of_thresh: 
                #for every threshold compute the twoing coeficient using a random feature
                for clas in self.c:
                    count_in_1[clas]=0
                    count_in_2[clas]=0
                count_1=0
                count_2=0
                for datum in X_data_in_node:
                    if self.X_train[datum,new_feature]>thresh:
                        count_in_1[self.y_train[datum]]+=1
                        count_1+=1
                    else:
                        count_in_2[self.y_train[datum]]+=1
                        count_2+=1
                twoing=0.0;cumulative_sum=0.0
                if count_1!=0 and count_2!=0:
                    prob_1=count_1/count_X_data_in_node
                    prob_2=count_2/count_X_data_in_node
                    for clas in self.c:
                        cumulative_sum+=abs(((count_in_1[clas]/count_1)-(count_in_2[clas]/count_2)))
                    twoing=(prob_1*prob_2/4)*(cumulative_sum**2)
                prob_thresh[thresh]=twoing
                sum_twoing+=twoing
            if sum_twoing!=0:
                if deterministic==0:
                    for thresh in set_of_thresh:
                        prob_thresh[thresh]/=sum_twoing 
                    new_threshold=rng.randomChoices(list(prob_thresh.keys()),weights=list(prob_thresh.values()))
                    self.last_move_prob_f_t=(1/len(self.X_train[0,:]),prob_thresh[new_threshold[0]]) 
                    #save the probability of the selected feature-threshold for eval
                else:
                    best_key=np.argmax(list(prob_thresh.values())) 
                    #select the best spliting in a deterministic way
                    new_threshold=list(prob_thresh.keys())[best_key]
            else:
                new_threshold = rng.randomInt(0, len(self.X_train)-1)
                new_threshold = (self.X_train[new_threshold, new_feature])
                self.last_move_prob_f_t=(1/len(self.X_train[0]),1/len(self.X_train[:])) 
                #save the probability of the selected feature-threshold for eval
            node_to_change[3] = new_feature
            node_to_change[4] = new_threshold
        return self
        
    def change_H_feature(self, rng=RNG(),deterministic=0,min_data=2):
        action = "change"
        self.lastAction = action
        '''
        we need to choose a new feature at first
        we then need to choose a new threshold base on the feature we have
        chosen and then pick unoformly a node and change their features and
        thresholds
        '''
        #max_depth = max([sublist[5] for sublist in self.tree])
 
        # Define the initial value
        #initial_value = 0.5
 
        # Define the number of times to divide
        #num_divisions = max_depth
        #temporary_list = [initial_value / (2 ** i) for i in range(1, num_divisions + 1)]
 
        #flip the list
        #change_per_depth_possibilities = temporary_list[::-1]
        #compute cumsum
        #cumulative_sum = np.cumsum(change_per_depth_possibilities)
 
        #random_number = rng.uniform()
        #random_number = random.random()
 
        #find the index in the cumsum list for the depth we need to change
        #index_found = np.searchsorted(cumulative_sum, random_number)
 
        #find the list of lists we can change
        #indices_with_index_found = [i for i, sublist in enumerate(self.tree) if sublist[5] == index_found]
 
        # Randomly select an index from the list of indices we can change
        #random_index = rng.randomChoice(indices_with_index_found)
        # random_index = random.choice(indices_with_index_found)
 
        # Retrieve the sublist corresponding to the randomly selected index
        #random_sublist = self.tree[random_index]
 
        #findex the index of this random sublist
        #random_sublist_index = self.tree.index(random_sublist)
        random_index = rng.randomInt(0, len(self.tree)-1)
        node_to_change = self.tree[random_index]
        #random_index = rng.randomInt(0, len(self.tree)-1)
        #node_to_change = self.tree[random_index]
        #new_feature = rng.randomInt(0, len(self.X_train[0])-1)
        X_data_in_node=self.data_in_node[node_to_change[0]] #the indexes of the data on the node
        count_X_data_in_node=len(X_data_in_node)
        num_class = len(set(self.y_train[X_data_in_node]))

        if count_X_data_in_node!=0 and num_class >= 2:
            prob_thresh={}
            count_in_1={}
            count_in_2={}
            sum_twoing=0.0
            sum_twoing_feat={}
            for feat in range(len(self.X_train[0])):
                array=np.array(list(set(self.X_train[:, feat])))
                if len(array) < 100:
                    set_of_thresh = array
                else:
                    set_of_thresh=[np.percentile(array,i) for i in range(1,101)]
                # divide all the threshold in percentile, and use it for selecting the thresholds
                sum_twoing_feat[feat]=0.0
                for thresh in set_of_thresh: 
                    #compute the twoing coficient for all the threshold of the current feature
                    for clas in self.c:
                        count_in_1[clas]=0
                        count_in_2[clas]=0
                    count_1=0
                    count_2=0
                    for datum in X_data_in_node:
                        if self.X_train[datum,feat]>thresh:
                            count_in_1[self.y_train[datum]]+=1
                            count_1+=1
                        else:
                            count_in_2[self.y_train[datum]]+=1
                            count_2+=1
                    twoing=0.0;cumulative_sum=0.0
                    if count_1!=0 and count_2!=0:
                        prob_1=count_1/count_X_data_in_node
                        prob_2=count_2/count_X_data_in_node
                        for clas in self.c:
                            cumulative_sum+=abs(((count_in_1[clas]/count_1)-(count_in_2[clas]/count_2))) 
                        twoing=(prob_1*prob_2/4)*(cumulative_sum**2) #Twoing coeficient for the feature-threshold 
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
                                prob_thresh_eval[(feat,thresh)]=(prob_thresh[(feat,thresh)]/sum_twoing_feat[feat])
                            else:
                                prob_thresh_eval[(feat,thresh)]=0.0
                            prob_thresh[(feat,thresh)]/=sum_twoing
                            sum_prob_feat_eval+=prob_thresh[(feat,thresh)]
                        prob_feat_eval[feat]=sum_prob_feat_eval 
                    feat_thresh=rng.randomChoices(list(prob_thresh.keys()),weights=list(prob_thresh.values()))
                    new_threshold=feat_thresh[0][1]
                    new_feature=int(feat_thresh[0][0])
                    self.last_move_prob_f_t=(prob_feat_eval[new_feature],prob_thresh_eval[(new_feature,new_threshold)])
                    #save the probability of the selected feature-threshold for eval
                    
                else:
                    best_key=np.argmax(list(prob_thresh.values()))
                    feat_thresh=list(prob_thresh.keys())[best_key]
                    new_threshold=feat_thresh[1]
                    new_feature=int(feat_thresh[0])
            else:
                new_feature = rng.randomInt(0, len(self.X_train[0])-1)
                new_threshold = rng.randomInt(0, len(self.X_train)-1)
                new_threshold = (self.X_train[new_threshold, new_feature])
                self.last_move_prob_f_t=(1/len(self.X_train[0]),1/len(self.X_train[:]))
        
            node_to_change[3] = new_feature
            node_to_change[4] = new_threshold
        return self

    def grow_H_feature(self, rng=RNG(),deterministic=0,min_data=2):
        action = "grow"
        self.lastAction = action
        '''
        grow tree by just simply creating the individual nodes. each node
        holds their node index, the left and right leaf index, the node
        feature and threshold
        '''
        random_index = rng.randomInt(0, len(self.leafs)-1)
        leaf_to_grow = self.leafs[random_index]

        # generating a random feature
        #feature = rng.randomInt(0, len(self.X_train[0])-1)
        
        X_data_in_leaf=self.data_in_leaf[leaf_to_grow]
        count_X_data_in_leaf=len(X_data_in_leaf)
        if (count_X_data_in_leaf<=min_data):
            return self.change_H_feature(rng=rng)
        if count_X_data_in_leaf!=0:
            prob_thresh={}
            count_in_1={}
            count_in_2={}
            sum_twoing=0.0
            sum_twoing_feat={}
            for feat in range(len(self.X_train[0])):
                array=np.array(list(set(self.X_train[:, feat])))
                if len(array) < 100:
                    set_of_thresh = array
                else:
                    set_of_thresh=[np.percentile(array,i) for i in range(1,101)]
                # divide all the threshold in percentile, and use it for selecting the thresholds
                sum_twoing_feat[feat]=0.0
                for thresh in set_of_thresh:
                    #compute the twoing coficient for all the threshold of the current feature
                    for clas in self.c:
                        count_in_1[clas]=0
                        count_in_2[clas]=0
                    count_1=0
                    count_2=0
                    for datum in X_data_in_leaf:
                        if self.X_train[datum,feat]>thresh:
                            count_in_1[self.y_train[datum]]+=1
                            count_1+=1
                        else:
                            count_in_2[self.y_train[datum]]+=1
                            count_2+=1
                    twoing=0.0;cumulative_sum=0.0
                    if count_1!=0 and count_2!=0:
                        prob_1=count_1/count_X_data_in_leaf
                        prob_2=count_2/count_X_data_in_leaf
                        for clas in self.c:
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
                        array=np.array(list(set(self.X_train[:, feat])))
                        if len(array) < 100:
                            set_of_thresh = array
                        else:
                            set_of_thresh=[np.percentile(array,i) for i in range(1,101)]
                        sum_prob_feat_eval=0.0
                        for thresh in set_of_thresh:
                            if sum_twoing_feat[feat]!=0.0:
                                prob_thresh_eval[(feat,thresh)]=prob_thresh[(feat,thresh)]/sum_twoing_feat[feat]
                            else:
                                prob_thresh_eval[(feat,thresh)]=0.0
                            prob_thresh[(feat,thresh)]/=sum_twoing
                            sum_prob_feat_eval+=prob_thresh[(feat,thresh)]
                        prob_feat_eval[feat]=sum_prob_feat_eval                        
                    feat_thresh=rng.randomChoices(list(prob_thresh.keys()),weights=list(prob_thresh.values()))
                    threshold=feat_thresh[0][1]
                    feature=int(feat_thresh[0][0])
                    self.last_move_prob_f_t=(prob_feat_eval[feature],prob_thresh_eval[(feature,threshold)])
                    #save the probability of the selected feature-threshold for eval
                else:
                    best_key=np.argmax(list(prob_thresh.values()))
                    feat_thresh=list(prob_thresh.keys())[best_key]
                    threshold=feat_thresh[1]
                    feature=int(feat_thresh[0])
            else:
                feature = rng.randomInt(0, len(self.X_train[0])-1)
                threshold = rng.randomInt(0, len(self.X_train)-1)
                threshold = (self.X_train[threshold, feature])
        else:
            feature = rng.randomInt(0, len(self.X_train[0])-1)
            threshold = rng.randomInt(0, len(self.X_train)-1)
            threshold = (self.X_train[threshold, feature])
           
        depth = self.depth_of_leaf(leaf_to_grow)
        node = [leaf_to_grow, max(self.leafs)+1, max(self.leafs)+2, feature,
                threshold, depth, self.parents_leafs[leaf_to_grow]]

        # add the new leafs on the leafs array
        leaf_1=max(self.leafs)+1
        self.leafs.append(leaf_1)
        leaf_2=max(self.leafs)+1
        self.leafs.append(leaf_2)
        self.parents_leafs[leaf_1]=leaf_to_grow
        self.parents_leafs[leaf_2]=leaf_to_grow
        # save the parents of the leafs
        del self.parents_leafs[leaf_to_grow]
        # delete from leafs the new node
        self.leafs.remove(leaf_to_grow)
        self.tree.append(node)
        self.node_last_modification=random_index
        return self
    def change_H_feature_ACO(self,pheromone_matrix,alpha,beta, rng=RNG(),deterministic=0,min_data=2):
        action = "change"
        self.lastAction = action
        '''
        we need to choose a new feature at first
        we then need to choose a new threshold base on the feature we have
        chosen and then pick unoformly a node and change their features and
        thresholds
        '''
        #max_depth = max([sublist[5] for sublist in self.tree])
 
        # Define the initial value
        #initial_value = 0.5
 
        # Define the number of times to divide
        #num_divisions = max_depth
        #temporary_list = [initial_value / (2 ** i) for i in range(1, num_divisions + 1)]
 
        #flip the list
        #change_per_depth_possibilities = temporary_list[::-1]
        #compute cumsum
        #cumulative_sum = np.cumsum(change_per_depth_possibilities)
 
        #random_number = rng.uniform()
        #random_number = random.random()
 
        #find the index in the cumsum list for the depth we need to change
        #index_found = np.searchsorted(cumulative_sum, random_number)
 
        #find the list of lists we can change
        #indices_with_index_found = [i for i, sublist in enumerate(self.tree) if sublist[5] == index_found]
 
        # Randomly select an index from the list of indices we can change
        #random_index = rng.randomChoice(indices_with_index_found)
        # random_index = random.choice(indices_with_index_found)
 
        # Retrieve the sublist corresponding to the randomly selected index
        #random_sublist = self.tree[random_index]
 
        #findex the index of this random sublist
        #random_sublist_index = self.tree.index(random_sublist)
        random_index = rng.randomInt(0, len(self.tree)-1)
        node_to_change = self.tree[random_index]
        self.node_last_modification=node_to_change
        #random_index = rng.randomInt(0, len(self.tree)-1)
        node_to_change = self.tree[random_index]
        feature_to_change = node_to_change[3]
        threshold_to_change = node_to_change[4]
        #new_feature = rng.randomInt(0, len(self.X_train[0])-1)
        X_data_in_node=self.data_in_node[node_to_change[0]]
        num_classes_in_node = len(np.unique(self.y_train[X_data_in_node]))
        # FOR THE ACO
        parent_node_index = node_to_change[6]
        for node in self.tree:
            if node[0] == parent_node_index:
                parent_node=node
                break
        #
        
        count_X_data_in_node=len(X_data_in_node)

        if count_X_data_in_node!=0 and num_classes_in_node>=2:
            prob_thresh={}
            #count_in_1={}
            #count_in_2={}
            sum_twoing=0.0
            sum_twoing_feat={}
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
                    #count_in_1 = np.zeros(len(self.c))
                    #count_in_2 = np.zeros(len(self.c))
                    #compute the twoing coficient for all the threshold of the current feature
                    #for clas in self.c:
                    #    count_in_1[clas]=0
                    #    count_in_2[clas]=0
                    #count_1=0
                    #count_2=0
                    mask = data_feat > thresh
                    count_in_1 = np.bincount(class_data_feat[mask],minlength=max(self.c)+1)
                    count_in_2 = np.bincount(class_data_feat[~mask],minlength=max(self.c)+1)
                    count_1 = len(class_data_feat[mask])
                    count_2 = len(class_data_feat[~mask])
                    #for datum in X_data_in_node:
                    #    if self.X_train[datum,feat]>thresh:
                    #        count_in_1[self.y_train[datum]]+=1
                    #        count_1+=1
                    #    else:
                    #        count_in_2[self.y_train[datum]]+=1
                    #        count_2+=1
                    twoing=0.0;cumulative_sum=0.0
                    if count_1!=0 and count_2!=0:
                        prob_1=count_1/count_X_data_in_node
                        prob_2=count_2/count_X_data_in_node
                        for clas in self.c:
                            cumulative_sum+=abs(((count_in_1[clas]/count_1)-(count_in_2[clas]/count_2)))
                        if parent_node_index == -1: #for the root node we use only the twoing criterion
                            twoing=(prob_1*prob_2/4)*(cumulative_sum**2)
                        else:    
                            twoing=(prob_1*prob_2/4)*(cumulative_sum**2)\
                                    *pheromone_matrix[((parent_node[3],parent_node[4]),(feat,thresh))]
                                    # for the other nodes we use the pheromone for the probability
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
                    new_threshold=feat_thresh[0][1]
                    new_feature=int(feat_thresh[0][0])
                    self.last_move_prob_f_t=(prob_feat_eval[new_feature],prob_thresh_eval[(new_feature,new_threshold)])
                    self.change_reverse[0] = prob_feat_eval[feature_to_change]
                    self.change_reverse[1] = prob_thresh_eval[(feature_to_change,threshold_to_change)]
                    #print(self.change_reverse)
                    if self.change_reverse[0] == 0.0:
                        self.change_reverse[0]=self.small_float 
                    if self.change_reverse[1] == 0.0:
                        self.change_reverse[1]=self.small_float 
                    #save the probability of the selected feature-threshold for eval
                    
                else:
                    best_key=np.argmax(list(prob_thresh.values()))
                    feat_thresh=list(prob_thresh.keys())[best_key]
                    new_threshold=feat_thresh[1]
                    new_feature=int(feat_thresh[0])
            else: 
                return self
            node_to_change[3] = new_feature
            node_to_change[4] = new_threshold
            self.node_last_modification = random_index
        return self
    def grow_H_feature_ACO(self,pheromone_matrix,alpha,beta, rng=RNG(),deterministic=0,min_data=2):
        action = "grow"
        self.lastAction = action
        '''
        grow tree by just simply creating the individual nodes. each node
        holds their node index, the left and right leaf index, the node
        feature and threshold
        '''
        random_index = rng.randomInt(0, len(self.leafs)-1)
        leaf_to_grow = self.leafs[random_index]
        
        # generating a random feature
        #feature = rng.randomInt(0, len(self.X_train[0])-1)
        error_flag=False
        X_data_in_leaf=self.data_in_leaf[leaf_to_grow]
        count_X_data_in_leaf=len(X_data_in_leaf)
        num_clases_in_node = len(np.unique(self.y_train[X_data_in_leaf]))
        if (count_X_data_in_leaf<=min_data or num_clases_in_node<2):
            return self.change_H_feature_ACO(pheromone_matrix,alpha,beta,rng=rng)
        # FOR THE ACO
        parent_node_index = self.parents_leafs[leaf_to_grow] 
        for node in self.tree:
            if node[0] == parent_node_index:
                parent_node=node
                break
        self.node_last_modification = parent_node
        #
        prob_thresh={}
        #count_in_1={}
        #count_in_2={}
        sum_twoing=0.0
        sum_twoing_feat={}
        class_data_feat= np.array(self.y_train[X_data_in_leaf])
        for feat in range(len(self.X_train[0])):
            array=np.array(list(set(self.X_train[:, feat])))
            if len(array) < 100:
                set_of_thresh = array
            else:
                set_of_thresh=[np.percentile(array,i) for i in range(1,101)]
            # divide all the threshold in percentile, and use it for selecting the thresholds
            sum_twoing_feat[feat]=0.0
            data_feat=np.array(self.X_train[X_data_in_leaf,feat])
            for thresh in set_of_thresh:
                #compute the twoing coficient for all the threshold of the current feature
                #count_in_1 = np.zeros(len(self.c))
                #count_in_2 = np.zeros(len(self.c))
                #for clas in self.c:
                #    count_in_1[clas]=0
                #    count_in_2[clas]=0
                #for datum in X_data_in_leaf:
                #    if self.X_train[datum,feat]>thresh:
                #        count_in_1[self.y_train[datum]]+=1
                #        count_1+=1
                #    else:
                #        count_in_2[self.y_train[datum]]+=1
                #        count_2+=1
                mask = data_feat > thresh
                count_in_1 = np.bincount(class_data_feat[mask],minlength=max(self.c)+1)
                count_in_2 = np.bincount(class_data_feat[~mask],minlength=max(self.c)+1)
                count_1 = len(class_data_feat[mask])
                count_2 = len(class_data_feat[~mask])

                twoing=0.0;cumulative_sum=0.0
                if count_1!=0 and count_2!=0:
                    prob_1=count_1/count_X_data_in_leaf
                    prob_2=count_2/count_X_data_in_leaf
                    for clas in self.c:
                        cumulative_sum+=abs(((count_in_1[clas]/count_1)-(count_in_2[clas]/count_2)))
                    if parent_node_index == -1:
                        twoing=(prob_1*prob_2/4)*(pow(cumulative_sum,2))
                    else:    
                        twoing=pow((prob_1*prob_2/4)*(cumulative_sum**2),beta)\
                                *pow(pheromone_matrix[((parent_node[3],parent_node[4]),(feat,thresh))],alpha)
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
                    array=np.array(list(set(self.X_train[:, feat])))
                    if len(array) < 100:
                        set_of_thresh = array
                    else:
                        set_of_thresh=[np.percentile(array,i) for i in range(1,101)]
                    sum_prob_feat_eval=0.0
                    for thresh in set_of_thresh:
                        if sum_twoing_feat[feat]!=0.0:
                            prob_thresh_eval[(feat,thresh)]=prob_thresh[(feat,thresh)]/sum_twoing_feat[feat]
                        else:
                            prob_thresh_eval[(feat,thresh)]=0.0
                        prob_thresh[(feat,thresh)]/=sum_twoing
                        sum_prob_feat_eval+=prob_thresh[(feat,thresh)]
                    prob_feat_eval[feat]=sum_prob_feat_eval                        
                feat_thresh=rng.randomChoices(list(prob_thresh.keys()),weights=list(prob_thresh.values()))
                threshold=feat_thresh[0][1]
                feature=int(feat_thresh[0][0])
                self.last_move_prob_f_t=(prob_feat_eval[feature],prob_thresh_eval[(feature,threshold)])
                #save the probability of the selected feature-threshold for eval
            else:
                best_key=np.argmax(list(prob_thresh.values()))
                feat_thresh=list(prob_thresh.keys())[best_key]
                threshold=feat_thresh[1]
                feature=int(feat_thresh[0])
        else:
            error_flag = True 
            #we can't split the data with our set of thresholds
         
        if not error_flag:
            depth = self.depth_of_leaf(leaf_to_grow)
            node = [leaf_to_grow, max(self.leafs)+1, max(self.leafs)+2, feature,
                    threshold, depth, self.parents_leafs[leaf_to_grow]]

            # add the new leafs on the leafs array
            leaf_1=max(self.leafs)+1
            self.leafs.append(leaf_1)
            leaf_2=max(self.leafs)+1
            self.leafs.append(leaf_2)
            # save the parents of the leafs
            self.parents_leafs[leaf_1]=leaf_to_grow
            self.parents_leafs[leaf_2]=leaf_to_grow
            # delete from leafs the new node
            self.leafs.remove(leaf_to_grow)
            del self.parents_leafs[leaf_to_grow]
            self.tree.append(node)
        return self
    def prune_ACO(self,pheromone_matrix,alpha,beta, rng=RNG(),deterministic=0):
        action = "prune"
        self.lastAction = action
        #print("i am here")
        '''
        For example when we have nodes 0,1,2 and leafs 3,4,5,6 when we prune
        we take the leafs 6 and 5 out, and the
        node 2, now becomes a leaf.
        '''
        new_parents_leafs={}
        candidates = self.prunable_node_indices()
        nc = len(candidates)
        random_index = rng.randomInt(0, nc-1)
        index_to_prune = candidates[random_index]
        node_to_prune = self.tree[index_to_prune]
        # Need to compute the prob of grow for the reverse, 
        parent_node_index = node_to_prune[-1]
        feature_node_to_prune = node_to_prune[3]
        threshold_node_to_prune = node_to_prune[4]
        # while random_index == 0:
        #     random_index = rng.randomInt(0, len(self.tree)-1)
        #     node_to_prune = self.tree[random_index]

        # if (node_to_prune[1] in self.leafs) and\
        #         (node_to_prune[2] in self.leafs):
        #     print("here")
            # remove the pruned leafs from leafs list and add the node as a
            # leaf
        self.leafs.append(node_to_prune[0])
        self.leafs.remove(node_to_prune[1])
        self.leafs.remove(node_to_prune[2])
        del self.parents_leafs[node_to_prune[1]]
        del self.parents_leafs[node_to_prune[2]]
        self.parents_leafs[node_to_prune[0]] = node_to_prune[6]
        # delete the specific node from the node lists
        del self.tree[index_to_prune]
        error_flag=False
        X_data_in_leaf=self.data_in_node[parent_node_index]
        count_X_data_in_leaf=len(X_data_in_leaf)
        num_clases_in_node = len(np.unique(self.y_train[X_data_in_leaf]))
        # FOR THE ACO
        for node in self.tree:
            if node[0] == parent_node_index:
                parent_node=node
                break
        self.node_last_modification = parent_node
        #
        prob_thresh={}
        #count_in_1={}
        #count_in_2={}
        sum_twoing=0.0
        sum_twoing_feat={}
        class_data_feat= np.array(self.y_train[X_data_in_leaf])
        for feat in range(len(self.X_train[0])):
            array=np.array(list(set(self.X_train[:, feat])))
            if len(array) < 100:
                set_of_thresh = array
            else:
                set_of_thresh=[np.percentile(array,i) for i in range(1,101)]
            # divide all the threshold in percentile, and use it for selecting the thresholds
            sum_twoing_feat[feat]=0.0
            data_feat=np.array(self.X_train[X_data_in_leaf,feat])
            for thresh in set_of_thresh:
                #compute the twoing coficient for all the threshold of the current feature
                #count_in_1 = np.zeros(len(self.c))
                #count_in_2 = np.zeros(len(self.c))
                #for clas in self.c:
                #    count_in_1[clas]=0
                #    count_in_2[clas]=0
                #for datum in X_data_in_leaf:
                #    if self.X_train[datum,feat]>thresh:
                #        count_in_1[self.y_train[datum]]+=1
                #        count_1+=1
                #    else:
                #        count_in_2[self.y_train[datum]]+=1
                #        count_2+=1
                mask = data_feat > thresh
                count_in_1 = np.bincount(class_data_feat[mask],minlength=max(self.c)+1)
                count_in_2 = np.bincount(class_data_feat[~mask],minlength=max(self.c)+1)
                count_1 = len(class_data_feat[mask])
                count_2 = len(class_data_feat[~mask])

                twoing=0.0;cumulative_sum=0.0
                if count_1!=0 and count_2!=0:
                    prob_1=count_1/count_X_data_in_leaf
                    prob_2=count_2/count_X_data_in_leaf
                    for clas in self.c:
                        cumulative_sum+=abs(((count_in_1[clas]/count_1)-(count_in_2[clas]/count_2)))
                    if parent_node_index == -1:
                        twoing=(prob_1*prob_2/4)*(pow(cumulative_sum,2))
                    else:    
                        twoing=pow((prob_1*prob_2/4)*(cumulative_sum**2),beta)\
                                *pow(pheromone_matrix[((parent_node[3],parent_node[4]),(feat,thresh))],alpha)
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
                    array=np.array(list(set(self.X_train[:, feat])))
                    if len(array) < 100:
                        set_of_thresh = array
                    else:
                        set_of_thresh=[np.percentile(array,i) for i in range(1,101)]
                    sum_prob_feat_eval=0.0
                    for thresh in set_of_thresh:
                        if sum_twoing_feat[feat]!=0.0:
                            prob_thresh_eval[(feat,thresh)]=prob_thresh[(feat,thresh)]/sum_twoing_feat[feat]
                        else:
                            prob_thresh_eval[(feat,thresh)]=0.0
                        prob_thresh[(feat,thresh)]/=sum_twoing
                        sum_prob_feat_eval+=prob_thresh[(feat,thresh)]
                    prob_feat_eval[feat]=sum_prob_feat_eval                        
                feat_thresh=rng.randomChoices(list(prob_thresh.keys()),weights=list(prob_thresh.values()))
                self.prune_reverse[0] = prob_feat_eval[feature_node_to_prune]
                self.prune_reverse[1] = prob_thresh_eval[(feature_node_to_prune,threshold_node_to_prune)]
                if self.prune_reverse[0] == 0.0:
                    self.prune_reverse[0]=self.small_float 
                if self.prune_reverse[1] == 0.0:
                    self.prune_reverse[1]=self.small_float 
            else:
                best_key=np.argmax(list(prob_thresh.values()))
                feat_thresh=list(prob_thresh.keys())[best_key]
                threshold=feat_thresh[1]
                feature=int(feat_thresh[0])
        else:
            error_flag = True 
            #we can't split the data with our set of thresholds
            
        new_leafs = []
        for node in self.tree:
            count1 = 0
            count2 = 0
            for check_node in self.tree:
                if node[1] == check_node[0]:
                    count1 = 1
                if node[2] == check_node[0]:
                    count2 = 1

            if count1 == 0:
                new_leafs.append(node[1])
                new_parents_leafs[node[1]] = node[0]

            if count2 == 0:
                new_leafs.append(node[2])
                new_parents_leafs[node[2]] = node[0]
        
        
        self.leafs[:] = new_leafs[:]
        self.parents_leafs = new_parents_leafs.copy()
        # print(self.leafs)
        # # print(self.tree)
        # print("***********")
        return self

    def grow_leaf_ACO(self, index, rng=RNG()):
        action = "grow"
        self.lastAction = action
        '''
        grow tree by just simply creating the individual nodes. each node
        holds their node index, the left and right leaf index, the node
        feature and threshold
        '''
        random_index = index
        leaf_to_grow = self.leafs[random_index]

        # generating a random feature
        feature = rng.randomInt(0, len(self.X_train[0])-1)
        # generating a random threshold
        array=np.array(list(set(self.X_train[:, feature])))
        if len(array) < 100:
            set_of_thresh = array
        else:
            set_of_thresh=[np.percentile(array,i) for i in range(1,101)]
        threshold = rng.randomChoice(set_of_thresh)
        depth = self.depth_of_leaf(leaf_to_grow)
        node = [leaf_to_grow, max(self.leafs)+1, max(self.leafs)+2, feature,
                threshold, depth, self.parents_leafs[leaf_to_grow]]

        # add the new leafs on the leafs array
        leaf_1=max(self.leafs)+1
        self.leafs.append(leaf_1)
        leaf_2=max(self.leafs)+1
        self.leafs.append(leaf_2)
        # save the parents of the leafs
        self.parents_leafs[leaf_1]=leaf_to_grow
        self.parents_leafs[leaf_2]=leaf_to_grow
        # delete from leafs the new node
        self.leafs.remove(leaf_to_grow)
        del self.parents_leafs[leaf_to_grow]
        self.tree.append(node)
        return self



import numpy as np
from math import log, inf
import copy
from discretesampling.base.random import RNG
from discretesampling.base.types import DiscreteVariableProposal


class TreeProposal(DiscreteVariableProposal):
    def __init__(self,moves_prob=[0.2499, 0.001, 0.25, 0.5] ):
    # or moves_prob=[0.2499, 0.001, 0.5, 0.25] 
        self.moves_prob = moves_prob  # good for Poisson and heart l = 12, and diabetes l = 10
    @classmethod
    def norm(self, tree):
        return len(tree.tree)

    @classmethod
    # Should return true if proposal is possible between x and y
    # (and possibly at other times)
    def heuristic(self, x, y):
        return y < x or abs(x-y) < 2

    def sample(self, start_tree, heuristic, rng=RNG(), num_nodes=40, pheromone_matrix=None, alpha = 1.0, beta = 1.0):
        '''
        start_tree is the strating tree, from where are sampling a new tree

        heuristic 0 uses uniform distrubtion for sampling features and thresholds,
        heuristic 1 uses uniform distrubtion only for features and the heuristic twoing criterion for thresholds,
        heuristic 2 uses heuristic twoing criterion for features and threholds,
        heuristic 3 uses ACS combining heuristic twoing criterion and pheromones. 
        
        num_nodes is the maximum number of non-terminal nodes. If this number is reached, then the Grow probability is 0

        pheromone_matrix is the structure that contains the pheromones. If heuristic 3 is not selected, then is None

        alpha and beta are the classic ACO parameters. 
        '''

        # self.moves_prob = [0.4, 0.1, 0.1, 0.4] # Good for chipman
        # initialise the probabilities of each move
        moves = ["prune", "swap", "change", "grow"]  # noqa
        moves_prob = self.moves_prob
        if len(start_tree.tree) == 1:
            moves_prob = [0.0, 0.0, 0.5, 0.5]
        #elif len(start_tree.tree) >= num_nodes:
        #    moves_prob = [0.2, 0.0, 0.8, 0.0]
        random_number = rng.random()
        moves_probabilities = np.cumsum(moves_prob)
        newTree = copy.copy(start_tree)
        if random_number < moves_probabilities[0]:
            # prune
            if heuristic == 0:
                newTree = newTree.prune(rng=rng)
            if heuristic == 3:
                newTree = newTree.prune_ACO(pheromone_matrix,alpha,beta,rng=rng)
        elif random_number < moves_probabilities[1]:
            # swap
            newTree = newTree.swap(rng=rng)
        elif random_number < moves_probabilities[2]:
            # change
            if heuristic == 0:
                newTree = newTree.change(rng=rng)
            if heuristic == 1:
                newTree = newTree.change_H_threshold(rng=rng)
            if heuristic == 2:
                newTree = newTree.change_H_feature(rng=rng)
            if heuristic == 3:
                newTree = newTree.change_H_feature_ACO(pheromone_matrix,alpha,beta,rng=rng)

        else:
            # grow
            if heuristic == 0:
                newTree = newTree.grow(rng=rng)
            if heuristic == 1:
                newTree = newTree.grow_H_threshold(rng=rng)
            if heuristic == 2:
                newTree = newTree.grow_H_feature(rng=rng)
            if heuristic == 3:
                newTree = newTree.grow_H_feature_ACO(pheromone_matrix,alpha,beta,rng=rng)

        return newTree
    
    def prunable_node_indices(self, tree):
        '''
        Give the prunable nodes of a tree, which are the nodes with two terminal nodes as sons
        '''
        candidates = []
        for i in range(1, len(tree.tree)): # cannot prune the root
            node_to_prune = tree.tree[i]
            if ((node_to_prune[1] in tree.leafs) and (node_to_prune[2] in tree.leafs)):
                candidates.append(i)
        return(candidates)

    def eval(self, start_tree, sampledTree,heuristic,heuristic_reverse):
        '''
        Compute the forward/reverse probability of a sampled tree

        heuristic chose the way in which the forward/reverse probability is selected
        0, 1, 2, 3.

        heuristic_reverse is 1 when heuristic = 1,2 or 3 and the revserse probability is being computed.
        '''
        initialTree = start_tree
        moves_prob = self.moves_prob
        logprobability = -inf
        if len(initialTree.tree) == 1:
            moves_prob = [0.0, 0.0, 0.5, 0.5]

        nodes_differences = [i for i in sampledTree.tree + initialTree.tree
                             if i not in sampledTree.tree or
                             i not in initialTree.tree]
        # In order to get sampledTree from initialTree we must have:
        # Grow
        if (len(initialTree.tree) == len(sampledTree.tree)-1):
            
            if heuristic==0:
                logprobability = (log(moves_prob[3])
                                - log(len(initialTree.X_train[0]))
                                - log(len(initialTree.X_train[:]))
                                - log(len(initialTree.leafs)))
            else:  
                #using the probability calculated with twoing criterion
                if heuristic_reverse == 0:
                    logprobability = (log(moves_prob[3])
                                    - log(sampledTree.last_move_prob_f_t[0]) 
                                    - log(sampledTree.last_move_prob_f_t[1])
                                    - log(len(initialTree.leafs)))
                else:
                    logprobability = (log(moves_prob[3])
                                    - log(start_tree.prune_reverse[0]) 
                                    - log(start_tree.prune_reverse[1])
                                    - log(len(initialTree.leafs)))
                
        # Prune
        elif (len(initialTree.tree) > len(sampledTree.tree)):
            logprobability = (log(moves_prob[0])
                              - log(len(self.prunable_node_indices(initialTree))))
        # Change
        elif (
            len(initialTree.tree) == len(sampledTree.tree)
            and (
                len(nodes_differences) == 2
                or len(nodes_differences) == 0
            )
        ):
            if heuristic==0:
                logprobability = (log(moves_prob[2])
                                - log(len(initialTree.tree))
                                - log(len(initialTree.X_train[0]))
                                - log(len(initialTree.X_train[:])))
            else:
                #using the probability calculated with twoing criterion
                if heuristic_reverse == 0:
                    logprobability = (log(moves_prob[2])
                                    - log(len(initialTree.tree))
                                    - log(sampledTree.last_move_prob_f_t[0])
                                    - log(sampledTree.last_move_prob_f_t[1]))
                else:
                    #cuado es reverse, sampled tree es el antiguo y start_tree es el nuevo
                    try:
                        logprobability = (log(moves_prob[2])
                                        - log(len(start_tree.tree))
                                        - log(start_tree.change_reverse[0])
                                        - log(start_tree.change_reverse[1]))
                    except: 
                        pass
        # swap
        elif (len(nodes_differences) == 4 and len(initialTree.tree) > 1):
            logprobability = (log(moves_prob[1])
                              - log(len(initialTree.tree))
                              - log(len(initialTree.tree) - 1)
                              + log(2))

        return logprobability




def forward(forward, forward_probability):
    forward.append(forward_probability)
    forward_probability = np.sum(forward)
    return forward_probability


def reverse(forward, reverse_probability):
    reverse_probability = reverse_probability + np.sum(forward)
    return reverse_probability

import copy
import numpy as np
import math
from tqdm.auto import tqdm
from discretesampling.base.random import RNG
from discretesampling.base.executor import Executor
from discretesampling.base.algorithms.smc_components.normalisation import normalise
from discretesampling.base.algorithms.smc_components.effective_sample_size import ess
from discretesampling.base.algorithms.smc_components.resampling import systematic_resampling
from discretesampling.base.algorithms.smc_components.importance_resampling_version3 import importance_resampling_v3
from discretesampling.domain.decision_tree.metrics import stats, accuracy,calculate_leaf_occurences



class DiscreteVariableSMC_ACO():

    def __init__(self, variableType, target, initialProposal,alpha=1,beta=1, proposal=None,
                 Lkernel=None,
                 use_optimal_L=False,
                 exec=Executor()):
        self.variableType = variableType
        self.alpha = alpha
        self.beta = beta 
        self.proposalType = variableType.getProposalType()
        self.proposal = proposal
        if proposal is None:
            self.proposal = self.proposalType()
        self.use_optimal_L = use_optimal_L
        self.exec = exec
        if use_optimal_L:
            self.LKernelType = variableType.getOptimalLKernelType()
        else:
            # By default getLKernelType just returns
            # variableType.getProposalType(), the same as the forward_proposal
            self.LKernelType = variableType.getLKernelType()

        self.Lkernel = Lkernel
        if Lkernel is None:
            self.Lkernel = self.LKernelType()

        self.initialProposal = initialProposal
        self.target = target

    def get_ant_tour(self,Tree):
        #get all the conbinations of feature and threshold on the tree
        tour = []
        for node in Tree.tree:
            if node[6] != -1:
                for node_search in Tree.tree:
                    if node_search[0] == node[6]:
                        parent_feat_threshold = (node_search[3],node_search[4])
                        child_feat_threshold = (node[3],node[4])
                        tour.append((parent_feat_threshold,child_feat_threshold))
        #use the tour for update the pheromone matrix
        return tour
    def update_pheromone(self,pheromone_matrix,tours,pheromone_per_ant,rho):
        number_of_tours=len(tours)
        #evaporte the pheromone 
        for edge in pheromone_matrix.keys():
            pheromone_matrix[edge] *= (1-rho)
        for i,tour in enumerate(tours): # can be an elitist version using only the best ants/trees
            if len(tour) != 0:
                for edge in tour:    
                    pheromone_matrix[edge] += pheromone_per_ant[i]
        return pheromone_matrix




    def sample(self, Tsmc, N,heuristic, seed=0, verbose=True):# HEURISTIC = 0 ORIGINAL, = 1 HERUISTIC THRESHOLD, =2 HEURISTIC FEATURE, =3 ACO
        loc_n = int(N/self.exec.P)
        rank = self.exec.rank
        mvrs_rng = RNG(seed)
        rngs = [RNG(i + rank*loc_n + 1 + seed) for i in range(loc_n)]  # RNG for each particle
        trees_per_it=[]
        eff = []
        if heuristic == 3:
            initialParticles = [self.initialProposal.sample_ACO(rngs[i], self.target) for i in range(loc_n)]
        if heuristic == 0:
            initialParticles = [self.initialProposal.sample(rngs[i], self.target) for i in range(loc_n)]
        current_particles = initialParticles
        logWeights = np.array([self.target.eval(p,heuristic) - self.initialProposal.eval(p, self.target) for p in initialParticles])
        # pheromone matrix
        pheromone_matrix={}
        if heuristic == 3:
            #make the pheromone matrix 
            num_features=len(self.initialProposal.X_train[0])
            percentile_thresholds={}
            for i in range(num_features):
                array=np.array(list(set(self.initialProposal.X_train[:, i])))
                if len(array) < 100:
                    set_of_thresh = array
                else:
                    set_of_thresh=[np.percentile(array,i) for i in range(1,101)]
                percentile_thresholds[i]=set_of_thresh
            labels=stats(current_particles[0],current_particles[0].X_train).predict_for_one_tree(current_particles[0],current_particles[0].X_train)
            acc = accuracy(labels,current_particles[0].y_train)
            acc_weight, tree_len_weight = 1.0, 1.0
            init = (acc/100)*acc_weight + (1/len(current_particles[0].tree))*tree_len_weight # from Boryczka, U., Kozak, J. (2010). Ant Colony Decision Trees 
            #set the initial pheromone in every component of the matrix
            for i in range(num_features):
                for j in percentile_thresholds[i]:
                    for k in range(num_features):
                        for l in percentile_thresholds[k]:
                            pheromone_matrix[((i,j),(k,l))]=init 
                            # pheromone_matrix[(feature_father,threshold_father),(feature_son,threshold_son)]
            pheromone_per_tree=np.zeros(N)
        #
        display_progress_bar = verbose and rank == 0
        progress_bar = tqdm(total=Tsmc, desc="SMC sampling", disable=not display_progress_bar)
        LWeights_Action = [[],[]]
        for t in range(Tsmc):
            tours=[]
            eff.append(logWeights)
            logWeights = normalise(logWeights, self.exec)
            if t!=0:
                LWeights_Action[0].append(logWeights)
                LWeights_Action[1].append([current_particle.lastAction for current_particle in current_particles])
                #print(LWeights_Action[0][t-1])
                #print(LWeights_Action[1][t-1])
            neff = ess(logWeights, self.exec)
            if math.log(neff) < math.log(N) - math.log(2):
                current_particles, logWeights,_ = systematic_resampling(
                    current_particles, np.exp(logWeights), mvrs_rng, N)
                #current_particles, logWeights = importance_resampling_v3(
                #    current_particles, np.exp(logWeights), mvrs_rng, N)

            new_particles = copy.copy(current_particles)
            forward_logprob = np.zeros(len(current_particles))
            # Sample new particles and calculate forward probabilities
            for i in range(loc_n):
                forward_proposal = self.proposal
                #print("tree --------------")
                #print(new_particles[i].tree)
                new_particles[i] = forward_proposal.sample(current_particles[i],heuristic, rng=rngs[i],
                        pheromone_matrix=pheromone_matrix,alpha=self.alpha,beta=self.beta)
                #print(new_particles[i].lastAction)
                #print(new_particles[i].tree)
                #print(new_particles[i].node_last_modification)
                forward_logprob[i] = forward_proposal.eval(current_particles[i], new_particles[i],heuristic,0)
                #print("end tree --------------")


            if self.use_optimal_L:
                Lkernel = self.LKernelType(
                    new_particles, current_particles, parallel=self.exec, num_cores=1
                )
            for i in range(loc_n): #independent for every tree
                if self.use_optimal_L:
                    reverse_logprob = Lkernel.eval(i)
                else:
                    Lkernel = self.Lkernel
                    reverse_logprob = Lkernel.eval(new_particles[i], current_particles[i],0,1) # Changing second to last arg from heuristic to 0 to have L=q_{U} as L-kernel

                current_target_logprob = self.target.eval(current_particles[i],heuristic)
                new_target_logprob = self.target.eval(new_particles[i],heuristic)
                labels=stats(new_particles[i],new_particles[i].X_train).predict_for_one_tree(new_particles[i],new_particles[i].X_train)
                acc = accuracy(labels,new_particles[i].y_train) #get the accuracy for every tree
                if heuristic == 3:
                    tours.append(self.get_ant_tour(new_particles[i])) #get the tours of every ant/tree
                    acc_weight, tree_len_weight = 1.0, 1.0
                    pheromone_per_tree[i]= acc*acc_weight + len(new_particles[i].tree)*tree_len_weight 
                    #compute the pheromone for every tree
                logWeights[i] += new_target_logprob - current_target_logprob + reverse_logprob - forward_logprob[i]
            if heuristic == 3:
                pheromone_matrix=self.update_pheromone(pheromone_matrix,tours,pheromone_per_tree,0.5) #update the pheromone 
            current_particles = new_particles
            trees_per_it.append(new_particles)
            progress_bar.update(1)
        progress_bar.close()
        return current_particles,trees_per_it, LWeights_Action

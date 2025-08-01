import copy
import numpy as np
import math
from tqdm.auto import tqdm
from discretesampling.base.random import RNG
from discretesampling.base.executor import Executor
from discretesampling.base.algorithms.smc_components.normalisation import normalise
from discretesampling.base.algorithms.smc_components.effective_sample_size import ess
from discretesampling.base.algorithms.smc_components.resampling import systematic_resampling


class DiscreteVariableSMC():

    def __init__(self, variableType, target, initialProposal, proposal=None,
                 Lkernel=None,
                 use_optimal_L=False,
                 exec=Executor()):
        self.variableType = variableType
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

    def sample(self, Tsmc, N,heuristic, seed=0, verbose=True):# HEURISTIC = 0 ORIGINAL, = 1 HERUISTIC THRESHOLD, =2 HEURISTIC FEATURE
        loc_n = int(N/self.exec.P)
        rank = self.exec.rank
        mvrs_rng = RNG(seed)
        rngs = [RNG(i + rank*loc_n + 1 + seed) for i in range(loc_n)]  # RNG for each particle
        trees_per_it=[]
        weights=[]
        initialParticles = [self.initialProposal.sample(rngs[i], self.target) for i in range(loc_n)]
        current_particles = initialParticles
        logWeights = np.array([self.target.eval(p,heuristic) - self.initialProposal.eval(p, self.target) for p in initialParticles])
        eff = []

        display_progress_bar = verbose and rank == 0
        progress_bar = tqdm(total=Tsmc, desc="SMC sampling", disable=not display_progress_bar)

        for t in range(Tsmc):
            logWeights = normalise(logWeights, self.exec)
            neff = ess(logWeights, self.exec)
            eff.append(neff)
            if math.log(neff) < math.log(N) - math.log(2):
                current_particles, logWeights = systematic_resampling(
                    current_particles, logWeights, mvrs_rng, exec=self.exec)

            new_particles = copy.copy(current_particles)

            forward_logprob = np.zeros(len(current_particles))

            # Sample new particles and calculate forward probabilities
            for i in range(loc_n):
                forward_proposal = self.proposal
                new_particles[i] = forward_proposal.sample(current_particles[i],heuristic, rng=rngs[i])
                forward_logprob[i] = forward_proposal.eval(current_particles[i], new_particles[i],heuristic=heuristic)

            if self.use_optimal_L:
                Lkernel = self.LKernelType(
                    new_particles, current_particles, parallel=self.exec, num_cores=1
                )
            weights.append([])
            for i in range(loc_n):
                if self.use_optimal_L:
                    reverse_logprob = Lkernel.eval(i)
                else:
                    Lkernel = self.Lkernel
                    reverse_logprob = Lkernel.eval(new_particles[i], current_particles[i],heuristic)

                current_target_logprob = self.target.eval(current_particles[i],heuristic)
                new_target_logprob = self.target.eval(new_particles[i],heuristic)
                weights[t].append(new_target_logprob)
                logWeights[i] += new_target_logprob - current_target_logprob + reverse_logprob - forward_logprob[i]
            current_particles = new_particles
            trees_per_it.append(new_particles)
            progress_bar.update(1)

        progress_bar.close()
        return current_particles,trees_per_it,eff

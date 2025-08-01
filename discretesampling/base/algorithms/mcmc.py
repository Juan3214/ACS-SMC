import math
import copy
from tqdm.auto import tqdm
from discretesampling.base.random import RNG


class DiscreteVariableMCMC():

    def __init__(self, variableType, target, initialProposal, proposal=None):
        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.proposal = proposal
        if proposal is None:
            self.proposal = self.proposalType()
        self.initialProposal = initialProposal
        self.target = target

    def sample(self, N, seed=0, verbose=True):
        rng = RNG(seed)
        initialSample = self.initialProposal.sample(rng)
        current = initialSample
        self.target.eval(current)
        samples = []

        display_progress_bar = verbose
        progress_bar = tqdm(total=N, desc="MCMC sampling", disable=not display_progress_bar)

        for i in range(N):
            forward_proposal = self.proposal
            proposed = forward_proposal.sample(current, rng=rng)

            reverse_proposal = self.proposal

            forward_logprob = forward_proposal.eval(current, proposed)
            reverse_logprob = reverse_proposal.eval(proposed, current)

            current_target_logprob = self.target.eval(current)
            proposed_target_logprob = self.target.eval(proposed)

            log_acceptance_ratio = proposed_target_logprob -\
                current_target_logprob + reverse_logprob - forward_logprob
            if log_acceptance_ratio > 0:
                log_acceptance_ratio = 0
            acceptance_probability = min(1, math.exp(log_acceptance_ratio))

            q = rng.random()
            # Accept/Reject
            if (q < acceptance_probability):
                current = proposed
            else:
                # Do nothing
                pass

            samples.append(copy.copy(current))
            progress_bar.update(1)

        progress_bar.close()
        return samples

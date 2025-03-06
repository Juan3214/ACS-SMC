import numpy as np
from discretesampling.base.random import RNG
from discretesampling.base.executor import Executor


def check_stability(ncopies, exec=Executor()):

    loc_n = len(ncopies)
    N = loc_n * exec.P
    rank = exec.rank

    sum_of_ncopies = exec.sum(ncopies)

    if sum_of_ncopies != N:
        # Find the index of the last particle to be copied
        idx = np.where(ncopies > 0)
        idx = idx[0][-1]+rank*loc_n if len(idx[0]) > 0 else np.array([-1])
        max_idx = exec.max(idx)
        # Find the core which has that particle, and increase/decrease its ncopies[i] till sum_of_ncopies == N
        if rank*loc_n <= max_idx <= (rank + 1)*loc_n - 1:
            ncopies[max_idx - rank*loc_n] -= sum_of_ncopies - N

    return ncopies


def get_number_of_copies(logw, rng=RNG(), exec=Executor()):
    N = len(logw) * exec.P

    cdf = exec.cumsum(np.exp(logw)*N)
    cdf_of_i_minus_one = cdf - np.reshape(np.exp(logw) * N, newshape=cdf.shape)

    u = np.array(rng.uniform(0.0, 1.0), dtype=logw.dtype)
    exec.bcast(u)
    ncopies = (np.ceil(cdf - u) - np.ceil(cdf_of_i_minus_one - u)).astype(int)
    ncopies = check_stability(ncopies, exec)

    return ncopies  # .astype(int)


def systematic_resampling(x, w,  mvrs_rng , N=None):
    if N is None:
        N = len(w)

    N = int(N)

    x_new = []
    w_new = np.ones(N) / N
    log_w_new = np.log(w_new)

    cw = np.cumsum(w)

    u = mvrs_rng.uniform()

    ncopies = np.zeros(len(x), dtype=int)

    for i in range(N):
        j = 0
        while cw[j] < (i + u) / N:
            j += 1

        x_new.append(x[j])
        ncopies[j] += 1

    return x_new, log_w_new, ncopies

def systematic_resampling_old(particles, logw, rng, exec=Executor()):
    loc_n = len(logw)
    N = loc_n * exec.P

    ncopies = get_number_of_copies(logw.astype('float32'), rng, exec)
    particles = exec.redistribute(particles, ncopies)
    logw = np.log(np.ones(loc_n) / N)

    return particles, logw

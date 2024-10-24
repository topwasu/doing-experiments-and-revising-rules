import numpy as np
from numpy.random import random

def systematic_resample(weights, N):
    """ Performs the systemic resampling algorithm used by particle filters.

    This algorithm separates the sample space into N divisions. A single random
    offset is used to to choose where to sample from for all divisions. This
    guarantees that every sample is exactly 1/N apart.

    Parameters
    ----------
    weights : list-like of float
        list of weights as floats

    Returns
    -------

    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (random() + np.arange(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def feedback_generator(structures, labels, possible_labels): 
    txt = ""
    for idx, (structure, label) in enumerate(zip(structures, labels)):
        # flip_label = 'yes' if label == 'no' else 'no'
        flip_label = possible_labels[0] if label == possible_labels[1] else possible_labels[1]
        # actual_output = 'good (follows the rule)' if label == 'yes' else 'bad (does NOT follow the rule)'
        # correct_output = 'good (follows the rule)' if label == 'no' else 'bad (does NOT follow the rule)'
        txt += f"{idx + 1}. {structure.to_text()}"
        txt += f"Correct output: {label}\n"
        txt += f"Rule's output: {flip_label}\n\n"
    return txt

def find_c(weights, N):
    # Sort the weights
    sorted_weights = np.sort(weights)
    # Find the smallest chi
    B_val = 0.0
    A_val = len(weights)
    for i in range(len(sorted_weights)):
        chi = sorted_weights[i]
        # Calculate A_val -- number of weights larger than chi
        A_val -= 1
        # Update B_val -- add the sum of weights smaller than or equal to chi
        B_val += chi
        if B_val / chi + A_val - N <= 1e-12:
            return (N - A_val) / B_val
    return N


# Taken from https://github.com/probcomp/hfppl/blob/main/hfppl/inference/smc_steer.py
def resample_optimal(weights, N):
    c = find_c(weights, N)
    # Weights for which c * w >= 1 are deterministically resampled
    deterministic = np.where(c * weights >= 1)[0]
    # Weights for which c * w <= 1 are stochastically resampled
    stochastic = np.where(c * weights < 1)[0]
    # Stratified sampling to generate N-len(deterministic) indices
    # from the stochastic weights
    n_stochastic = len(stochastic)
    n_resample = N - len(deterministic)
    if n_resample == 0:
        return deterministic, np.array([], dtype=int), c
    K = np.sum(weights[stochastic]) / (n_resample)
    u = np.random.uniform(0, K)
    i = 0
    stoch_resampled = np.array([], dtype=int)
    while i < n_stochastic:
        u = u - weights[stochastic[i]]
        if u <= 0:
            # Add stochastic[i] to resampled indices
            stoch_resampled = np.append(stoch_resampled, stochastic[i])
            # Update u
            u = u + K
            i = i + 1
        else:
            i += 1

    # return deterministic, stoch_resampled
    
    # Concatenate the deterministic and stochastic resampled indices
    resampled = np.concatenate((deterministic, stoch_resampled))
    return resampled
import multiprocessing
import time
from collections import defaultdict

import numba
from numba import prange
# from numba_progress import ProgressBar

import numpy as np
from tqdm import tqdm

def calc_weights_fast(matrix_mapped, identity_threshold, empty_value, num_cpus=1):
    """Rapidly calculates sequence weights using optional parallelization.

    Computes weights by clustering sequences based on sequence identity.
    Handles empty sequences and provides parallel computation option.

    Args:
        matrix_mapped (numpy.ndarray): NxL matrix of N sequences of length L,
            mapped to numerical values
        identity_threshold (float): Sequences with identity above this
            threshold are clustered together
        empty_value (int): Value indicating gaps or invalid characters
        num_cpus (int, optional): Number of CPUs for parallel computation.
            Defaults to 1 (serial)

    Returns:
        numpy.ndarray: Array of sequence weights

    Note:
        Adapted from EVCouplings (https://github.com/debbiemarkslab/EVcouplings).
        When num_cpus > 1, uses Numba parallel computation. On clusters,
        available CPUs may be less than multiprocessing.cpu_count().
    """
    empty_idx = is_empty_sequence_matrix(matrix_mapped, empty_value=empty_value)  # e.g. sequences with just gaps or lowercase, no valid AAs
    N = matrix_mapped.shape[0]

    # Original EVCouplings code structure, plus gap handling
    if num_cpus != 1:
        # print("Calculating weights using Numba parallel (experimental) since num_cpus > 1. If you want to disable multiprocessing set num_cpus=1.")
        # print("Default number of threads for Numba:", numba.config.NUMBA_NUM_THREADS)

        # num_cpus > numba.config.NUMBA_NUM_THREADS will give an error.
        # But we'll leave it so that the user has to be explicit.
        numba.set_num_threads(num_cpus)
        print("Set number of threads to:", numba.get_num_threads())  # Sometimes Numba uses all the CPUs anyway

        num_cluster_members = calc_num_cluster_members_nogaps_parallel(matrix_mapped[~empty_idx], identity_threshold,
                                                                       invalid_value=empty_value)

    else:
        # Use the serial version
        num_cluster_members = calc_num_cluster_members_nogaps(matrix_mapped[~empty_idx], identity_threshold,
                                                              invalid_value=empty_value)

    # Empty sequences: weight 0
    weights = np.zeros((N))
    weights[~empty_idx] = 1.0 / num_cluster_members
    return weights

# Below are util functions copied from EVCouplings
def is_empty_sequence_matrix(matrix, empty_value):
    """Identifies empty sequences in a matrix.

    A sequence is considered empty if all positions contain the empty_value.
    Used to identify sequences that should be excluded from weight calculations.

    Args:
        matrix (numpy.ndarray): 2D matrix of sequences
        empty_value (int or float): Value indicating empty position

    Returns:
        numpy.ndarray: Boolean array indicating empty sequences (True for empty)

    Raises:
        AssertionError: If matrix is not 2D or empty_value is not numeric
    """
    assert len(matrix.shape) == 2, f"Matrix must be 2D; shape={matrix.shape}"
    assert isinstance(empty_value, (int, float)), f"empty_value must be a number; type={type(empty_value)}"
    empty_idx = np.all((matrix == empty_value), axis=1)
    return empty_idx


def map_from_alphabet(alphabet, default):
    """Creates a mapping dictionary from a given alphabet.

    Maps characters to indices, with a default value for unmapped characters.

    Args:
        alphabet (str): Characters to map to indices
        default (str): Character to use for unmapped values.
            Must be present in alphabet.

    Returns:
        defaultdict: Mapping from characters to indices

    Raises:
        ValueError: If default character is not in alphabet
    """
    map_ = {
        c: i for i, c in enumerate(alphabet)
    }

    try:
        default = map_[default]
    except KeyError:
        raise ValueError(
            "Default {} is not in alphabet {}".format(default, alphabet)
        )

    return defaultdict(lambda: default, map_)



def map_matrix(matrix, map_):
    """Maps elements in a matrix using a character mapping.

    Args:
        matrix (numpy.ndarray): Matrix to remap
        map_ (defaultdict): Character to index mapping

    Returns:
        numpy.ndarray: Matrix with characters mapped to indices
    """
    return np.vectorize(map_.__getitem__)(matrix)


# Fastmath should be safe here, as we can assume that there are no NaNs in the input etc.
@numba.jit(nopython=True, fastmath=True)  #parallel=True
def calc_num_cluster_members_nogaps(matrix, identity_threshold, invalid_value):
    """Calculates cluster sizes excluding gaps from identity calculations.

    Serial implementation of sequence clustering based on identity threshold.
    Modified from EVCouplings to use non-gapped length and exclude gaps from matches.

    Args:
        matrix (numpy.ndarray): NxL matrix of N sequences of length L,
            mapped to numerical values using map_matrix
        identity_threshold (float): Identity threshold for sequence clustering
        invalid_value (int): Value indicating gaps or invalid positions

    Returns:
        numpy.ndarray: Number of cluster members for each sequence
        (inverse of sequence weight)

    Note:
        Adapted from EVCouplings with modifications for gap handling:
        - Uses non-gapped length for identity calculation
        - Excludes gaps from sequence similarity matches
        - Optimized with Numba no-Python mode
    """
    N, L = matrix.shape
    L = 1.0 * L

    # Empty sequences are filtered out before this function and are ignored
    # minimal cluster size is 1 (self)
    num_neighbors = np.ones((N))
    L_non_gaps = L - np.sum(matrix == invalid_value, axis=1)  # Edit: From EVE, use the non-gapped length
    # compare all pairs of sequences
    for i in range(N - 1):
        for j in range(i + 1, N):
            pair_matches = 0
            for k in range(L):
                # Edit(Lood): Don't count gaps as matches
                if matrix[i, k] == matrix[j, k] and matrix[i, k] != invalid_value:
                    pair_matches += 1

            # Edit(Lood): Calculate identity as fraction of non-gapped positions (so asymmetric)
            # Note: Changed >= to > to match EVE / DeepSequence code
            if pair_matches / L_non_gaps[i] > identity_threshold:
                num_neighbors[i] += 1
            if pair_matches / L_non_gaps[j] > identity_threshold:
                num_neighbors[j] += 1

    return num_neighbors


@numba.jit(nopython=True, fastmath=True, parallel=True)
def calc_num_cluster_members_nogaps_parallel(matrix, identity_threshold, invalid_value):
    """Calculates cluster sizes in parallel excluding gaps from identity calculations.

    Parallel implementation of sequence clustering based on identity threshold.
    Uses Numba for parallelization and optimized computation.

    Args:
        matrix (numpy.ndarray): NxL matrix of N sequences of length L,
            mapped to numerical values using map_matrix
        identity_threshold (float): Identity threshold for sequence clustering
        invalid_value (int): Value indicating gaps or invalid positions

    Returns:
        numpy.ndarray: Number of cluster members for each sequence
        (inverse of sequence weight)

    Note:
        - Uses non-gapped positions for identity calculation
        - Employs asymmetric similarity calculation
        - Optimized with Numba parallel processing
    """
    N, L = matrix.shape
    L = 1.0 * L

    # Empty sequences are filtered out before this function and are ignored
    # minimal cluster size is 1 (self)
    num_neighbors = np.ones((N))
    L_non_gaps = L - np.sum(matrix == invalid_value, axis=1)  # Edit: From EVE, use the non-gapped length
    # compare all pairs of sequences
    # Edit: Rewrote loop without any dependencies between inner and outer loops, so that it can be parallelized
    for i in prange(N):
        num_neighbors_i = 1
        for j in range(N):
            if i == j:
                continue
            pair_matches = 0
            for k in range(L):  # This should hopefully be vectorised by numba
                # Edit(Lood): Don't count gaps as matches
                if matrix[i, k] == matrix[j, k] and matrix[i, k] != invalid_value:
                    pair_matches += 1

            # Edit(Lood): Calculate identity as fraction of non-gapped positions (so this similarity is asymmetric)
            # Note: Changed >= to > to match EVE / DeepSequence code
            if pair_matches / L_non_gaps[i] > identity_threshold:
                num_neighbors_i += 1

        num_neighbors[i] = num_neighbors_i

    return num_neighbors

@numba.jit(nopython=True, fastmath=True, parallel=True)
def calc_num_cluster_members_nogaps_parallel_print(matrix, identity_threshold, invalid_value, progress_proxy=None, update_frequency=1000):
    """Parallel cluster size calculation with progress tracking.

    Version of calc_num_cluster_members_nogaps_parallel that includes progress
    tracking for long-running calculations.

    Args:
        matrix (numpy.ndarray): NxL matrix of N sequences of length L
        identity_threshold (float): Identity threshold for clustering
        invalid_value (int): Value indicating gaps/invalid positions
        progress_proxy (numba_progress.ProgressBar, optional): Progress bar handle.
            Defaults to None
        update_frequency (int, optional): Progress update interval.
            Defaults to 1000

    Returns:
        numpy.ndarray: Number of cluster members for each sequence

    Note:
        Particularly useful for multi-hour weight calculations on large MSAs.
    """

    N, L = matrix.shape
    L = 1.0 * L

    # Empty sequences are filtered out before this function and are ignored
    # minimal cluster size is 1 (self)
    num_neighbors = np.ones((N))
    L_non_gaps = L - np.sum(matrix == invalid_value, axis=1)  # Edit: From EVE, use the non-gapped length
    # compare all pairs of sequences
    # Edit: Rewrote loop without any dependencies between inner and outer loops, so that it can be parallelized
    for i in prange(N):
        num_neighbors_i = 1
        for j in range(N):
            if i == j:
                continue
            pair_matches = 0
            for k in range(L):  # This should hopefully be vectorised by numba
                # Edit(Lood): Don't count gaps as matches
                if matrix[i, k] == matrix[j, k] and matrix[i, k] != invalid_value:
                    pair_matches += 1
            # Edit(Lood): Calculate identity as fraction of non-gapped positions (so this similarity is asymmetric)
            # Note: Changed >= to > to match EVE / DeepSequence code
            if pair_matches / L_non_gaps[i] > identity_threshold:
                num_neighbors_i += 1

        num_neighbors[i] = num_neighbors_i
        if progress_proxy is not None and i % update_frequency == 0:
            progress_proxy.update(update_frequency)

    return num_neighbors

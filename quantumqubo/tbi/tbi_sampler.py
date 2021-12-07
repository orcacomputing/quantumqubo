import math
import numba as nb
import numpy as np
from numba.typed import List
import time
from collections import Counter


class TBISampler():
    """
    A class that is used to efficiently sample from a single loop PT-Series time bin interferometer (TBI)

    Methods:
        sample(theta_list, n_samples=1): returns samples from the TBI output
    """


    def sample(self, input_state, theta_list, n_samples=1):
        """Returns samples from the output of a single loop TBI. Calls a fast numba implementation

        Args:
            input_state: (list) input modes. The left-most entry corresponds to the first mode entering the loop
            theta_list (List[float]): List of beam splitter angles
            n_samples (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
            Dict: a dictionary of the form state: counts
        """
        
        # Python lists as arguments are deprecated in Numba
        input_state = List(input_state)
        theta_list = List(theta_list)

        samples = self.jit_sampler(theta_list, input_state, n_samples)
        samples = [tuple(sample) for sample in samples]

        return dict(Counter(samples))

    @staticmethod
    @nb.jit
    def jit_sampler(theta_list, input_state, n_samples):
        samples = []
        for k in range(n_samples):
            new_state = _get_one_sample(input_state, theta_list)
            samples.append(new_state)
        return samples


# Table of values of n! for n up to 20
FACTORIAL_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')


@nb.jit
def _fast_factorial(n):
    if n > 20:  # We never get 20 photons in the loop at the same time
        raise ValueError
    return FACTORIAL_TABLE[n]


@nb.njit
def _rand_choice_nb(arr, prob):
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


@nb.jit
def _calculate_prefactor(n, m, k, p):
    """Calculates numerical factor used in expression of amplitudes at output of a beam splitter with input |n,m>"""

    if k > n or p > m:
        raise Exception("Incorrect inputs in amplitude calculation.")

    term1 = np.sqrt(_fast_factorial(k + p)) * np.sqrt(_fast_factorial(n + m - p - k))
    term2 = np.sqrt(_fast_factorial(n)) * np.sqrt(_fast_factorial(m))
    term3 = _fast_factorial(n) * _fast_factorial(m) / (
                _fast_factorial(k) * _fast_factorial(n - k) * _fast_factorial(p) * _fast_factorial(m - p))
    return term1 * term3 / term2


@nb.jit
def _calculate_probs_after_bs(n, m, theta):

    if m > 1:
        raise Exception('Cannot receive more than one photon per input mode')

    amplitudes = []
    choices_output = []

    for k in range(n + 1):
        for p in range(m + 1):
            if k + p not in choices_output:
                amplitudes.append(_calculate_prefactor(n, m, k, p) * ((-1) ** p) * (np.cos(theta) ** (m - p + k)) * (
                            np.sin(theta) ** (n - k + p)))
                choices_output.append(k + p)
            else:
                amplitudes[-1] += _calculate_prefactor(n, m, k, p) * ((-1) ** p) * (np.cos(theta) ** (m - p + k)) * (
                            np.sin(theta) ** (n - k + p))

    probabilities = np.array(amplitudes) ** 2

    return np.array(choices_output), probabilities


@nb.jit
def _get_one_sample(input_list, theta_list):
    n_photon_loop = 0
    N = len(theta_list)
    sample = []
    for i in range(N):
        choices_output, probabilities = _calculate_probs_after_bs(n_photon_loop, input_list[i], theta_list[i])

        if abs(np.sum(probabilities) - 1) > 0.0001:
            raise Exception('Probabilities do not sum to one')

        probabilities = probabilities/np.sum(probabilities)

        output_photons = _rand_choice_nb(choices_output, probabilities)
        n_photon_loop = n_photon_loop + input_list[i] - output_photons

        sample.append(output_photons)

    sample.append(n_photon_loop)

    return sample


if __name__ == '__main__':
    N = 30  # number of photons
    input_state = np.array([1]*N)
    theta_list = np.array([np.pi / 4.5] * N)
    tbi = TBISampler()

    start = time.time()
    res = tbi.sample(input_state, theta_list, n_samples=1000)
    end = time.time()

    print('time = ', end - start)  # in sec
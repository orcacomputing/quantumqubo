import numpy as np
import torch
import matplotlib.pyplot as plt
import numba as nb


def parity_mapping(state, parity):
    '''
    Function that does the following parity mapping:

    Args:
        state: a numpy array representing the number of photons in each mode.
        parity: type of parity mapping done (0 or 1)

    Returns:
        a bit string
    '''

    bit_string = [(ni+parity) % 2 for ni in state]

    return bit_string


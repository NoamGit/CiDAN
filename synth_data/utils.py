import numpy as np
import pandas as pd


def fano_factor(interval_vector):
    """
    Calculates Fano factor according to
    :param interval_vector:
    :return:
    """
    raise NotImplementedError


def coefficient_of_variation(interval_vector):
    """
    Calculates coefficient_of_variation according to $$\frac{E[\tau]}{Var[\tau]}
    :param interval_vector:
    :return:
    """
    return np.mean(interval_vector.flatten()) / np.std(interval_vector.flatten())

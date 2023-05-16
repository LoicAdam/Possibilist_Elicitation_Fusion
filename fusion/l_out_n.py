# -*- coding: utf-8 -*-
"""This module gives tools to do l-out-of-k fusion as shown in paper."""

import itertools
import numpy as np
from elicitation.fusion import tnorm, tconorm

def find_incorrect_answers(polytope_list):
    """
    Determine all the incorrect answers in a list of polytopes.

    Parameters
    ----------
    polytope_list : list
        List of polytopes.

    Returns
    -------
    list
        List of incorrect answers.

    """
    all_detected_incorrect_answers = []
    for polytope in polytope_list:
        detected_incorrect_answers = len(np.where(np.asarray(polytope.get_answers()) < 1)[0])
        all_detected_incorrect_answers.append(detected_incorrect_answers)
    return all_detected_incorrect_answers

def k_among_n_fusion(polytope_list, k, n):
    """
    l-out-of-k fusion as shown in the paper.

    Parameters
    ----------
    polytope_list : list
        List of polytopes..
    k : interger
        Number of correct answers.
    n : interger
        Number of total answers.

    Returns
    -------
    array_like
        The new confidence degrees.

    """
    tconorms = []
    for polytope in polytope_list:
        answers = np.asarray(polytope.get_answers())
        combs_k = itertools.combinations(range(0,n),k)
        tnorms = []
        for comb_k in combs_k:
            selected_answers = answers[list(comb_k)]
            tnorms.append(tnorm(selected_answers))
        tconorms.append(tconorm(tnorms))
    return np.asarray(tconorms)

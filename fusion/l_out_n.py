# -*- coding: utf-8 -*-
"""This module gives tools to do l-out-of-k fusion as shown in paper."""

import itertools
import numpy as np
from elicitation.fusion import tnorm, tconorm
from elicitation.regret_calculation import pmr_polytope
from elicitation.focal_set import compute_epmr_emr
from elicitation.regret_strategies import PessimistStrategy

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
        Number of incorrect answers.
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

def new_optimal_recommendation(polytope_list, possibility_list, alternatives,
                               model):
    """
    Determine the optimal recommendation.

    Parameters
    ----------
    polytope_list : list
        List of polytopes.
    possibility_list : list
        List of possibility for each polytope.
    alternatives : array_like
        Alternatives.
    model : Model
        The model.

    Returns
    -------
    dict
        Information about the recommended alternative.

    """
    pmr_list = []
    for polytope in polytope_list:                    
        pmr = pmr_polytope(alternatives, polytope, model)
        pmr_list.append(pmr)
    _, emr = compute_epmr_emr(pmr_list, possibility_list, 'zero')
    scores = model.get_model_score(alternatives)
    regret_strategy = PessimistStrategy(alternatives)
    best_alt, best_alt_id, regret = regret_strategy.get_best_alternative(emr)
    mmr_real = np.max(scores) - scores[best_alt_id]
    memr_estimated = regret
    result = {}
    result['best_alternative'] = best_alt
    result['memr_estimated'] = memr_estimated
    result['mmr_real'] = mmr_real
    result['inconsistency'] = 0
    return result

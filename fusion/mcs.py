# -*- coding: utf-8 -*-
"""This module gives tools to have the 'best' MCS."""

import itertools
import numpy as np
from elicitation.fusion import tnorm

def get_answers(polytope_list, nb_questions):
    """
    Get all the answers from all the polytopes.

    Parameters
    ----------
    polytope_list : list
        List of polytopes.
    nb_questions : integer
        Number of questions.

    Returns
    -------
    list
        All the answers.

    """
    all_answers = np.zeros((len(polytope_list), nb_questions))
    for i in range(0, len(polytope_list)):
        all_answers[i,:] = polytope_list[i].get_answers()
    return all_answers

def is_subset_element_in_list(element, list_to_check):
    '''
    Check if an element is inside the list.

    Parameters
    ----------
    element : tuple
        Element.
    list_to_check : list of tuple
        List of elements.

    Returns
    -------
    bool
        Is inside the list or not.

    '''
    for element_to_check in list_to_check:
        if set(element).issubset(set(element_to_check)):
            return True
    return False

def find_all_maximum_coherent_subsets(answers, n):
    """
    Find all the coherent subsets regardless of size.

    Parameters
    ----------
    answers : list
        The answers.
    n : integer
        The number of answers.

    Returns
    -------
    list
        List of coherent subsets.

    """
    mcs_list = []
    combs_k = list(itertools.chain(*[itertools.combinations(range(0,n),k) for k in range(n,0,-1)]))
    for comb_k in combs_k:
        selected_answers = answers[:,list(comb_k)]
        if (selected_answers == np.ones(len(comb_k))).all(1).any():
            if is_subset_element_in_list(comb_k, mcs_list) is False:
                mcs_list.append(list(comb_k))
    return mcs_list

def find_coherent_subsets(answers, k, n):
    """
    Find all the coherent subsets of a given size.

    Parameters
    ----------
    answers : list
        The answers.
    k : integer
        The size of coherent subset we want.
    n : integer
        The number of answers.

    Returns
    -------
    list
        List of coherent subsets.

    """
    cs_list = []
    combs_k = itertools.combinations(range(0,n),k)
    for comb_k in combs_k:
        for answers_poly in answers:
            selected_answers = answers_poly[list(comb_k)]
            if np.min(selected_answers) == 1:
                flag = 0
                for l in cs_list:
                    if set(comb_k).issubset(set(l)):
                        flag = 1
                        break
                if flag == 0:
                    cs_list.append(list(comb_k))
    return cs_list

def find_best_cs(cs_list, conf_list):
    """
    Find the best coherent subset according the paper.

    Parameters
    ----------
    cs : list
        List of coherent subsets.
    conf : list
        The confidence degrees.

    Returns
    -------
    array_like
        The best coherent subset.

    """
    if len(cs_list) == 1:
        return cs_list[0]
    confidence_answers = conf_list[cs_list]
    average_conf = np.mean(confidence_answers, axis = 1)
    best_cs = cs_list[np.argmax(average_conf)]
    return best_cs

def update_possibility_list(all_answers, best_cs, tnorm_rule = "product"):
    """
    Update the list of possibikity according to the best coherent subset.

    Parameters
    ----------
    all_answers : list
        List of answers.
    best_cs : array_like
        Best coherent subset.
    tnorm_rule : string, optional
        The T-norm to use. The default is "product".

    Returns
    -------
    list
        The updated confidence degrees list.
    """
    nb_polytopes = all_answers.shape[0]
    possibility_list = np.zeros(nb_polytopes)
    subset_answers = all_answers[:, best_cs]
    for i in range(0, nb_polytopes):
        possibility_list[i] = tnorm(subset_answers[i,:], tnorm_rule)
    return possibility_list

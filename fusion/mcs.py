# -*- coding: utf-8 -*-
"""This module gives tools to have the 'best' MCS."""

import itertools
import copy
import numpy as np

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

def conf_answers(answers):
    """
    Determine the confidence degrees from the answers.

    Parameters
    ----------
    answers : list
        The answers.

    Returns
    -------
    interger
        The confidence degrees.

    """
    return np.min(answers, axis = 0)

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

def update_polytope_list(polytope_list, best_cs, tnorm_rule = "product"):
    """
    Update the list of polytopes according to the best coherent subset.

    Parameters
    ----------
    polytope_list : list
        List of polytopes.
    best_cs : array_like
        Best coherent subset.
    tnorm_rule : string, optional
        The T-norm to use. The default is "product".

    Returns
    -------
    list
        The updated polytope list.
    list
        The updated confidence degrees list.

    """
    new_polytope_list = []
    possibility_list = []
    answers_in_list = []
    polytope_list_copy = copy.deepcopy(polytope_list)
    for polytope in polytope_list_copy:
        polytope.subset_answers(best_cs, tnorm_rule)
        flag = 0
        for answers_in in answers_in_list:
            if polytope.get_answers() == answers_in:
                flag = 1
                break
        if flag == 0:
            new_polytope_list.append(polytope)
            answers_in_list.append(polytope.get_answers())
            possibility_list.append(polytope.get_possibility())
    return new_polytope_list, possibility_list
        
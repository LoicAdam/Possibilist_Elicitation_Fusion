# -*- coding: utf-8 -*-
"""This module gives tools for focal sets (compute empr notably)."""

from copy import deepcopy
import numpy as np

def _compute_levels(possibility_list):
    """
    Determine the levels (for the focal sets)

    Parameters
    ----------
    possibility_list : list
        Possibility for each polytope.

    Returns
    -------
    levels : array_like
        Levels.
    ind_sort : array_like
        Indices for sorting.

    """
    ind_sort = np.asarray(possibility_list).argsort()
    sorted_possibility_list = np.asarray(possibility_list)[ind_sort[::-1]]
    levels = np.unique(sorted_possibility_list)
    if np.min(levels) != 0:
        levels = np.append(0,levels)
    levels = np.sort(levels)[::-1]
    return levels, ind_sort
    
def compute_epmr_emr(pmr_list, possibility_list, inconsistency_type = 'maximum'):
    """
    Compute the EMPR and EMR

    Parameters
    ----------
    pmr_list : list
        PMR for each polytope.
    possibility_list : list
        Possibility for each polytope.
    inconsistency_type : string, optional
        How uncertainty is handeled. The default is 'maximum'.

    Returns
    -------
    epmr : float
        epmr.
    emr : float
        emr.

    """
    new_pmr_list, mr_list, new_possibility_list = _update_pmr_mr(pmr_list, possibility_list, inconsistency_type)
    levels, ind_sort = _compute_levels(new_possibility_list)
    epmr = _epmr_compute(new_pmr_list, new_possibility_list, levels, ind_sort)
    emr = _emr_compute(mr_list, new_possibility_list, levels, ind_sort)
    return epmr, emr
    
def _update_pmr_mr(pmr_list, possibility_list, inconsistency_type = 'maximum'):
    """
    Update the PMR and possibility list to handle uncertainty.

    Parameters
    ----------
    pmr_list : list
        PMR for each polytope.
    possibility_list : list
        Possibility for each polytope.
    inconsistency_type : string, optional
        How uncertainty is handeled. The default is 'maximum'.

    Returns
    -------
    new_pmr_list : list
        Updated pmr list.
    mr_list : list
        mr list.
    new_possibility_list : list
        Updated possibility list.

    """
    new_pmr_list = deepcopy(pmr_list)
    new_possibility_list = deepcopy(possibility_list)
    if np.max(possibility_list) != 1:
        new_possibility_list.append(1)
        if inconsistency_type == 'maximum':
            new_pmr_list.append(np.max(pmr_list, axis = 0))
        elif inconsistency_type == 'zero':
            #Equivalent to Guillot min model max(0, regret)
            new_pmr_list.append(np.zeros((new_pmr_list[0].shape[0], new_pmr_list[0].shape[0])))
        else:
            raise NotImplementedError(inconsistency_type, 'is an unknown rule.')
    mr_list = np.max(new_pmr_list, axis = 2)    
    return new_pmr_list, mr_list, new_possibility_list

def _epmr_compute(pmr_list, possibility_list, levels, ind_sort):
    """
    Compute the EMPR

    Parameters
    ----------
    pmr_list : list
        PMR for each polytope.
    possibility_list : list
        Possibility for each polytope.
    levels : array_like
        Levels.
    ind_sort : array_like
        Indices for sorting.

    Returns
    -------
    float
        epmr.

    """
    sorted_pmr_list = np.asarray(pmr_list)[ind_sort[::-1]]
    sorted_possibility_list = np.asarray(possibility_list)[ind_sort[::-1]]
    cur_conf = levels[0]
    next_conf = levels[1]
    j = 0
    res = np.zeros((pmr_list[0].shape[0],pmr_list[0].shape[0]))
    for i in range(0, len(pmr_list)):
        if sorted_possibility_list[i] != cur_conf:
            j = j+1
            cur_conf = levels[j]
            next_conf = levels[j+1]
        res = res + (cur_conf-next_conf) * sorted_pmr_list[i]
    return res

def _emr_compute(mr_list, possibility_list, levels, ind_sort):
    """
    Compute the EMR

    Parameters
    ----------
    mr_list : list
        MR for each polytope.
    possibility_list : list
        Possibility for each polytope.
    levels : array_like
        Levels.
    ind_sort : array_like
        Indices for sorting.

    Returns
    -------
    float
        emr.

    """
    sorted_mr_list = np.asarray(mr_list)[ind_sort[::-1]]
    sorted_possibility_list = np.asarray(possibility_list)[ind_sort[::-1]]
    cur_conf = levels[0]
    next_conf = levels[1]
    j = 0
    res = np.zeros((mr_list[0].shape[0]))
    for i in range(0, len(mr_list)):
        if sorted_possibility_list[i] != cur_conf:
            j = j+1
            cur_conf = levels[j]
            next_conf = levels[j+1] 
        res = res + (cur_conf-next_conf) * sorted_mr_list[i]
    return res

def compute_max_values_level(max_list, possibility_list, inconsistency_type = 'maximum'):
    """
    Determine the max using focal sets.

    Parameters
    ----------
    max_list : list
        Max for each polytope.
    possibility_list : list
        Possibility for each polytope.
    inconsistency_type : string, optional
        How uncertainty is handeled. The default is 'maximum'.

    Returns
    -------
    float
        max.

    """
    new_max_list, new_possibility_list = _update_max_values(max_list, possibility_list, inconsistency_type)
    levels, ind_sort = _compute_levels(new_possibility_list)
    max_values_level = _max_values_level_compute(new_max_list, new_possibility_list, levels, ind_sort)
    return max_values_level

def _update_max_values(max_list, possibility_list, inconsistency_type = 'maximum'):
    """
    Update the list of max and possbility to handle uncertainty.

    Parameters
    ----------
    max_list : list
        Max for each polytope.
    possibility_list : list
        Possibility for each polytope.
    inconsistency_type : string, optional
        How uncertainty is handeled. The default is 'maximum'.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    new_max_list : list
        Max list.
    new_possibility_list : list
        Updated possibility list.

    """
    new_max_list = deepcopy(max_list)
    new_possibility_list = deepcopy(possibility_list)
    if np.max(possibility_list) != 1:
        new_possibility_list.append(1)
        if inconsistency_type == 'maximum':
            new_max_list.append(np.maximum.reduce(max_list))
        elif inconsistency_type == 'zero':
            #Equivalent to Guillot min model max(0, regret)
            new_max_list.append(np.zeros(len(max_list[0])))
        else:
            raise NotImplementedError(inconsistency_type, 'is an unknown rule.')
    return new_max_list, new_possibility_list

def _max_values_level_compute(max_list, possibility_list, levels, ind_sort):
    """
    Compute the max

    Parameters
    ----------
    max_list : list
        Max for each polytope.
    possibility_list : list
        Possibility for each polytope.
    levels : array_like
        Levels.
    ind_sort : array_like
        Indices for sorting.

    Returns
    -------
    float
        The max.

    """
    sorted_max_list = np.asarray(max_list)[ind_sort[::-1]]
    sorted_possibility_list = np.asarray(possibility_list)[ind_sort[::-1]]
    cur_conf = levels[0]
    next_conf = levels[1]
    j = 0
    res = 0
    for i in range(0, len(max_list)):
        if sorted_possibility_list[i] != cur_conf:
            j = j+1
            cur_conf = levels[j]
            next_conf = levels[j+1]
        res = res + (cur_conf-next_conf) * sorted_max_list[i]
    return res

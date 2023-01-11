# -*- coding: utf-8 -*-
"""Elicitation"""

import time
import numpy as np
from elicitation.polytope import Polytope, construct_constrainst, cut_polytope, intersection_checker
from alternatives.data_preparation import get_pareto_efficient_alternatives
from elicitation.regret_strategies import PessimistStrategy, RandomStrategy
from elicitation.dm import get_choice_fixed
from elicitation.focal_set import compute_epmr_emr, compute_max_values_level
from elicitation.regret_calculation import pmr_polytope, mr_polytope, max_polytope

def robust_elicitation(alternatives, model, max_iter = -1,
                       rational = None, regret_limit = 10**-8):
    """
    Robust elicitation classic with CSS.

    Parameters
    ----------
    alternatives : array_like
        Alternatives.
    model : Model
        Model.
    max_iter : integer, optional
        If a maximum interation and how much. The default is -1.
    rational : list, optional
        To know if some answers should be rational or not. The default is None.
    regret_limit : float, optional
        If a regret limit. The default is 10**-8.

    Returns
    -------
    dict
        Elicitation information.

    """
    
    alternatives = get_pareto_efficient_alternatives(alternatives) #Get rid of non optimal solutions.
    nb_alternatives = len(alternatives)
    
    constraints = model.get_model_constrainsts()            
    constraints_a = constraints['A_eq']
    constraints_b = constraints['b_eq']
    bounds = constraints['bounds']
    first_polytope = Polytope(None,None,constraints_a, constraints_b, bounds)
    
    regret_strategy = PessimistStrategy(alternatives)
    
    #Maximal number of iterations.
    number_pairs = int(nb_alternatives*(nb_alternatives-1)/2)
    if max_iter != -1:
        max_iter = np.minimum(max_iter, number_pairs) 
    else:
        max_iter = number_pairs
        
    memr_estimated_list = np.zeros(max_iter)
    mmr_real_list = np.zeros(max_iter)
    rational_list = np.zeros(max_iter)
    
    scores = model.get_model_score(alternatives)
    
    if rational is None:
        rational = np.ones(max_iter)
    
    ite = 0
    
    start_time = time.time()
    
    pmr = pmr_polytope(alternatives, first_polytope, model)
    mr =  mr_polytope(pmr)
    
    while ite < max_iter:
                    
        candidate_alt, candidate_alt_id = regret_strategy.give_candidate(mr)
        worst_alt, _ = regret_strategy.give_oponent(pmr, candidate_alt_id)
        choice = get_choice_fixed(candidate_alt, worst_alt, rational[ite], model)
        best_prefered = choice['accepted']
        rational_list[ite] = choice['rational']
                
        best_alt, best_alt_id, regret = regret_strategy.get_best_alternative(mr)
            
        mmr_real_list[ite] = np.max(scores) - scores[best_alt_id]
        memr_estimated_list[ite] = regret
        
        if np.max(regret) <= regret_limit :# (ite != 0 and best_emr > memr_estimated_list[ite-1]) or :
            break
        
        new_constraint_a, new_constraint_b = construct_constrainst(candidate_alt, worst_alt, best_prefered, model)
        first_polytope.add_answer(new_constraint_a, new_constraint_b, 1, "minimum")
        pmr = pmr_polytope(alternatives, first_polytope, model)
        mr =  mr_polytope(pmr)
             
        ite = ite+1

    memr_estimated_list = memr_estimated_list[0:np.minimum(ite,max_iter)]
    mmr_real_list = mmr_real_list[0:np.minimum(ite,max_iter)]
    rational_list = rational_list[0:np.minimum(ite,max_iter)]
    d = {}
    d['time'] = time.time() - start_time
    d['best_alternative'] = best_alt
    d['memr_estimated'] = memr_estimated_list
    d['mmr_real'] = mmr_real_list
    d['rational'] = rational_list
    d['ite'] = ite
    return d

def robust_elicitation_random(alternatives, model, max_iter = -1,
                              rational = None, regret_limit = 10**-8):
    """
    Robust elicitation classic with random questions.

    Parameters
    ----------
    alternatives : array_like
        Alternatives.
    model : Model
        Model.
    max_iter : integer, optional
        If a maximum interation and how much. The default is -1.
    rational : list, optional
        To know if some answers should be rational or not. The default is None.
    regret_limit : float, optional
        If a regret limit. The default is 10**-8.

    Returns
    -------
    dict
        Elicitation information.

    """
    
    alternatives = get_pareto_efficient_alternatives(alternatives) #Get rid of non optimal solutions.
    nb_alternatives = len(alternatives)
    
    constraints = model.get_model_constrainsts()            
    constraints_a = constraints['A_eq']
    constraints_b = constraints['b_eq']
    bounds = constraints['bounds']
    first_polytope = Polytope(None,None,constraints_a, constraints_b, bounds)
    
    regret_strategy = RandomStrategy(alternatives)
    
    #Maximal number of iterations.
    number_pairs = int(nb_alternatives*(nb_alternatives-1)/2)
    if max_iter != -1:
        max_iter = np.minimum(max_iter, number_pairs) 
    else:
        max_iter = number_pairs
        
    mmr_real_list = np.zeros(max_iter)
    rational_list = np.zeros(max_iter)
    
    scores = model.get_model_score(alternatives)
    
    if rational is None:
        rational = np.ones(max_iter)
    
    ite = 0
    
    start_time = time.time()
    
    max_values = max_polytope(alternatives, first_polytope, model)
    
    while ite < max_iter:
                    
        candidate_alt, candidate_alt_id = regret_strategy.give_candidate(nb_alternatives)
        worst_alt, _ = regret_strategy.give_oponent(nb_alternatives, candidate_alt_id)
        choice = get_choice_fixed(candidate_alt, worst_alt, rational[ite], model)
        best_prefered = choice['accepted']
        rational_list[ite] = choice['rational']
                
        best_alt, best_alt_id = regret_strategy.get_best_alternative(max_values)
            
        mmr_real_list[ite] = np.max(scores) - scores[best_alt_id]
        
        new_constraint_a, new_constraint_b = construct_constrainst(candidate_alt, worst_alt, best_prefered, model)
        first_polytope.add_answer(new_constraint_a, new_constraint_b, 1, "minimum")
        max_values = max_polytope(alternatives, first_polytope, model)
             
        ite = ite+1

    mmr_real_list = mmr_real_list[0:np.minimum(ite,max_iter)]
    rational_list = rational_list[0:np.minimum(ite,max_iter)]
    d = {}
    d['time'] = time.time() - start_time
    d['best_alternative'] = best_alt
    d['mmr_real'] = mmr_real_list
    d['rational'] = rational_list
    d['ite'] = ite
    return d

def possibilist_elicitation(alternatives, model, confidence, t_norm = 'product',
                            max_iter = -1, inconsistency_type = 'maximum',
                            rational = None, regret_limit = 10**-10, min_possibility = 0):
    """
    Possibilist elicitation with CSS.

    Parameters
    ----------
    alternatives : array_like
        Alternatives.
    model : Model
        Model.
    confidence : list
        Confidence degrees.
    t_norm : string, optional
        Which T-norm to use. The default is 'product'.
    max_iter : integer, optional
        If a maximum interation and how much. The default is -1.
    inconsistency_type : string, optional
        Inconsistency in the EPMR. The default is 'maximum'.
    rational : list, optional
        To know if some answers should be rational or not. The default is None.
    regret_limit : float, optional
        If a regret limit. The default is 10**-10.
    min_possibility : float, optional
        Min possibility to consider a polytope. The default is 0.

    Returns
    -------
    dict
        Elicitation information.

    """
    alternatives = get_pareto_efficient_alternatives(alternatives) #Get rid of non optimal solutions.
    nb_alternatives = len(alternatives)
        
    constraints = model.get_model_constrainsts()            
    constraints_a = constraints['A_eq']
    constraints_b = constraints['b_eq']
    bounds = constraints['bounds']
    first_polytope = Polytope(None,None,constraints_a, constraints_b, bounds)
    
    polytope_list = []
    polytope_list.append(first_polytope)
    
    regret_strategy = PessimistStrategy(alternatives)
            
    #Maximal number of iterations.
    number_pairs = int(nb_alternatives*(nb_alternatives-1)/2)
    if max_iter != -1:
        max_iter = np.minimum(max_iter, number_pairs) 
    else:
        max_iter = number_pairs
        
    memr_estimated_list = np.zeros(max_iter)
    mmr_real_list = np.zeros(max_iter)
    rational_list = np.zeros(max_iter)
    inconsistency_list = np.zeros(max_iter)
              
    scores = model.get_model_score(alternatives)
    
    if rational is None:
        rational = np.ones(max_iter)
    
    ite = 0
    
    import time
    start_time = time.time()
    
    epmr = pmr_polytope(alternatives, first_polytope, model)
    emr =  mr_polytope(epmr)
    
    while ite < max_iter:
        
        pmr_list = []
        possibility_list = []
        new_polytope_list = []
            
        candidate_alt, candidate_alt_id = regret_strategy.give_candidate(emr)
        worst_alt, _ = regret_strategy.give_oponent(epmr, candidate_alt_id)
        choice = get_choice_fixed(candidate_alt, worst_alt, rational[ite], model)
        best_prefered = choice['accepted']
        rational_list[ite] = choice['rational']
                
        best_alt, best_alt_id, regret = regret_strategy.get_best_alternative(emr)
            
        mmr_real_list[ite] = np.max(scores) - scores[best_alt_id]
        memr_estimated_list[ite] = regret
        
        new_constraint_a, new_constraint_b = construct_constrainst(candidate_alt, worst_alt, best_prefered, model)
                
        for polytope in polytope_list:
        
            side = intersection_checker(polytope, new_constraint_a, new_constraint_b)
            
            '''If the new constrainst intersects with the current polytope:
            - Create two new ones,
            - Keep those with a suffissant possibility,
            - Delete the original polytope'''
            if side == 0:
                
                polytope_1, polytope_2 = cut_polytope(polytope, new_constraint_a, new_constraint_b, confidence[ite], t_norm)
                del polytope

                if polytope_1.get_possibility() > min_possibility:
                    new_polytope_list.append(polytope_1)
                    pmr = pmr_polytope(alternatives, polytope_1, model)
                    pmr_list.append(pmr)
                    possibility_list.append(polytope_1.get_possibility())
                    
                else:
                    del polytope_1
                    
                if polytope_2.get_possibility() > min_possibility:
                    new_polytope_list.append(polytope_2)
                    pmr = pmr_polytope(alternatives, polytope_2, model)
                    pmr_list.append(pmr)
                    possibility_list.append(polytope_2.get_possibility())
                    
                else:
                    del polytope_2
                    
            #Else, just update the possibility.
            else:
                
                if side == 1:
                    polytope.add_answer(new_constraint_a, new_constraint_b, 1, t_norm)
                else:
                    polytope.add_answer(-new_constraint_a, new_constraint_b, 1-confidence[ite], t_norm)
                                    
                if polytope.get_possibility() > min_possibility:
                    new_polytope_list.append(polytope)
                    pmr = pmr_polytope(alternatives, polytope, model)
                    pmr_list.append(pmr)
                    possibility_list.append(polytope.get_possibility())

                else:
                    del polytope
                    
        inconsistency_list[ite] = 1-np.max(possibility_list)
        polytope_list = new_polytope_list
        
        if np.max(regret) <= regret_limit :# (ite != 0 and best_emr > memr_estimated_list[ite-1]) or :
            break
             
        epmr, emr = compute_epmr_emr(pmr_list, possibility_list, inconsistency_type)
        

        ite = ite+1

    memr_estimated_list = memr_estimated_list[0:np.minimum(ite,max_iter)]
    mmr_real_list = mmr_real_list[0:np.minimum(ite,max_iter)]
    inconsistency_list = inconsistency_list[0:np.minimum(ite,max_iter)]
    rational_list = rational_list[0:np.minimum(ite,max_iter)]
    d = {}
    d['time'] = time.time() - start_time
    d['best_alternative'] = best_alt
    d['memr_estimated'] = memr_estimated_list
    d['mmr_real'] = mmr_real_list
    d['inconsistency'] = inconsistency_list
    d['rational'] = rational_list
    d['ite'] = ite
    d['polytopes'] = polytope_list
    return d

def possibilist_elicitation_random(alternatives, model, confidence, t_norm = 'product',
                                   max_iter = -1, inconsistency_type = 'maximum',
                                   rational = None, regret_limit = 10**-10, min_possibility = 0):
    """
    Possibilist elicitation random questioning.

    Parameters
    ----------
    alternatives : array_like
        Alternatives.
    model : Model
        Model.
    confidence : list
        Confidence degrees.
    t_norm : string, optional
        Which T-norm to use. The default is 'product'.
    max_iter : integer, optional
        If a maximum interation and how much. The default is -1.
    inconsistency_type : string, optional
        Inconsistency in the EPMR. The default is 'maximum'.
    rational : list, optional
        To know if some answers should be rational or not. The default is None.
    regret_limit : float, optional
        If a regret limit. The default is 10**-10.
    min_possibility : float, optional
        Min possibility to consider a polytope. The default is 0.

    Returns
    -------
    dict
        Elicitation information.

    """
    
    alternatives = get_pareto_efficient_alternatives(alternatives) #Get rid of non optimal solutions.
    nb_alternatives = len(alternatives)
        
    constraints = model.get_model_constrainsts()            
    constraints_a = constraints['A_eq']
    constraints_b = constraints['b_eq']
    bounds = constraints['bounds']
    first_polytope = Polytope(None,None,constraints_a, constraints_b, bounds)
    
    polytope_list = []
    polytope_list.append(first_polytope)
    
    regret_strategy = RandomStrategy(alternatives)
            
    #Maximal number of iterations.
    number_pairs = int(nb_alternatives*(nb_alternatives-1)/2)
    if max_iter != -1:
        max_iter = np.minimum(max_iter, number_pairs) 
    else:
        max_iter = number_pairs
        
    mmr_real_list = np.zeros(max_iter)
    rational_list = np.zeros(max_iter)
    inconsistency_list = np.zeros(max_iter)
              
    scores = model.get_model_score(alternatives)
    
    if rational is None:
        rational = np.ones(max_iter)
    
    ite = 0
    
    import time
    start_time = time.time()
    
    max_values = max_polytope(alternatives, first_polytope, model)
    
    while ite < max_iter:
        
        max_list = []
        possibility_list = []
        new_polytope_list = []
            
        candidate_alt, candidate_alt_id = regret_strategy.give_candidate(nb_alternatives)
        worst_alt, _ = regret_strategy.give_oponent(nb_alternatives, candidate_alt_id)
        choice = get_choice_fixed(candidate_alt, worst_alt, rational[ite], model)
        best_prefered = choice['accepted']
        rational_list[ite] = choice['rational']
                
        best_alt, best_alt_id = regret_strategy.get_best_alternative(max_values)
            
        mmr_real_list[ite] = np.max(scores) - scores[best_alt_id]
        
        new_constraint_a, new_constraint_b = construct_constrainst(candidate_alt, worst_alt, best_prefered, model)
                
        for polytope in polytope_list:
        
            side = intersection_checker(polytope, new_constraint_a, new_constraint_b)
            
            '''If the new constrainst intersects with the current polytope:
            - Create two new ones,
            - Keep those with a suffissant possibility,
            - Delete the original polytope'''
            if side == 0:
                
                polytope_1, polytope_2 = cut_polytope(polytope, new_constraint_a, new_constraint_b, confidence[ite], t_norm)
                del polytope

                if polytope_1.get_possibility() > min_possibility:
                    new_polytope_list.append(polytope_1)
                    max_values = max_polytope(alternatives, polytope_1, model)
                    max_list.append(max_values)
                    possibility_list.append(polytope_1.get_possibility())
                    
                else:
                    del polytope_1
                    
                if polytope_2.get_possibility() > min_possibility:
                    new_polytope_list.append(polytope_2)
                    max_values = max_polytope(alternatives, polytope_2, model)
                    max_list.append(max_values)
                    possibility_list.append(polytope_2.get_possibility())
                    
                else:
                    del polytope_2

            #Else, just update the possibility.
            else:
                
                if side == 1:
                    polytope.add_answer(new_constraint_a, new_constraint_b, 1, t_norm)
                else:
                    polytope.add_answer(-new_constraint_a, new_constraint_b, 1-confidence[ite], t_norm)
                                    
                if polytope.get_possibility() > min_possibility:
                    new_polytope_list.append(polytope)
                    max_values = max_polytope(alternatives, polytope, model)
                    max_list.append(max_values)
                    possibility_list.append(polytope.get_possibility())

                else:
                    del polytope
                    
        inconsistency_list[ite] = 1-np.max(possibility_list)
        polytope_list = new_polytope_list
             
        max_values = compute_max_values_level(max_list, possibility_list, inconsistency_type)
        

        ite = ite+1

    mmr_real_list = mmr_real_list[0:np.minimum(ite,max_iter)]
    inconsistency_list = inconsistency_list[0:np.minimum(ite,max_iter)]
    rational_list = rational_list[0:np.minimum(ite,max_iter)]
    d = {}
    d['time'] = time.time() - start_time
    d['best_alternative'] = best_alt
    d['mmr_real'] = mmr_real_list
    d['inconsistency'] = inconsistency_list
    d['rational'] = rational_list
    d['ite'] = ite
    d['polytopes'] = polytope_list
    return d
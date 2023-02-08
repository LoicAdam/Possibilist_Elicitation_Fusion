# -*- coding: utf-8 -*-
"""
Create data for elicitation.
"""

import os
import sys
import time
import pickle
import multiprocessing
from multiprocessing import Value
import numpy as np
from alternatives.data_preparation import generate_alternatives_score
from elicitation.elicitation import possibilist_elicitation, robust_elicitation
from elicitation.models import ModelWeightedSum
from fusion.l_out_n import find_incorrect_answers, k_among_n_fusion, new_optimal_recommendation

nb_parameters = 4
nb_questions = 15
nb_repetitions = 200
nb_alternatives = 50
t_norm = 'product'
conf_type = 'uniform'
elicitation_type = 'random'
question_type = 'random'
  
def init_globals(counter):
    global cnt
    cnt = counter
        
def conf_set():
        
    if conf_type == "strong":
        confidence_values = np.round(np.random.beta(7, 2, size = ((nb_repetitions, nb_questions))), decimals = 2)
    elif conf_type == "weak":
        confidence_values = np.round(np.random.beta(2, 7, size = ((nb_repetitions, nb_questions))), decimals = 2)
    elif conf_type == "uniform":
        confidence_values = np.round(np.random.beta(5, 5, size = ((nb_repetitions, nb_questions))), decimals = 2)
    elif conf_type == "intermediate":
        confidence_values = np.round(np.random.uniform(0.01, 0.99, size = ((nb_repetitions, nb_questions))), decimals = 2)
    else:
        raise ValueError("I did not code that.")
    
    random_mask = np.random.uniform(size = (nb_repetitions, nb_questions))
    rational = np.where(random_mask <= confidence_values + (1-confidence_values)/2, 1, 0)
    
    non_zeros_lines = np.count_nonzero(rational, axis = 1) #We add an error if none.
    for j in range(0, nb_repetitions):
        if non_zeros_lines[j] == nb_questions:
            rational[j, np.random.randint(0, nb_questions)] = 0

    return confidence_values, rational
        
def elicitation_classic(alternatives, model_values, rational):
    model = ModelWeightedSum(model_values)
    res = robust_elicitation(alternatives, model, max_iter = nb_questions,
                             rational = rational)    
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return res

def elicitation_possibilist_zero(alternatives, model_values, confidence_values, rational):
    model = ModelWeightedSum(model_values)
    res = possibilist_elicitation(alternatives, model, confidence_values, t_norm,
                                  max_iter = nb_questions, rational = rational,
                                  inconsistency_type = 'zero', question_type = question_type)
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return res

def elicitation_possibilist_maximum(alternatives, model_values, confidence_values, rational):
    model = ModelWeightedSum(model_values)
    res = possibilist_elicitation(alternatives, model, confidence_values, t_norm,
                                  max_iter = nb_questions, rational = rational,
                                  inconsistency_type = 'maximum', question_type = question_type)
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return res

def k_n_from_polytopes(alternatives, elicitation, model_values):
    
    polytope_list = elicitation['polytopes']
    all_detected_incorrect_answers = np.asarray(find_incorrect_answers(polytope_list))
    nb_wrong_answers_suggested = np.min(all_detected_incorrect_answers)
    fusion = k_among_n_fusion(polytope_list, nb_questions - np.min(all_detected_incorrect_answers), nb_questions)
    correction_k_n = new_optimal_recommendation(polytope_list, fusion, 
                                                alternatives, ModelWeightedSum(model_values))
    mmr_real_k_n = correction_k_n['mmr_real']
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return nb_wrong_answers_suggested, mmr_real_k_n

if __name__ == '__main__':
    
    alternatives_all = np.zeros((nb_repetitions, nb_alternatives, nb_parameters))
    for i in range(0, nb_repetitions):
        alternatives_all[i,:,:] = generate_alternatives_score(nb_alternatives, nb_parameters = nb_parameters, value = nb_parameters/2)
    model_values_all = np.random.dirichlet(np.ones(nb_parameters), size = nb_repetitions)
    confidence_values_all, rational_all = conf_set()
    
    number_of_workers = np.minimum(np.maximum(multiprocessing.cpu_count() - 2,1), 30)

    if elicitation_type == "css":
        
        start_time = time.time()
        cnt = Value('i', 0)
        with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
            elicitation_classic = pool.starmap(elicitation_classic, zip(alternatives_all, model_values_all, rational_all))
        sys.stdout.flush()
        pool.close()
        pool.join()
        print("Temps classique : ", time.time() - start_time)

    ###
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        elicitation_zero = pool.starmap(elicitation_possibilist_zero, zip(alternatives_all, model_values_all, confidence_values_all, rational_all))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Temps zero : ", time.time() - start_time)
    
    ###
    
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        elicitation_maximum = pool.starmap(elicitation_possibilist_maximum, zip(alternatives_all, model_values_all, confidence_values_all, rational_all))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Temps maximum : ", time.time() - start_time)
    
    ###
    
                
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        elicitation_k_n_zero = pool.starmap(k_n_from_polytopes, zip(alternatives_all, elicitation_zero, model_values_all))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Temps kn zero : ", time.time() - start_time)
    
    ###
    
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        elicitation_k_n_maximum = pool.starmap(k_n_from_polytopes, zip(alternatives_all, elicitation_maximum, model_values_all))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Temps kn max : ", time.time() - start_time)
    
    if not os.path.exists('data/' + elicitation_type + '/'):
        os.makedirs('data/' + elicitation_type + '/')
    with open('data/' + elicitation_type + '/elicitation_' + conf_type + '.pk','wb') as f:
        
        d = {}
        d['alternatives'] = alternatives_all
        d['model'] = model_values_all
        d['confidence'] = confidence_values_all
        d['rational'] = rational_all
        if elicitation_type == "css":
            d['elicitation_classic'] = elicitation_classic
        d['elicitation_zero'] = elicitation_zero
        d['elicitation_maximum'] = elicitation_maximum
        d['k_n_zero'] = elicitation_k_n_zero
        d['k_n_maximum'] = elicitation_k_n_maximum
        
        pickle.dump(d,f)
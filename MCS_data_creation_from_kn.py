# -*- coding: utf-8 -*-
"""
Create data for MCS.
"""

import os
import time
import pickle
import sys
import multiprocessing
from multiprocessing import Value
import numpy as np
from elicitation.models import ModelWeightedSum
from fusion.l_out_n import new_optimal_recommendation
from fusion.mcs import get_answers, conf_answers, find_coherent_subsets, find_best_cs, update_polytope_list

conf_type = 'uniform'
elicitation_type = 'random'

def init_globals(counter):
    global cnt
    cnt = counter
    
try:
    with open('data/' + elicitation_type + '/elicitation_' + conf_type + '.pk', 'rb') as f:
        d = pickle.load(f)
except IOError:  #file doesn't exist, no high-scores registered.
    d = {}
    
alternatives = d['alternatives']
model_values = d['model']
elicitation_zero = d['elicitation_zero'] 
elicitation_maximum = d['elicitation_maximum']
k_n_zero = d['k_n_zero']
k_n_maximum = d['k_n_maximum']

nb_repetitions = alternatives.shape[0]

def k_n_mcs_from_polytopes(alt, elicitation, model_val, nb_wrong_answers_suggested):
    polytope_list = elicitation['polytopes']
    nb_questions = elicitation['ite']
    answers = get_answers(polytope_list, nb_questions)
    conf = conf_answers(answers)
    cs = find_coherent_subsets(answers, nb_questions - nb_wrong_answers_suggested, nb_questions)
    best_cs = find_best_cs(cs, conf)
    new_polytope_list, possibility_list = update_polytope_list(polytope_list, best_cs, "product")
    correction_cs = new_optimal_recommendation(new_polytope_list, possibility_list, alt, 
                                               ModelWeightedSum(model_val))
    mmr_cs = correction_cs['mmr_real']
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return best_cs, mmr_cs

if __name__ == '__main__':
    
    number_of_workers = np.minimum(np.maximum(multiprocessing.cpu_count() - 2,1), 30)

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        res_zero = pool.starmap(k_n_mcs_from_polytopes, zip(alternatives, elicitation_zero, model_values, list(zip(*k_n_zero))[0]))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Temps mcs zero : ", time.time() - start_time)
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        res_max = pool.starmap(k_n_mcs_from_polytopes, zip(alternatives, elicitation_maximum, model_values, list(zip(*k_n_maximum))[0]))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Temps mcs max : ", time.time() - start_time)
    
    if not os.path.exists('data/' + elicitation_type + '/'):
        os.makedirs('data/' + elicitation_type + '/')
    with open('data/' + elicitation_type + '/mcs_' + conf_type + '.pk','wb') as f:
        
        d = {}
        d['res_zero'] = res_zero
        d['res_max'] = res_max
        pickle.dump(d,f)
    ###
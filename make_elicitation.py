# -*- coding: utf-8 -*-
"""Make the recommendations"""

import os
import sys
import time
import pickle
import multiprocessing
from multiprocessing import Value
import numpy as np
from scipy.optimize import linprog
from elicitation.elicitation import get_recommendation, robust_elicitation, possibilist_elicitation
from elicitation.models import ModelWeightedSum
from elicitation.polytope import Polytope
from fusion.l_out_n import find_incorrect_answers, k_among_n_fusion
from fusion.mcs import get_answers, find_all_maximum_coherent_subsets, update_possibility_list

nb_questions = 15
conf_type = 'uniform'
nb_parameters = 4
path = 'data/criteria_' + str(nb_parameters) + '/' + str(conf_type) + '/questions_' + str(nb_questions) + '/'

def init_globals(counter):
    global cnt
    cnt = counter
    
def make_dateset_certain(alternatives, model_values, rational):
    model = ModelWeightedSum(model_values)
    res = robust_elicitation(alternatives, model, max_iter = nb_questions,
                             rational = rational)
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return res
    
def make_dataset_zero(alternatives, model_values, confidence_values, rational):
    res = possibilist_elicitation(alternatives, ModelWeightedSum(model_values), 
                                  confidence_values, max_iter = nb_questions, 
                                  inconsistency_type = 'zero', rational = rational)
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return res

def make_dataset_ignorance(alternatives, model_values, confidence_values, rational):
    res = possibilist_elicitation(alternatives, ModelWeightedSum(model_values), 
                                  confidence_values, max_iter = nb_questions, 
                                  inconsistency_type = 'ignorance', rational = rational)
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return res

def get_number_errors(polytope_list):
    nb_detected_incorrect_answers = find_incorrect_answers(polytope_list)
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return np.min(nb_detected_incorrect_answers)

def l_out_of_n(polytope_list, value_list, alternatives, model_values, 
               nb_answers, nb_detected_incorrect_answers, inconsistency_type):
    possibility_list = k_among_n_fusion(polytope_list, nb_answers - nb_detected_incorrect_answers,
                                        nb_answers)
    res = get_recommendation(value_list, possibility_list, alternatives,
                             ModelWeightedSum(model_values), inconsistency_type,
                             polytopes = False)
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return res

def list_all_mcs(polytope_list, confidence):
    answers = get_answers(polytope_list, nb_questions)
    mcs_list = find_all_maximum_coherent_subsets(answers, nb_questions)
    d = {}
    d['mcs'] = mcs_list
    d['confidence'] = [confidence[mcs] for mcs in mcs_list]
    d['confidence_mean'] = np.asarray([np.mean(confidence[mcs]) for mcs in mcs_list])
    d['size'] = np.asarray([len(mcs) for mcs in mcs_list])
    d['answers'] = answers
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return d

def recommendation_all_mcs(mcs_list, polytope_list, value_list, answers,
                           alternatives, model_values):
    real_regret_list = []
    for i in range(0, len(mcs_list)):
        mcs = mcs_list[i]
        updated_possibility_list = update_possibility_list(answers, mcs, "product")
        real_regret_list.append(get_recommendation(value_list, updated_possibility_list,
                                                   alternatives, ModelWeightedSum(model_values),
                                                   polytopes = False))
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return real_regret_list

def epsilon_consistency(A_ub, b_ub, alternatives, model_values, inconsistency_type):
    model = ModelWeightedSum(model_values)
    constraints = model.get_model_constrainsts()
    A_eq = constraints['A_eq']
    b_eq = constraints['b_eq']
    bounds = constraints['bounds']
    n, p = A_ub.shape
    c = np.ones((p+n,1))
    c[0:p] = 0
    A_ub_new = np.hstack((A_ub, -np.identity(n)))
    A_eq_new = np.hstack((A_eq, np.ones((1,n))))
    bounds_new = bounds
    bounds_new = bounds_new + tuple((0, None) for _ in range(n))
    linprog_res = linprog(c, A_ub_new, b_ub, A_eq_new, b_eq, bounds_new,
                          method = 'highs')
    b_ub_new = b_ub + linprog_res.x[p:]
    new_polytope = Polytope(A_ub,b_ub_new,A_eq,b_eq, bounds)
    res = get_recommendation([new_polytope], [1], alternatives,
                             model, inconsistency_type)
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return res

if __name__ == '__main__':

    try:
        with open(path + 'dataset.pk','rb') as f:
            d = pickle.load(f)
    except IOError:  #file doesn't exist, no high-scores registered.
        d = {}

    alternatives_all = d['alternatives']
    model_values_all = d['model']
    confidence_values_all = d['confidence']
    rational_all = d['rational']
    nb_repetitions = alternatives_all.shape[0]

    ### Classic and possibilistic ###

    number_of_workers = np.minimum(np.maximum(multiprocessing.cpu_count() - 2,1), 30)

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        elicitation_classic = pool.starmap(make_dateset_certain, zip(alternatives_all, model_values_all, rational_all))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time classic: ", time.time() - start_time)
    
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'classic.pk','wb') as f:
        d = {}
        d['real_regret_classic'] = np.asarray([d['real_regret'] for d in elicitation_classic])
        pickle.dump(d,f)
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        elicitation_zero = pool.starmap(make_dataset_zero, zip(alternatives_all, model_values_all, confidence_values_all, rational_all))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time zero: ", time.time() - start_time)
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        elicitation_ignorance = pool.starmap(make_dataset_ignorance, zip(alternatives_all, model_values_all, confidence_values_all, rational_all))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time ignorance: ", time.time() - start_time)
    
    nones_zero = [i for i, x in enumerate(elicitation_zero) if x is None]
    nones_ignorance = [i for i, x in enumerate(elicitation_ignorance) if x is None]
    nones_union = list(set().union(nones_zero, nones_ignorance))
    if len(nones_union) != 0:
        elicitation_zero = [None if (i in nones_union) else elicitation_zero[i] for i in range(0, nb_repetitions)]
        elicitation_ignorance = [None if (i in nones_union) else elicitation_ignorance[i] for i in range(0, nb_repetitions)]
        alternatives_all = np.delete(alternatives_all, nones_union, 0)
        model_values_all = np.delete(model_values_all, nones_union, 0)
        confidence_values_all = np.delete(confidence_values_all, nones_union, 0)
        rational_all = np.delete(rational_all, nones_union, 0)
        nb_repetitions = nb_repetitions - len(nones_union)
        with open(path + '/dataset.pk','wb') as f:
            d = {}
            d['alternatives'] = alternatives_all
            d['model'] = model_values_all
            d['confidence'] = confidence_values_all
            d['rational'] = rational_all
            pickle.dump(d,f)
        
    with open(path + 'zero.pk','wb') as f:
        d = {}
        d['real_regret_zero'] = np.asarray([d['real_regret'] for d in elicitation_zero if d is not None])
        d['inconsistency_zero'] = np.asarray([d['inconsistency'] for d in elicitation_zero if d is not None]) 
        pickle.dump(d,f)
    
    polytope_zero = [d['polytope_list'] for d in elicitation_zero if d is not None]
    possibility_zero = [d['possibility_list'] for d in elicitation_zero if d is not None]
    pmr_zero = [d['value_list'] for d in elicitation_zero if d is not None]
    A_zero = [d['A'] for d in elicitation_zero if d is not None]
    b_zero = [d['b'] for d in elicitation_zero if d is not None]
    nb_questions_zero = [A.shape[0] for A in A_zero if d is not None]
    
    with open(path + 'ignorance.pk','wb') as f:
        d = {}
        d['real_regret_ignorance'] = np.asarray([d['real_regret'] for d in elicitation_ignorance if d is not None])
        d['inconsistency_ignorance'] = np.asarray([d['inconsistency'] for d in elicitation_ignorance if d is not None]) 
        pickle.dump(d,f)
    
    polytope_ignorance = [d['polytope_list'] for d in elicitation_ignorance if d is not None]
    possibility_ignorance = [d['possibility_list'] for d in elicitation_ignorance if d is not None]
    pmr_ignorance = [d['value_list'] for d in elicitation_ignorance if d is not None]
    A_ignorance = [d['A'] for d in elicitation_ignorance if d is not None]
    b_ignorance = [d['b'] for d in elicitation_ignorance if d is not None]
    nb_questions_ignorance = [A.shape[0] for A in A_ignorance if d is not None]

    ### Fusion zero ###

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        detected_errors_zero = pool.starmap(get_number_errors, zip(polytope_zero))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time number errors zero: ", time.time() - start_time)
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        l_out_of_n_zero = pool.starmap(l_out_of_n,
                                       zip(polytope_zero, pmr_zero,
                                           alternatives_all, model_values_all,
                                           nb_questions_zero, detected_errors_zero,
                                           np.repeat("zero", nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendations l-out-of-n zero: ", time.time() - start_time)
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        mcs_all_zero = pool.starmap(list_all_mcs, zip(polytope_zero, confidence_values_all))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time list MCS zero: ", time.time() - start_time)
    
    mcs_zero = [d['mcs'] for d in mcs_all_zero]
    mcs_zero_confidence = [d['confidence'] for d in mcs_all_zero]
    mcs_zero_confidence_mean = [d['confidence_mean'] for d in mcs_all_zero]
    mcs_zero_size = [d['size'] for d in mcs_all_zero]
    mcs_zero_answers = [d['answers'] for d in mcs_all_zero]
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        regret_mcs_zero = pool.starmap(recommendation_all_mcs,
                                       zip(mcs_zero, polytope_zero, pmr_zero,
                                           mcs_zero_answers, alternatives_all,
                                           model_values_all))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendations MCS zero ", time.time() - start_time)

    with open(path + 'mcs_zero.pk','wb') as f:
        d = {}
        d['mcs_zero'] = mcs_zero
        d['confidence_zero'] = mcs_zero_confidence
        d['confidence_mean_zero'] = mcs_zero_confidence_mean
        d['size_zero'] = mcs_zero_size
        d['real_regret_mcs_zero'] = [np.asarray([mcs['real_regret'] for mcs in mcs_list]) for mcs_list in regret_mcs_zero]
        pickle.dump(d,f)

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        epsilon_zero = pool.starmap(epsilon_consistency,
                                    zip(A_zero, b_zero, alternatives_all,
                                        model_values_all, np.repeat("zero", nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendations epsilon zero: ", time.time() - start_time)

    with open(path + 'correction_zero.pk','wb') as f:
        d = {}
        d['real_regret_zero_l_out_of_n'] = np.asarray([d['real_regret'] for d in l_out_of_n_zero])
        d['real_regret_zero_epsilon'] = np.asarray([d['real_regret'] for d in epsilon_zero])
        d['nb_answers_zero'] = np.asarray(nb_questions_zero) 
        d['nb_errors_zero'] = np.asarray(detected_errors_zero) 
        pickle.dump(d,f)

    ### Fusion ignorance ###

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        detected_errors_ignorance = pool.starmap(get_number_errors, zip(polytope_ignorance))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time number errors ignorance: ", time.time() - start_time)
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        l_out_of_n_ignorance = pool.starmap(l_out_of_n,
                                       zip(polytope_ignorance, pmr_ignorance,
                                           alternatives_all, model_values_all,
                                           nb_questions_ignorance, detected_errors_ignorance,
                                           np.repeat("ignorance", nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendations l-out-of-n ignorance: ", time.time() - start_time)

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        mcs_all_ignorance = pool.starmap(list_all_mcs, zip(polytope_ignorance, confidence_values_all))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time list MCS ignorance: ", time.time() - start_time)
    
    mcs_ignorance = [d['mcs'] for d in mcs_all_ignorance]
    mcs_ignorance_confidence = [d['confidence'] for d in mcs_all_ignorance]
    mcs_ignorance_confidence_mean = [d['confidence_mean'] for d in mcs_all_ignorance]
    mcs_ignorance_size = [d['size'] for d in mcs_all_ignorance]
    mcs_ignorance_answers = [d['answers'] for d in mcs_all_ignorance]
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        regret_mcs_ignorance = pool.starmap(recommendation_all_mcs,
                                       zip(mcs_ignorance, polytope_ignorance, pmr_ignorance,
                                           mcs_ignorance_answers, alternatives_all,
                                           model_values_all))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendations MCS ignorance ", time.time() - start_time)

    with open(path + 'mcs_ignorance.pk','wb') as f:
        d = {}
        d['mcs_ignorance'] = mcs_ignorance
        d['confidence_ignorance'] = mcs_ignorance_confidence
        d['confidence_mean_ignorance'] = mcs_ignorance_confidence_mean
        d['size_ignorance'] = mcs_ignorance_size
        d['real_regret_mcs_ignorance'] = [np.asarray([mcs['real_regret'] for mcs in mcs_list]) for mcs_list in regret_mcs_ignorance]
        pickle.dump(d,f)

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        epsilon_ignorance = pool.starmap(epsilon_consistency,
                                              zip(A_ignorance, b_ignorance, alternatives_all,
                                                  model_values_all, np.repeat("ignorance", nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendations epsilon ignorance: ", time.time() - start_time)

    with open(path + 'correction_ignorance.pk','wb') as f:
        d = {}
        d['real_regret_ignorance_l_out_of_n'] = np.asarray([d['real_regret'] for d in l_out_of_n_ignorance])
        d['real_regret_ignorance_epsilon'] = np.asarray([d['real_regret'] for d in epsilon_ignorance])
        d['nb_errors_ignorance'] = np.asarray(detected_errors_ignorance) 
        d['nb_answers_ignorance'] = np.asarray(nb_questions_ignorance) 
        pickle.dump(d,f)

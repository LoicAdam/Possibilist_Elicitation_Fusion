# -*- coding: utf-8 -*-
"""Create datasets with questions and answers"""

import os
import pickle
import numpy as np
from alternatives.data_preparation import generate_alternatives_score

nb_parameters = 4
nb_questions = 23
nb_repetitions = 300
nb_alternatives = 50
conf_type = 'uniform'
path = 'data/criteria_' + str(nb_parameters) + '/' + str(conf_type) + '/questions_' + str(nb_questions) + '/'

def init_globals(counter):
    global cnt
    cnt = counter

def conf_set():

    if conf_type == "strong":
        confidence_values = np.round(np.random.beta(7, 2, size = ((nb_repetitions, nb_questions))),
                                     decimals = 2)
    elif conf_type == "weak":
        confidence_values = np.round(np.random.beta(2, 7, size = ((nb_repetitions, nb_questions))),
                                     decimals = 2)
    elif conf_type == "intermediate":
        confidence_values = np.round(np.random.beta(5, 5, size = ((nb_repetitions, nb_questions))),
                                     decimals = 2)
    elif conf_type == "uniform":
        confidence_values = np.round(np.random.uniform(0.01, 0.99, size = ((nb_repetitions, nb_questions))),
                                     decimals = 2)   
    else:
        raise NotImplementedError("I did not code that.")    

    random_mask = np.random.uniform(size = (nb_repetitions, nb_questions))
    rational = np.where(random_mask <= confidence_values + (1-confidence_values)/2, 1, 0)
    non_zeros_lines = np.count_nonzero(rational, axis = 1) #We add an error if none.
    for j in range(0, nb_repetitions):
        if non_zeros_lines[j] == nb_questions:
            rational[j, np.random.randint(0, nb_questions)] = 0
    return confidence_values, rational

if __name__ == '__main__':

    alternatives_all = np.zeros((nb_repetitions, nb_alternatives, nb_parameters))
    for i in range(0, nb_repetitions):
        alternatives_all[i,:,:] = generate_alternatives_score(nb_alternatives,
                                                              nb_parameters = nb_parameters,
                                                              value = nb_parameters/2)
    model_values_all = np.random.dirichlet(np.ones(nb_parameters), size = nb_repetitions)
    confidence_values_all, rational_all = conf_set()

    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/dataset.pk','wb') as f:
        d = {}
        d['alternatives'] = alternatives_all
        d['model'] = model_values_all
        d['confidence'] = confidence_values_all
        d['rational'] = rational_all
        pickle.dump(d,f)
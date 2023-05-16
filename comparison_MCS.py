# -*- coding: utf-8 -*-
"""
Compare MCSs.
"""

from scipy import stats

import scipy.stats
import pickle
import os
import matplotlib.pyplot as plt 
import numpy as np
import tikzplotlib

conf_type = 'strong'
nb_questions = 15
nb_parameters = 4
path_data = 'data/criteria_' + str(nb_parameters) + '/' + str(conf_type) + '/questions_' + str(nb_questions) + '/'

if __name__ == '__main__':
    
    try:
        with open(path_data + 'dataset.pk', 'rb') as f:
            dataset = pickle.load(f)
    except IOError:
        dataset = {}
    try:
        with open(path_data + 'classic.pk', 'rb') as f:
            data_classic = pickle.load(f)
    except IOError:
        data_classic = {}
    try:
        with open(path_data + 'zero.pk', 'rb') as f:
            data_zero = pickle.load(f)
    except IOError:
        data_zero = {}
    try:
        with open(path_data + 'correction_zero.pk', 'rb') as f:
            data_zero_correction = pickle.load(f)
    except IOError:
        data_zero_correction = {}
    try:
        with open(path_data + 'mcs_zero.pk', 'rb') as f:
            data_zero_mcs = pickle.load(f)
    except IOError:
        data_zero_mcs = {}    
        
    inconsistent_data = np.where(data_zero['inconsistency_zero'] > 0)[0]
    positions = np.zeros((len(inconsistent_data), 3))
    nb_mcs = np.zeros(len(inconsistent_data))

    k = 0
    for i in inconsistent_data:
        
        mcs_size = data_zero_mcs['size_zero'][i]
        mcs_confidence_mean = data_zero_mcs['confidence_mean_zero'][i]
        mcs_res = data_zero_mcs['real_regret_mcs_zero'][i]
        mcs_min = np.min(mcs_res)
        
        mcs_confidence_sorted_idx = np.argsort(mcs_confidence_mean)[::-1]
        mcs_size_confidence_sorted_idx = np.lexsort((mcs_confidence_mean, mcs_size))[::-1] #Size then Confidence
        mcs_random_sorted_idx = np.arange(0, len(mcs_res))
        np.random.shuffle(mcs_random_sorted_idx)
        
        mcs_confidence_sorted_res = mcs_res[mcs_confidence_sorted_idx]
        mcs_size_confidence_sorted_res = mcs_res[mcs_size_confidence_sorted_idx]
        mcs_random_sorted_res = mcs_res[mcs_random_sorted_idx]
        
        positions[k, 0] = np.where(mcs_confidence_sorted_res == mcs_min)[0][0]
        positions[k, 1] = np.where(mcs_size_confidence_sorted_res == mcs_min)[0][0]
        positions[k, 2] = np.where(mcs_random_sorted_res == mcs_min)[0][0]
        nb_mcs[k] = len(mcs_size)
        k += 1

    positions_mean = np.mean(positions, axis = 0)
    
    positions_stats = np.zeros((positions.shape[1],5))
    for i in range(0, positions.shape[1]):
        positions_stats[i,:] = [np.min(positions[:,i]),
                                np.quantile(positions[:,i], 0.25),
                                np.quantile(positions[:,i], 0.5),
                                np.quantile(positions[:,i], 0.75),
                                np.max(positions[:,i])]

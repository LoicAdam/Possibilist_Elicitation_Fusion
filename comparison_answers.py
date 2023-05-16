# -*- coding: utf-8 -*-
"""
Compare strategies using random questions.
"""
import pickle
import os
import matplotlib.pyplot as plt 
import numpy as np
import tikzplotlib
import seaborn

conf_type = 'uniform'
nb_questions = 15
nb_parameters = 4
path_data = 'data/criteria_' + str(nb_parameters) + '/' + str(conf_type) + '/questions_' + str(nb_questions) + '/'
path_results = 'results/criteria_' + str(nb_parameters) + '/' + str(conf_type) + '/questions_' + str(nb_questions) + '/'
    
if __name__ == '__main__':
    
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    
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
        with open(path_data + 'ignorance.pk', 'rb') as f:
            data_ignorance = pickle.load(f)
    except IOError:
        data_ignorance = {}   
    try:
        with open(path_data + 'correction_zero.pk', 'rb') as f:
            data_zero_correction = pickle.load(f)
    except IOError:
        data_zero_correction = {}
    try:
        with open(path_data + 'correction_ignorance.pk', 'rb') as f:
            data_ignorance_correction = pickle.load(f)
    except IOError:
        data_ignorance_correction = {}
    try:
        with open(path_data + 'mcs_zero.pk', 'rb') as f:
            data_zero_mcs = pickle.load(f)
    except IOError:
        data_zero_mcs = {}
    try:
        with open(path_data + 'mcs_ignorance.pk', 'rb') as f:
            data_ignorance_mcs = pickle.load(f)
    except IOError:
        data_ignorance_mcs = {}        
        
    rational_all = dataset['rational']
    nb_errors_real_zero = data_zero_correction['nb_answers_zero'] - np.count_nonzero(rational_all, axis = 1)
    nb_errors_detected_zero = np.asarray(data_zero_correction['nb_errors_zero'])
    nb_errors_detected_group_zero = np.zeros((np.max(nb_errors_real_zero)+1, np.max(nb_errors_real_zero)+1)).astype(int)
    for i in range(0, len(nb_errors_detected_zero)):
        nb_errors_detected_group_zero[nb_errors_detected_zero[i], nb_errors_real_zero[i]] += 1
    difference_zero = nb_errors_real_zero - nb_errors_detected_zero
    difference_mean_zero = np.mean(difference_zero)

    fig, ax = plt.subplots()
    labels =  np.where(nb_errors_detected_group_zero[:,:] > 0, nb_errors_detected_group_zero[:,:] , '')
    labels = labels.astype(str)
    ax = seaborn.heatmap(nb_errors_detected_group_zero[:,:], annot=labels, fmt = '',
                         cbar=False, cmap = "Blues", vmax = 30)
    ax.invert_yaxis()
    ax.set_xlabel('Real number of errors', fontsize = 9)
    ax.set_ylabel('Number of errors detected', fontsize = 9)
    ax.set_xlim(0,np.max(nb_errors_real_zero)+1)
    fig.set_dpi(300.0)
    tikzplotlib.save(path_results + 'nb_wrong_answers_detected_zero.tex')
    plt.savefig(path_results + 'nb_wrong_answers_detected_zero.png', dpi=300)
    
    rational_all = dataset['rational']
    nb_errors_real_ignorance = data_ignorance_correction['nb_answers_ignorance'] - np.count_nonzero(rational_all, axis = 1)
    nb_errors_detected_ignorance = np.asarray(data_ignorance_correction['nb_errors_ignorance'])
    nb_errors_detected_group_ignorance = np.zeros((np.max(nb_errors_real_ignorance)+1, np.max(nb_errors_real_ignorance)+1)).astype(int)
    for i in range(0, len(nb_errors_detected_ignorance)):
        nb_errors_detected_group_ignorance[nb_errors_detected_ignorance[i], nb_errors_real_ignorance[i]] += 1
    difference_ignorance = nb_errors_real_ignorance - nb_errors_detected_ignorance
    difference_mean_ignorance = np.mean(difference_ignorance)

    fig, ax = plt.subplots()
    labels =  np.where(nb_errors_detected_group_ignorance[:,:] > 0, nb_errors_detected_group_ignorance[:,:] , '')
    labels = labels.astype(str)
    ax = seaborn.heatmap(nb_errors_detected_group_ignorance[:,:], annot=labels, fmt = '',
                         cbar=False, cmap = "Blues", vmax = 30)
    ax.invert_yaxis()
    ax.set_xlabel('Real number of errors', fontsize = 9)
    ax.set_ylabel('Number of errors detected', fontsize = 9)
    ax.set_xlim(0,np.max(nb_errors_real_ignorance)+1)
    fig.set_dpi(300.0)
    tikzplotlib.save(path_results + 'nb_wrong_answers_detected_ignorance.tex')
    plt.savefig(path_results + 'nb_wrong_answers_detected_ignorance.png', dpi=300)
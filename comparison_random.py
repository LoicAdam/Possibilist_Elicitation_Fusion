# -*- coding: utf-8 -*-
"""
Compare strategies using random elicitation.
"""
import pickle
import os
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import tikzplotlib

if __name__ == '__main__':
    
    elicitation_type = "random/"
    data_type = "uniform"
    
    if not os.path.exists("results/" + elicitation_type + data_type):
        os.makedirs("results/" + elicitation_type + data_type)
    
    try:
        with open('data/' + elicitation_type + 'elicitation_' + data_type + '.pk', 'rb') as f:
            d = pickle.load(f)
    except IOError:
        d = {}
        
    try:
        with open('data/' +  elicitation_type + 'mcs_' + data_type + '.pk', 'rb') as f:
            e = pickle.load(f)
    except IOError:
        e = {}   

    rational = d['rational']
    elicitation_zero = d['elicitation_zero'] 
    elicitation_maximum = d['elicitation_maximum']
    k_n_zero = d['k_n_zero']
    k_n_maximum = d['k_n_maximum']
    mmr_mcs_zero = np.asarray(list(zip(*e['res_zero']))[1])
    mmr_mcs_maximum = np.asarray(list(zip(*e['res_max']))[1])
        
    nb_questions = rational.shape[1]
    nb_repetitions = rational.shape[0]
    
    nb_errors_all = nb_questions - np.count_nonzero(rational, axis = 1)
    inconsistency_before_zero = np.zeros(nb_repetitions)
    inconsistency_before_maximum = np.zeros(nb_repetitions)
    mmr_real_zero = np.zeros(nb_repetitions)
    mmr_real_maximum = np.zeros(nb_repetitions)
    
    for i in range(0, nb_repetitions):
        
        inconsistency_before_zero[i] = elicitation_zero[i]['inconsistency'][-1]
        inconsistency_before_maximum[i] = elicitation_maximum[i]['inconsistency'][-1]
        mmr_real_zero[i] = elicitation_zero[i]['mmr_real'][-1]
        mmr_real_maximum[i] = elicitation_maximum[i]['mmr_real'][-1]
      
    inconsistency_threshold = np.median(inconsistency_before_zero[inconsistency_before_zero > 0])

    #
    
    unique_errors, counts_errors = np.unique(nb_errors_all, return_counts=True)
    fig, ax = plt.subplots()
    ax.bar(unique_errors, counts_errors)
    label = ax.set_xlabel('Number real errors', fontsize = 9)
    fig.set_dpi(300.0)
    tikzplotlib.save('results/' + elicitation_type + data_type + '/nb_errors.tex')
    plt.savefig('results/' + elicitation_type + data_type + '/nb_errors.png', dpi=300)
    
    nb_wrong_answers_suggested_zero, mmr_real_k_n_zero = list(map(list, zip(*k_n_zero)))
    nb_wrong_answers_suggested_zero = np.array(nb_wrong_answers_suggested_zero).astype(int)
    mmr_real_k_n_zero = np.asarray(mmr_real_k_n_zero)
    
    difference_zero = nb_errors_all - nb_wrong_answers_suggested_zero
    unique_zero, counts_zero = np.unique(difference_zero, return_counts=True)
    
    fig, ax = plt.subplots()
    ax.bar(unique_zero, counts_zero)
    label = ax.set_xlabel('Non-detected errors', fontsize = 9)
    fig.set_dpi(300.0)
    tikzplotlib.save('results/' + elicitation_type + data_type + '/nb_answers_detected_zero.tex')
    plt.savefig('results/' + elicitation_type + data_type + '/nb_answers_detected_zero.png', dpi=300)
    
    mat_zero = np.zeros((np.max(nb_errors_all)+1,np.max(nb_errors_all)+1))
    mat_zero[0,0] = nb_repetitions - len(nb_errors_all)
    for i in range(0, len(nb_errors_all)):
        mat_zero[nb_wrong_answers_suggested_zero[i], nb_errors_all[i]] += 1
      
    #
    
    nb_wrong_answers_suggested_maximum, mmr_real_k_n_maximum = list(map(list, zip(*k_n_maximum)))
    nb_wrong_answers_suggested_maximum = np.array(nb_wrong_answers_suggested_maximum).astype(int)
    mmr_real_k_n_maximum = np.asarray(mmr_real_k_n_maximum)

    difference_maximum = nb_errors_all - nb_wrong_answers_suggested_maximum
    unique_maximum, counts_maximum = np.unique(difference_maximum, return_counts=True)
    
    fig, ax = plt.subplots()
    ax.bar(unique_maximum, counts_maximum)
    label = ax.set_xlabel('Non-detected errors', fontsize = 9)
    fig.set_dpi(300.0)
    tikzplotlib.save('results/' + elicitation_type + data_type + '/nb_answers_detected_max.tex')
    plt.savefig('results/' + elicitation_type + data_type + '/nb_answers_detected_max.png', dpi=300)
    
    mat_maximum = np.zeros((np.max(nb_errors_all)+1,np.max(nb_errors_all)+1))
    mat_maximum[0,0] = nb_repetitions - len(nb_errors_all)
    for i in range(0, len(nb_errors_all)):
        mat_maximum[nb_wrong_answers_suggested_maximum[i], nb_errors_all[i]] += 1
 
    data_inconsistency_non_zero = []
    data_inconsistency_non_zero = [mmr_real_zero[inconsistency_before_zero != 0],
                                   mmr_real_k_n_zero[inconsistency_before_zero != 0],
                                   mmr_mcs_zero[inconsistency_before_zero != 0],
                                   mmr_real_maximum[inconsistency_before_maximum != 0],
                                   mmr_real_k_n_maximum[inconsistency_before_maximum != 0],
                                   mmr_mcs_maximum[inconsistency_before_maximum != 0]]
    
    fig, ax = plt.subplots()
    ax.boxplot(data_inconsistency_non_zero)
    plt.xticks([1, 2, 3, 4, 5, 6], ['zero', 'kn zero', 'mcs zero', 
                                       'max', 'kn max', 'mcs max'])
    ax.set_ylim(-0.05,1)
    fig.set_dpi(300.0)
    tikzplotlib.save('results/' + elicitation_type + data_type + '/compare_non_zero.tex')
    plt.savefig('results/' + elicitation_type + data_type + '/compare_non_zero.png', dpi=300)
    
    data_inconsistency_zero = [mmr_real_zero[inconsistency_before_zero == 0],
                              mmr_real_k_n_zero[inconsistency_before_zero == 0],
                              mmr_mcs_zero[inconsistency_before_zero == 0],
                              mmr_real_maximum[inconsistency_before_maximum == 0],
                              mmr_real_k_n_maximum[inconsistency_before_maximum == 0],
                              mmr_mcs_maximum[inconsistency_before_maximum == 0]]
    fig, ax = plt.subplots()
    ax.boxplot(data_inconsistency_zero)
    plt.xticks([1, 2, 3, 4, 5, 6], ['zero', 'kn zero', 'mcs zero', 
                                       'max', 'kn max', 'mcs max'])
    ax.set_ylim(-0.05,1)
    fig.set_dpi(300.0)
    tikzplotlib.save('results/' + elicitation_type + data_type + '/compare_zero.tex')
    plt.savefig('results/' + elicitation_type + data_type + '/compare_zero.png', dpi=300)
    
    data_inconsistency_low = [mmr_real_zero[(0 < inconsistency_before_zero) * (inconsistency_before_zero <= inconsistency_threshold)],
                              mmr_real_k_n_zero[(0 < inconsistency_before_zero) * (inconsistency_before_zero <= inconsistency_threshold)],
                              mmr_mcs_zero[(0 < inconsistency_before_zero) * (inconsistency_before_zero <= inconsistency_threshold)],
                              mmr_real_maximum[(0 < inconsistency_before_zero) * (inconsistency_before_maximum <= inconsistency_threshold)],
                              mmr_real_k_n_maximum[(0 < inconsistency_before_zero) * (inconsistency_before_maximum <= inconsistency_threshold)],
                              mmr_mcs_maximum[(0 < inconsistency_before_zero) * (inconsistency_before_maximum <= inconsistency_threshold)]]
    fig, ax = plt.subplots()
    ax.boxplot(data_inconsistency_low)
    plt.xticks([1, 2, 3, 4, 5, 6], ['zero', 'kn zero', 'mcs zero', 
                                       'max', 'kn max', 'mcs max'])
    ax.set_ylim(-0.05,1)
    fig.set_dpi(300.0)
    tikzplotlib.save('results/' + elicitation_type + data_type + '/compare_low.tex')
    plt.savefig('results/' + elicitation_type + data_type + '/compare_low.png', dpi=300)
    
    data_inconsistency_high = [mmr_real_zero[inconsistency_before_zero > inconsistency_threshold],
                              mmr_real_k_n_zero[inconsistency_before_zero > inconsistency_threshold],
                              mmr_mcs_zero[inconsistency_before_zero > inconsistency_threshold],
                              mmr_real_maximum[inconsistency_before_maximum > inconsistency_threshold],
                              mmr_real_k_n_maximum[inconsistency_before_maximum > inconsistency_threshold],
                              mmr_mcs_maximum[inconsistency_before_maximum > inconsistency_threshold]]
    fig, ax = plt.subplots()
    ax.boxplot(data_inconsistency_high)
    plt.xticks([1, 2, 3, 4, 5, 6], ['zero', 'kn zero', 'mcs zero', 
                                       'max', 'kn max', 'mcs max'])
    ax.set_ylim(-0.05,1)
    fig.set_dpi(300.0)
    tikzplotlib.save('results/' + elicitation_type + data_type + '/compare_high.tex')
    plt.savefig('results/' + elicitation_type + data_type + '/compare_high.png', dpi=300)
    
    gain_knn_zero = data_inconsistency_non_zero[0] - data_inconsistency_non_zero[1]
    gain_knn_max = data_inconsistency_non_zero[3] - data_inconsistency_non_zero[4]
    gain_mcs_zero = data_inconsistency_non_zero[0] - data_inconsistency_non_zero[2]
    gain_mcs_max = data_inconsistency_non_zero[3] - data_inconsistency_non_zero[5]
    
    fig, ax = plt.subplots()
    ax.boxplot([gain_knn_zero, gain_mcs_zero, gain_knn_max, gain_mcs_max])
    plt.xticks([1, 2, 3, 4], ['kn zero', 'mcs zero', 'kn max', 'mcs max'])
    ax.set_ylim(-1.05,1.05)
    fig.set_dpi(300.0)
    tikzplotlib.save('results/' + elicitation_type + data_type + '/gain.tex')
    plt.savefig('results/' + elicitation_type + data_type + '/gain.png', dpi=300)
    
    data = np.asarray([mmr_real_zero, mmr_real_k_n_zero,
                       mmr_mcs_zero, mmr_real_maximum, mmr_real_k_n_maximum,
                       mmr_mcs_maximum]).T
    mat_better = np.zeros((6,6))
    mat_strictly_better = np.zeros((6,6))
    
    for i in range(0, nb_repetitions):
        for j in range(0,6):
            for k in range(0,6):
                
                if data[i,j] < data[i,k]:
                    mat_strictly_better[j,k] += 1
                    mat_better[j,k] +=1
                elif data[i,j] == data[i,k]:
                    mat_better[j,k] +=1
                    
    #Export for R, because I love R for statistical tests.
    if not os.path.exists('data/' + elicitation_type):
        os.makedirs('data/' + elicitation_type)
    DF = pd.DataFrame(np.asarray(data_inconsistency_non_zero).T)
    DF.rename(columns = {0:'zero',
                         1:'zero_kn',
                         2:'zero_mcs',
                         3:'max',
                         4:'max_kn',
                         5:'max_mcs'}, 
            inplace = True)
    DF.to_csv('data/' + elicitation_type + 'data_' + data_type + '.csv')
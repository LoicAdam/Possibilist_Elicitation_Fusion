# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 17:08:27 2023

@author: adamloic
"""

# -*- coding: utf-8 -*-
"""
Statistical tests.
"""

from scipy import stats
import pickle
import os
import matplotlib.pyplot as plt 
import numpy as np
import tikzplotlib
import matplotlib.lines as mlines

conf_type = 'uniform'
nb_questions = 15
nb_parameters = 4
path_data = 'data/criteria_' + str(nb_parameters) + '/' + str(conf_type) + '/questions_' + str(nb_questions) + '/'
path_results = 'results/comparison_uncertainty_strategy/'

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

    nb_repetitions = dataset['alternatives'].shape[0]
    mcs_confidence_real_regret = np.zeros((nb_repetitions,2))
    mcs_size_real_regret = np.zeros((nb_repetitions,2))
    mcs_size_confidence_real_regret = np.zeros((nb_repetitions,2))
    mcs_best_real_regret = np.zeros((nb_repetitions,2))

    for i in range(0,len(data_zero['real_regret_zero'])):
        
        mcs_size_zero = data_zero_mcs['size_zero'][i]
        mcs_confidence_mean_zero = data_zero_mcs['confidence_mean_zero'][i]
        mcs_res_zero = data_zero_mcs['real_regret_mcs_zero'][i]

        mcs_size_ignorance = data_ignorance_mcs['size_ignorance'][i]
        mcs_confidence_mean_ignorance = data_ignorance_mcs['confidence_mean_ignorance'][i]
        mcs_res_ignorance = data_ignorance_mcs['real_regret_mcs_ignorance'][i]

        max_confidence_mcs_idx_zero = np.random.choice(np.where(mcs_confidence_mean_zero == np.max(mcs_confidence_mean_zero))[0])
        max_confidence_mcs_idx_ignorance = np.random.choice(np.where(mcs_confidence_mean_ignorance == np.max(mcs_confidence_mean_ignorance))[0])
        mcs_size_real_regret[i,:] = [mcs_res_zero[max_confidence_mcs_idx_zero],
                                     mcs_res_ignorance[max_confidence_mcs_idx_ignorance]]

        max_size_mcs_idx_list_zero = np.where(mcs_size_zero == np.max(mcs_size_zero))[0]
        max_size_mcs_idx_zero = np.random.choice(max_size_mcs_idx_list_zero)
        max_size_mcs_idx_list_ignorance = np.where(mcs_size_ignorance == np.max(mcs_size_ignorance))[0]
        max_size_mcs_idx_ignorance = np.random.choice(max_size_mcs_idx_list_ignorance)
        mcs_confidence_real_regret[i,:] = [mcs_res_zero[max_size_mcs_idx_zero],
                                           mcs_res_ignorance[max_size_mcs_idx_ignorance]]

        max_size_confidence_mean_zero = mcs_confidence_mean_zero[max_size_mcs_idx_list_zero]
        max_size_max_confidence_mcs_idx_zero = np.random.choice(np.where(max_size_confidence_mean_zero == np.max(max_size_confidence_mean_zero))[0])
        max_size_confidence_mean_ignorance = mcs_confidence_mean_ignorance[max_size_mcs_idx_list_ignorance]
        max_size_max_confidence_mcs_idx_ignorance = np.random.choice(np.where(max_size_confidence_mean_ignorance == np.max(max_size_confidence_mean_ignorance))[0])
        mcs_size_confidence_real_regret[i,:] = [mcs_res_zero[max_size_max_confidence_mcs_idx_zero],
                                                mcs_res_ignorance[max_size_max_confidence_mcs_idx_ignorance]]

        mcs_best_real_regret[i,:] = [np.min(mcs_res_zero),
                                     np.min(mcs_res_ignorance)]
        
    data_zero_non_zero = [data_classic['real_regret_classic'][data_zero['inconsistency_zero'] != 0],
                          data_zero['real_regret_zero'][data_zero['inconsistency_zero'] != 0],
                          data_zero_correction['real_regret_zero_l_out_of_n'][data_zero['inconsistency_zero'] != 0],
                          mcs_best_real_regret[:,0][data_zero['inconsistency_zero'] != 0],
                          mcs_confidence_real_regret[:,0][data_zero['inconsistency_zero'] != 0],
                          mcs_size_confidence_real_regret[:,0][data_zero['inconsistency_zero'] != 0],
                          mcs_size_real_regret[:,0][data_zero['inconsistency_zero'] != 0],
                          data_zero_correction['real_regret_zero_epsilon'][data_zero['inconsistency_zero'] != 0]]
    
    data_ignorance_non_zero = [data_classic['real_regret_classic'][data_ignorance['inconsistency_ignorance'] != 0],
                               data_ignorance['real_regret_ignorance'][data_ignorance['inconsistency_ignorance'] != 0],
                               data_ignorance_correction['real_regret_ignorance_l_out_of_n'][data_ignorance['inconsistency_ignorance'] != 0],
                               mcs_best_real_regret[:,1][data_zero['inconsistency_zero'] != 0],
                               mcs_confidence_real_regret[:,1][data_ignorance['inconsistency_ignorance'] != 0],
                               mcs_size_confidence_real_regret[:,1][data_ignorance['inconsistency_ignorance'] != 0],
                               mcs_size_real_regret[:,1][data_ignorance['inconsistency_ignorance'] != 0],
                               data_ignorance_correction['real_regret_ignorance_epsilon'][data_ignorance['inconsistency_ignorance'] != 0]]

    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots()
    j = 0
    for data_css in [data_zero_non_zero, data_ignorance_non_zero]:
        for d in data_css:
            print(stats.shapiro(d)[1] < 0.05)
        y = np.asarray([0,5,10,15,20,25,30,35]) + np.ones(8) * j
        for i in range(0, len(data_css)):
            data = data_css[i]
            data_mean = np.mean(data)
            data_IC = stats.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data),
                              scale=stats.sem(data))
            ax.plot((data_IC[0],data_IC[1]),(y[i],y[i]), '-', color = cmap((2-j)/2))
            ax.plot((data_mean,data_mean),(y[i],y[i]), '|', color = cmap((2-j)/2))
        j+=1
    ax.set_xlim(-0.005,0.3)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[1:8] = ['classic', 'possibilist', 'l out of n', 'MCS best', 'MCS conf',
                   'MCS size + conf', 'MCS size', 'epsilon']
    ax.set_yticklabels(labels)
    line_zero = mlines.Line2D([], [], color=cmap(2/2), marker='|', linestyle='-',
                             label='zero')
    line_ignorance = mlines.Line2D([], [], color=cmap(1/2), marker='|', linestyle='-',
                             label='igorance')
    
    plt.legend(loc="right", handles=[line_zero, line_ignorance])
    ax.set_xlabel('Real regret', fontsize = 9)
    fig.set_dpi(300.0)
    tikzplotlib.save(path_results + conf_type + '_' + str(nb_parameters) + '_' + str(nb_questions) + '.tex')
    plt.savefig(path_results + conf_type + '_' + str(nb_parameters) + '_' + str(nb_questions) + '.png', dpi=300, bbox_inches = 'tight')
    
    np.savetxt(path_data + 'data_zero.csv', data_zero_non_zero, delimiter=",")
    np.savetxt(path_data + 'data_ignorance.csv', data_ignorance_non_zero, delimiter=",")
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 23:15:57 2019

@author: hokaihong
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings

#Neural network regression packages
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Concatenate, MaxPooling2D, Lambda
from keras.layers import Conv2D, Add, Reshape, Conv1D, TimeDistributed, Permute
from keras.optimizers import RMSprop, Adam, SGD 
from keras.models import load_model

import keras.backend as K

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import DataConversionWarning

#============= minmax scaler warning ignore ==============#
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#============= import data and split to training/testing ==============#
path_import = r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Data\\'
path_export = r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Data\\'

df_raw = pd.read_pickle(path_import + 'RaceData3.pkl')
df = df_raw.copy()


# =============================================================================
# Utility functions

def Convert3D_X(X, size = 12):
    return X.reshape(int(X.shape[0]/size), size, X.shape[1])

def Convert3D_y(y, size = 12):
    return y.values.reshape(int(y.shape[0]/size), size)

def DCM_results(model, X_data, df_data):
    y_pred = model.predict(X_data)
    y_pred = pd.DataFrame(y_pred.reshape(y_pred.size))
    y_pred = y_pred.rename(columns = {0: 'prob_est'})
    results = df_data[['Race ID', 'Distance', 'Course', 'Race Class', 'Horse Name',
                       'Age', 'Draw', 'Rating', 'Loading', 'Jockey', 'Stable',
                       'Final Position', 'Time_val','oddbr', 'oddfn',
                       'd_place1', 'd_place2', 'd_place3', 'd_quin']]
    results['odds_prob_br'] = np.exp(df_data['log_odd_imply_prob'])
    results = results.reset_index(drop = True)
    results = pd.concat([results, y_pred], axis = 1)
    #Derive Plc odds column
    results['plcfn'] = 0
    results.loc[(results['Final Position'] == 1), 'plcfn'] = results['d_place1']
    results.loc[(results['Final Position'] == 2), 'plcfn'] = results['d_place2']
    results.loc[(results['Final Position'] == 3), 'plcfn'] = results['d_place3']
    
    results['rank_pred'] = results.groupby('Race ID')['prob_est'].rank(ascending = False)
    return results

# =============================================================================
#'oddfn', 'oddon', 'oddbr', 
col_var_fundamental = ['Rating', 'Age','Draw', 'Loading','avg_prev_rating', 
           'avg_dist_rating', 'avg_going_rating', 'new_dist',
           'LTD_horse_win_rate', 'LTD_jockey_win_cnt', 'LTD_jockey_win_rate',
           'ema_horse_quantile', 'ema_jockey_quantile', 'prev_margin', 'DOK',
           'class_up', 'class_down', 'class_new', 'class_bleeding', 'weight_diff',
           'prev_t_outpf']

col_var_tech = ['log_odd_imply_prob']

col_var = col_var_fundamental + col_var_tech

#L_MNL
#split = randint(3,len(col_var)-3)
#col_var_tmp = random.sample(col_var, len(col_var))
col_var_l = ['ema_horse_quantile', 'Rating', 'ema_jockey_quantile', 'log_odd_imply_prob']
col_var_nn = [x for x in col_var if x not in col_var_l]


# =============================================================================
#Base Strategy - betting on the hottest

df_base = df[['Race ID', 'Horse Name', 'oddbr', 'oddfn', 'Final Position']]
df_base['Rank_oddbr'] = df_base.groupby('Race ID')['oddbr'].rank(ascending = True)

cnt_win = df_base[(df_base['Final Position'] == 1) & (df_base['Rank_oddbr'] == 1)].shape[0]
cnt_total = len(df_base['Race ID'].unique())
#Accuracy
print('Accuracy of Hottest Runner:' + str(cnt_win/cnt_total))

#constant bet
df_base_win = df_base[(df_base['Final Position'] == 1) & (df_base['Rank_oddbr'] == 1)]
print('% Asset after all gamble:')
print(df_base_win['oddfn'].sum() / cnt_total)



# =============================================================================
#
""" Produce inputs for Keras models
"""


df_train = pd.read_pickle(r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Python\data_paper_DCM\df_train.pkl')
df_test = pd.read_pickle(r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Python\data_paper_DCM\df_test.pkl')


sc = StandardScaler()
race_size = 14
scaling = True

x_train_raw = df_train[col_var]
x_train_func_raw = df_train[col_var_fundamental]
x_train_tech_raw = df_train[col_var_tech]
x_train_l_raw = df_train[col_var_l]
x_train_nn_raw = df_train[col_var_nn]

x_test_raw = df_test[col_var]
x_test_func_raw = df_test[col_var_fundamental]
x_test_tech_raw = df_test[col_var_tech]
x_test_l_raw = df_test[col_var_l]
x_test_nn_raw = df_test[col_var_nn]


x_train = x_train_raw
x_train_func = x_train_func_raw
x_train_tech = x_train_tech_raw
x_train_l = x_train_l_raw
x_train_nn = x_train_nn_raw

x_test = x_test_raw
x_test_func = x_test_func_raw
x_test_tech = x_test_tech_raw
x_test_l = x_test_l_raw
x_test_nn = x_test_nn_raw

data_list = [x_train, x_train_func, x_train_tech, x_train_l, x_train_nn, 
             x_test, x_test_func, x_test_tech, x_test_l, x_test_nn]
temp_list = []
for x_data in data_list:
    if scaling:
        x_data = Convert3D_X(sc.fit_transform(x_data), size = race_size)
    else:
        x_data = Convert3D_X(x_data, size = race_size)
    x_data = np.transpose(x_data, (0,2,1))
    x_data = np.expand_dims(x_data, -1)
    temp_list.append(x_data)

[x_train, x_train_func, x_train_tech, x_train_l, x_train_nn, 
             x_test, x_test_func, x_test_tech, x_test_l, x_test_nn] = temp_list

y_train = Convert3D_y(df_train['win'], size = race_size)
y_test = Convert3D_y(df_test['win'], size = race_size)


    

#==============================================================================
"""Load the Keras models
"""
path_import_model = r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Python\data_paper_DCM\\'

cl = load_model(path_import_model + 'cl.h5')
cl2 = load_model(path_import_model + 'cl2.h5')
nn_mnl = load_model(path_import_model + 'nn_mnl.h5')
lmnl = load_model(path_import_model + 'lmnl.h5')






""" DCM related results
"""


summary_type = 'train'

if summary_type == 'train': 
    df_cl = DCM_results(cl, x_train, df_train)
    df_cl2 = DCM_results(cl2, [x_train_func,x_train_tech], df_train)
    df_nnmnl = DCM_results(nn_mnl, x_train, df_train)
    df_lmnl = DCM_results(lmnl, [x_train_l, x_train_nn], df_train)

else:
    df_cl = DCM_results(cl, x_test, df_test)
    df_cl2 = DCM_results(cl2, [x_test_func,x_test_tech], df_test)
    df_nnmnl = DCM_results(nn_mnl, x_test, df_test)
    df_lmnl = DCM_results(lmnl, [x_test_l, x_test_nn], df_test)

col_summary = df_cl.columns
col_summary = [x for x in col_summary if x not in ['prob_est', 'rank_pred']]

df_summary = df_cl.rename(columns = {'prob_est': 'prob_est_cl',
                                     'rank_pred': 'rank_pred_cl'})    
df_summary = df_summary.merge(df_cl2, on = col_summary).rename(columns = {'prob_est': 'prob_est_cl2',
                                                                          'rank_pred': 'rank_pred_cl2'})    
df_summary = df_summary.merge(df_nnmnl, on = col_summary).rename(columns = {'prob_est': 'prob_est_nnmnl',
                                                                            'rank_pred': 'rank_pred_nnmnl'})    
df_summary = df_summary.merge(df_lmnl, on = col_summary).rename(columns = {'prob_est': 'prob_est_lmnl',
                                                                           'rank_pred': 'rank_pred_lmnl'})    


    
df_summary.to_csv(r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Python\dcm_summary.csv')
df_summary_corr = df_summary[['rank_pred_cl', 'rank_pred_cl2', 'rank_pred_nnmnl', 'rank_pred_lmnl']].corr()
hist, bin_edges = np.histogram(df_summary['prob_est_cl'], bins = 20)

bin_range = [np.quantile(df_summary['prob_est_cl'], x) for x in np.linspace(0,1,11)]
np.histogram(df_summary['prob_est_cl'], bins = bin_range)



# =============================================================================
# Winner Prediction bets
#fig, axs = plt.subplots(nrows = 4,ncols = 2)
fig, axs = plt.subplots(nrows = 2,ncols = 1)
i = 0

models = ['cl', 'cl2', 'nnmnl', 'lmnl']
    
for summary_type in ['train']: #, 'test']:
    j = 0
    if summary_type == 'train': 
        df_cl = DCM_results(cl, x_train, df_train)
        df_cl2 = DCM_results(cl2, [x_train_func,x_train_tech], df_train)
        df_nnmnl = DCM_results(nn_mnl, x_train, df_train)
        df_lmnl = DCM_results(lmnl, [x_train_l, x_train_nn], df_train)
    
    else:
        df_cl = DCM_results(cl, x_test, df_test)
        df_cl2 = DCM_results(cl2, [x_test_func,x_test_tech], df_test)
        df_nnmnl = DCM_results(nn_mnl, x_test, df_test)
        df_lmnl = DCM_results(lmnl, [x_test_l, x_test_nn], df_test)

    col_summary = df_cl.columns
    col_summary = [x for x in col_summary if x not in ['prob_est', 'rank_pred']]
    
    df_summary = df_cl.rename(columns = {'prob_est': 'prob_est_cl',
                                         'rank_pred': 'rank_pred_cl'})    
    df_summary = df_summary.merge(df_cl2, on = col_summary).rename(columns = {'prob_est': 'prob_est_cl2',
                                                                              'rank_pred': 'rank_pred_cl2'})    
    df_summary = df_summary.merge(df_nnmnl, on = col_summary).rename(columns = {'prob_est': 'prob_est_nnmnl',
                                                                                'rank_pred': 'rank_pred_nnmnl'})    
    df_summary = df_summary.merge(df_lmnl, on = col_summary).rename(columns = {'prob_est': 'prob_est_lmnl',
                                                                               'rank_pred': 'rank_pred_lmnl'})
    
    df_winner = df_summary
    
    for model in models:
        df_winner_model = df_winner[(df_winner['rank_pred_{}'.format(model)] == 1) & (df_winner['oddbr'] < 50)].reset_index(drop = True)
        df_winner_model['cost'] = -1
        df_winner_model['dividend_win'] = (df_winner_model['Final Position'] == 1) * df_winner_model['oddfn']
        df_winner_model['dividend_plc'] = (df_winner_model['Final Position'] <= 3) * df_winner_model['plcfn']/10
        #Rebate is when the bet amount is larger than $10000 HKD, if losing, HKJC will rebate 10% of the bet
#        df_winner_model['rebate_win'] = (df_winner_model['dividend_win'] == 0) * 0.1
#        df_winner_model['rebate_plc'] = (df_winner_model['dividend_plc'] == 0) * 0.1
        
        df_winner_model['PL_win'] = df_winner_model['dividend_win'] + df_winner_model['cost']
        df_winner_model['PL_plc'] = df_winner_model['dividend_plc'] + df_winner_model['cost']
        
#        df_winner_model['PL_win_rebate'] = df_winner_model['dividend_win'] + df_winner_model['cost'] + df_winner_model['rebate_win']
#        df_winner_model['PL_plc_rebate'] = df_winner_model['dividend_plc'] + df_winner_model['cost'] + df_winner_model['rebate_plc']
#        
        df_winner_model['NetWealth_win_{}'.format(model)] = df_winner_model['PL_win'].cumsum()
        df_winner_model['NetWealth_plc_{}'.format(model)] = df_winner_model['PL_plc'].cumsum()
#        df_winner_model['NetWealth_win_{}_w/Rebate'.format(model)] = df_winner_model['PL_win_rebate'].cumsum()
#        df_winner_model['NetWealth_plc_{}_w/Rebate'.format(model)] = df_winner_model['PL_plc_rebate'].cumsum()
#        
        axs[0].set_xlabel('bet count')
        axs[0].set_ylabel('Cumulative Return')
        axs[1].set_xlabel('bet count')
        axs[1].set_ylabel('Cumulative Return')
        
#        
#        axs[j, i].set_xlabel('bet count')
#        axs[j, i].set_ylabel('Cumulative Return')
#        #axs[j, i].set_title('Winner strategy with {} on {}ing data'.format(model, summary_type))
        
#        df_winner_model['NetWealth_win_{}'.format(model)].plot(ax = axs[j, i], legend = True)
#        df_winner_model['NetWealth_plc_{}'.format(model)].plot(ax = axs[j, i], legend = True)
        df_winner_model['NetWealth_win_{}'.format(model)].plot(ax = axs[0], legend = True)
        df_winner_model['NetWealth_plc_{}'.format(model)].plot(ax = axs[1], legend = True)
#        df_winner_model['NetWealth_win_{}_w/Rebate'.format(model)].plot(ax = axs[j, i], legend = True)
#        df_winner_model['NetWealth_plc_{}_w/Rebate'.format(model)].plot(ax = axs[j, i], legend = True)
#        
        print('\n')
        print('################__{}__{}__################'.format(summary_type, model))
        print('Bet count: {}'.format(df_winner_model.shape[0]))
        print('count Win Pool bets that wins: {} ({:.2f}%)'.format((df_winner_model['PL_win'] > 0).sum(), 100 * (df_winner_model['PL_win'] > 0).sum() / df_winner_model.shape[0] ))
        print('count Place Pool bets that wins: {} ({:.2f}%)'.format((df_winner_model['PL_plc'] > 0).sum(), 100 * (df_winner_model['PL_plc'] > 0).sum() / df_winner_model.shape[0]))
        print('% gain of Win pool bets: {:.2f} ({:.2f}%)'.format(df_winner_model['NetWealth_win_{}'.format(model)].iloc[-1] , 100 * df_winner_model['NetWealth_win_{}'.format(model)].iloc[-1] / df_winner_model.shape[0]))
#        print('% gain of Win pool bets with Rebate: {:.2f} ({:.2f}%)'.format(df_winner_model['NetWealth_win_{}_w/Rebate'.format(model)].iloc[-1], 100 * df_winner_model['NetWealth_win_{}_w/Rebate'.format(model)].iloc[-1] / df_winner_model.shape[0]))
        print('% gain of Place pool bets: {:.2f} ({:.2f}%)'.format(df_winner_model['NetWealth_plc_{}'.format(model)].iloc[-1], 100 * df_winner_model['NetWealth_plc_{}'.format(model)].iloc[-1] / df_winner_model.shape[0]))
#        print('% gain of Place pool bets with Rebate: {:.2f} ({:.2f}%)'.format(df_winner_model['NetWealth_plc_{}_w/Rebate'.format(model)].iloc[-1], 100 * df_winner_model['NetWealth_plc_{}_w/Rebate'.format(model)].iloc[-1] / df_winner_model.shape[0]))
        
        j = j + 1
        
    i = i + 1

plt.show()





# =============================================================================
# #Expected Value based bets
fig, axs = plt.subplots(nrows = 4,ncols = 2)
i = 0
prob_lb = 0

models = ['cl'] #, 'cl2', 'nnmnl', 'lmnl']
    
for summary_type in ['train']: #, 'test']:
    j = 0
    if summary_type == 'train': 
        df_cl = DCM_results(cl, x_train, df_train)
        df_cl2 = DCM_results(cl2, [x_train_func,x_train_tech], df_train)
        df_nnmnl = DCM_results(nn_mnl, x_train, df_train)
        df_lmnl = DCM_results(lmnl, [x_train_l, x_train_nn], df_train)
    
    else:
        df_cl = DCM_results(cl, x_test, df_test)
        df_cl2 = DCM_results(cl2, [x_test_func,x_test_tech], df_test)
        df_nnmnl = DCM_results(nn_mnl, x_test, df_test)
        df_lmnl = DCM_results(lmnl, [x_test_l, x_test_nn], df_test)

    col_summary = df_cl.columns
    col_summary = [x for x in col_summary if x not in ['prob_est', 'rank_pred']]
    
    df_summary = df_cl.rename(columns = {'prob_est': 'prob_est_cl',
                                         'rank_pred': 'rank_pred_cl'})    
    df_summary = df_summary.merge(df_cl2, on = col_summary).rename(columns = {'prob_est': 'prob_est_cl2',
                                                                              'rank_pred': 'rank_pred_cl2'})    
    df_summary = df_summary.merge(df_nnmnl, on = col_summary).rename(columns = {'prob_est': 'prob_est_nnmnl',
                                                                                'rank_pred': 'rank_pred_nnmnl'})    
    df_summary = df_summary.merge(df_lmnl, on = col_summary).rename(columns = {'prob_est': 'prob_est_lmnl',
                                                                               'rank_pred': 'rank_pred_lmnl'})
    
    df_EV = df_summary
    df_EV['EV_cl'] = df_EV['oddbr'] * df_EV['prob_est_cl']
    df_EV['EV_cl2'] = df_EV['oddbr'] * df_EV['prob_est_cl2']
    df_EV['EV_nnmnl'] = df_EV['oddbr'] * df_EV['prob_est_nnmnl']
    df_EV['EV_lmnl'] = df_EV['oddbr'] * df_EV['prob_est_lmnl']
    
    
    for model in models:
        df_EV_model = df_EV[(df_EV['EV_{}'.format(model)] > 1) & 
                            (df_EV['oddbr'] < 50) &
                            (df_EV['prob_est_{}'.format(model)] > prob_lb)].reset_index(drop = True)
        df_EV_model['cost'] = -1
        df_EV_model['dividend_win'] = (df_EV_model['Final Position'] == 1) * df_EV_model['oddfn']
        df_EV_model['dividend_plc'] = (df_EV_model['Final Position'] <= 3) * df_EV_model['plcfn']/10
        #Rebate is when the bet amount is larger than $10000 HKD, if losing, HKJC will rebate 10% of the bet
        df_EV_model['rebate_win'] = (df_EV_model['dividend_win'] == 0) * 0.1
        df_EV_model['rebate_plc'] = (df_EV_model['dividend_plc'] == 0) * 0.1
        
        df_EV_model['PL_win'] = df_EV_model['dividend_win'] + df_EV_model['cost']
        df_EV_model['PL_plc'] = df_EV_model['dividend_plc'] + df_EV_model['cost']
        
        df_EV_model['PL_win_rebate'] = df_EV_model['dividend_win'] + df_EV_model['cost'] + df_EV_model['rebate_win']
        df_EV_model['PL_plc_rebate'] = df_EV_model['dividend_plc'] + df_EV_model['cost'] + df_EV_model['rebate_plc']
        
        df_EV_model['NetWealth_win_{}'.format(model)] = df_EV_model['PL_win'].cumsum()
        df_EV_model['NetWealth_plc_{}'.format(model)] = df_EV_model['PL_plc'].cumsum()
        df_EV_model['NetWealth_win_{}_w/Rebate'.format(model)] = df_EV_model['PL_win_rebate'].cumsum()
        df_EV_model['NetWealth_plc_{}_w/Rebate'.format(model)] = df_EV_model['PL_plc_rebate'].cumsum()
        
        axs[j, i].set_xlabel('bet count')
        axs[j, i].set_ylabel('Cumulative Return')
        axs[j, i].set_title('EV strategy with {} on {}ing data'.format(model, summary_type))
        
        df_EV_model['NetWealth_win_{}'.format(model)].plot(ax = axs[j, i], legend = True)
        df_EV_model['NetWealth_plc_{}'.format(model)].plot(ax = axs[j, i], legend = True)
        df_EV_model['NetWealth_win_{}_w/Rebate'.format(model)].plot(ax = axs[j, i], legend = True)
        df_EV_model['NetWealth_plc_{}_w/Rebate'.format(model)].plot(ax = axs[j, i], legend = True)
        
        print('\n')
        print('################__{}__{}__################'.format(summary_type, model))
        print('Bet count: {}'.format(df_EV_model.shape[0]))
        print('count Win Pool bets that wins: {} ({:.2f}%)'.format((df_EV_model['PL_win'] > 0).sum(), 100 * (df_EV_model['PL_win'] > 0).sum() / df_EV_model.shape[0] ))
        print('count Place Pool bets that wins: {} ({:.2f}%)'.format((df_EV_model['PL_plc'] > 0).sum(), 100 * (df_EV_model['PL_plc'] > 0).sum() / df_EV_model.shape[0]))
        print('% gain of Win pool bets: {:.2f} ({:.2f}%)'.format(df_EV_model['NetWealth_win_{}'.format(model)].iloc[-1] , 100 * df_EV_model['NetWealth_win_{}'.format(model)].iloc[-1] / df_EV_model.shape[0]))
        print('% gain of Win pool bets with Rebate: {:.2f} ({:.2f}%)'.format(df_EV_model['NetWealth_win_{}_w/Rebate'.format(model)].iloc[-1], 100 * df_EV_model['NetWealth_win_{}_w/Rebate'.format(model)].iloc[-1] / df_EV_model.shape[0]))
        print('% gain of Place pool bets: {:.2f} ({:.2f}%)'.format(df_EV_model['NetWealth_plc_{}'.format(model)].iloc[-1], 100 * df_EV_model['NetWealth_plc_{}'.format(model)].iloc[-1] / df_EV_model.shape[0]))
        print('% gain of Place pool bets with Rebate: {:.2f} ({:.2f}%)'.format(df_EV_model['NetWealth_plc_{}_w/Rebate'.format(model)].iloc[-1], 100 * df_EV_model['NetWealth_plc_{}_w/Rebate'.format(model)].iloc[-1] / df_EV_model.shape[0]))
        
        j = j + 1
        
    i = i + 1

plt.show()



df_EV_win = df_EV_model[df_EV_model['PL_win_rebate'] > 0]
df_EV_lose = df_EV_model[df_EV_model['PL_win_rebate'] < 0]

sns.distplot(df_EV_win['prob_est_lmnl'], norm_hist = True)
sns.distplot(df_EV_lose['prob_est_lmnl'], norm_hist = True)



# =============================================================================
# Kelly Criteria based bets

fig, axs = plt.subplots(nrows = 4,ncols = 2)

i = 0
models = ['cl', 'cl2', 'nnmnl', 'lmnl']
cap = 10
frac = 0.0235
f_bound = 0
prob_lb = 0
odds_lb = 0.1


for summary_type in ['train', 'test']:
    j = 0
    
    if summary_type == 'train': 
        df_cl = DCM_results(cl, x_train, df_train)
        df_cl2 = DCM_results(cl2, [x_train_func,x_train_tech], df_train)
        df_nnmnl = DCM_results(nn_mnl, x_train, df_train)
        df_lmnl = DCM_results(lmnl, [x_train_l, x_train_nn], df_train)
    
    else:
        df_cl = DCM_results(cl, x_test, df_test)
        df_cl2 = DCM_results(cl2, [x_test_func,x_test_tech], df_test)
        df_nnmnl = DCM_results(nn_mnl, x_test, df_test)
        df_lmnl = DCM_results(lmnl, [x_test_l, x_test_nn], df_test)

    col_summary = df_cl.columns
    col_summary = [x for x in col_summary if x not in ['prob_est', 'rank_pred']]
    
    df_summary = df_cl.rename(columns = {'prob_est': 'prob_est_cl',
                                         'rank_pred': 'rank_pred_cl'})    
    df_summary = df_summary.merge(df_cl2, on = col_summary).rename(columns = {'prob_est': 'prob_est_cl2',
                                                                              'rank_pred': 'rank_pred_cl2'})    
    df_summary = df_summary.merge(df_nnmnl, on = col_summary).rename(columns = {'prob_est': 'prob_est_nnmnl',
                                                                                'rank_pred': 'rank_pred_nnmnl'})    
    df_summary = df_summary.merge(df_lmnl, on = col_summary).rename(columns = {'prob_est': 'prob_est_lmnl',
                                                                               'rank_pred': 'rank_pred_lmnl'})
    
    df_Kelly = df_summary
    
    df_Kelly['f_cl'] = ((df_Kelly['oddbr'] - 1) * df_Kelly['prob_est_cl'] - ( 1 - df_Kelly['prob_est_cl'])) / (df_Kelly['oddbr'] - 1)
    df_Kelly['f_cl2'] = ((df_Kelly['oddbr'] - 1) * df_Kelly['prob_est_cl2'] - ( 1 - df_Kelly['prob_est_cl2'])) / (df_Kelly['oddbr'] - 1)
    df_Kelly['f_nnmnl'] = ((df_Kelly['oddbr'] - 1) * df_Kelly['prob_est_nnmnl'] - ( 1 - df_Kelly['prob_est_nnmnl'])) / (df_Kelly['oddbr'] - 1)
    df_Kelly['f_lmnl'] = ((df_Kelly['oddbr'] - 1) * df_Kelly['prob_est_lmnl'] - ( 1 - df_Kelly['prob_est_lmnl'])) / (df_Kelly['oddbr'] - 1)

    #Apply a a fraction and cap to betting size
    df_Kelly['f_cl'] = (df_Kelly['f_cl'] * frac).clip(lower = None, upper = cap)
    df_Kelly['f_cl2'] = (df_Kelly['f_cl2'] * frac).clip(lower = None, upper = cap)
    df_Kelly['f_nnmnl'] = (df_Kelly['f_nnmnl'] * frac).clip(lower = None, upper = cap)
    df_Kelly['f_lmnl'] = (df_Kelly['f_lmnl'] * frac).clip(lower = None, upper = cap)
    
    for model in models:
    #filter for bets to be made
        df_Kelly_bet = df_Kelly[(df_Kelly['f_{}'.format(model)] > f_bound) & 
                                (df_Kelly['oddbr'] > odds_lb) & 
                                (df_Kelly['oddbr'] < 50) &
                                (df_Kelly['prob_est_{}'.format(model)] > prob_lb)].reset_index(drop = True)

        df_Kelly_bet['cost'] = -df_Kelly_bet['f_{}'.format(model)]
        df_Kelly_bet['dividend_win'] = (df_Kelly_bet['Final Position'] == 1) * df_Kelly_bet['f_{}'.format(model)] * df_Kelly_bet['oddfn']
        df_Kelly_bet['dividend_plc'] = (df_Kelly_bet['Final Position'] <= 3) * df_Kelly_bet['f_{}'.format(model)] * df_Kelly_bet['plcfn']/10
        
        df_Kelly_bet['rebate_win'] = (df_Kelly_bet['dividend_win'] == 0) * df_Kelly_bet['f_{}'.format(model)] * 0.1
        df_Kelly_bet['rebate_plc'] = (df_Kelly_bet['dividend_plc'] == 0) * df_Kelly_bet['f_{}'.format(model)] * 0.1
        
        #PL here is in % terms
        df_Kelly_bet['PL_win'] = 1 + df_Kelly_bet['dividend_win'] + df_Kelly_bet['cost']
        df_Kelly_bet['PL_plc'] = 1 + df_Kelly_bet['dividend_plc'] + df_Kelly_bet['cost']    
        df_Kelly_bet['PL_win_rebate'] = 1 + df_Kelly_bet['dividend_win'] + df_Kelly_bet['cost'] + df_Kelly_bet['rebate_win']
        df_Kelly_bet['PL_plc_rebate'] = 1 + df_Kelly_bet['dividend_plc'] + df_Kelly_bet['cost'] + df_Kelly_bet['rebate_plc']
            
        df_Kelly_bet['NetWealth_win_{}'.format(model)] = df_Kelly_bet['PL_win'].cumprod()
        df_Kelly_bet['NetWealth_plc_{}'.format(model)] = df_Kelly_bet['PL_plc'].cumprod()
        df_Kelly_bet['NetWealth_win_{}_w/Rebate'.format(model)] = df_Kelly_bet['PL_win_rebate'].cumprod()
        df_Kelly_bet['NetWealth_plc_{}_w/Rebate'.format(model)] = df_Kelly_bet['PL_plc_rebate'].cumprod()
        
        axs[j, i].set_xlabel('bet count')
        axs[j, i].set_ylabel('Cumulative Return')
        axs[j, i].set_title('Kelly strategy with {} on {}ing data'.format(model, summary_type))
        
        df_Kelly_bet['NetWealth_win_{}'.format(model)].plot(ax = axs[j, i], legend = True)
        df_Kelly_bet['NetWealth_plc_{}'.format(model)].plot(ax = axs[j, i], legend = True)
        df_Kelly_bet['NetWealth_win_{}_w/Rebate'.format(model)].plot(ax = axs[j, i], legend = True)
        df_Kelly_bet['NetWealth_plc_{}_w/Rebate'.format(model)].plot(ax = axs[j, i], legend = True)
    
        print('################__{}__{}__################'.format(summary_type, model))
        print('Bet count: {}'.format(df_Kelly_bet.shape[0]))
        print('count Win Pool bets that wins: {} ({:.2f}%)'.format((df_Kelly_bet['PL_win'] > 1).sum(), 100 * (df_Kelly_bet['PL_win'] > 1).sum() / df_Kelly_bet.shape[0] ))
        print('count Place Pool bets that wins: {} ({:.2f}%)'.format((df_Kelly_bet['PL_plc'] > 1).sum(), 100 * (df_Kelly_bet['PL_plc'] > 1).sum() / df_Kelly_bet.shape[0]))
        print('Final Asset of Win pool bets: {:.2f} ({:.2f}%)'.format(df_Kelly_bet['NetWealth_win_{}'.format(model)].iloc[-1] , 100 * (df_Kelly_bet['NetWealth_win_{}'.format(model)].iloc[-1] - 1)))
        print('Final Asset of Win pool bets with Rebate: {:.2f} ({:.2f}%)'.format(df_Kelly_bet['NetWealth_win_{}_w/Rebate'.format(model)].iloc[-1], 100 * (df_Kelly_bet['NetWealth_win_{}_w/Rebate'.format(model)].iloc[-1] - 1)))
        print('Final Asset of Place pool bets: {:.2f} ({:.2f}%)'.format(df_Kelly_bet['NetWealth_plc_{}'.format(model)].iloc[-1], 100 * (df_Kelly_bet['NetWealth_plc_{}'.format(model)].iloc[-1] - 1)))
        print('Final Asset of Place pool bets with Rebate: {:.2f} ({:.2f}%)'.format(df_Kelly_bet['NetWealth_plc_{}_w/Rebate'.format(model)].iloc[-1], 100 * (df_Kelly_bet['NetWealth_plc_{}_w/Rebate'.format(model)].iloc[-1] - 1)))
        
        j = j + 1
        
    i = i + 1
    
plt.show()



fig, ax1 = plt.subplots()
ax1.set_xlabel('Betting Size')
ax1.set_ylabel('Frequency')
df_Kelly['f_cl'].hist(bins = 100, ax = ax1)

print(df_Kelly['f_cl'].describe() * 100)


df_Kelly_bet['win'] = (df_Kelly_bet['dividend_win'] > 0) * 1
sns.scatterplot(x = df_Kelly_bet['odds_prob_br'], y = df_Kelly_bet['prob_est_lmnl'], hue = df_Kelly_bet['win'])



df_Kelly_bet['net_PL'] = df_Kelly_bet['PL_win_rebate'] - 1
df_Kelly_win = df_Kelly_bet[df_Kelly_bet['net_PL'] > 0]

df_Kelly_lose = df_Kelly_bet[df_Kelly_bet['net_PL'] < 0]

sns.distplot(df_Kelly_win['prob_est_lmnl'], norm_hist = True)
sns.distplot(df_Kelly_lose['prob_est_lmnl'], norm_hist = True)

# =============================================================================
# DCM Ensembled (avg Rank)

fig, axs = plt.subplots(ncols = 2)

i = 0
        
for summary_type in ['train', 'test']:
    if summary_type == 'train': 
        df_cl = DCM_results(cl, x_train, df_train)
        df_cl2 = DCM_results(cl2, [x_train_func,x_train_tech], df_train)
        df_nnmnl = DCM_results(nn_mnl, x_train, df_train)
        df_lmnl = DCM_results(lmnl, [x_train_l, x_train_nn], df_train)
    
    else:
        df_cl = DCM_results(cl, x_test, df_test)
        df_cl2 = DCM_results(cl2, [x_test_func,x_test_tech], df_test)
        df_nnmnl = DCM_results(nn_mnl, x_test, df_test)
        df_lmnl = DCM_results(lmnl, [x_test_l, x_test_nn], df_test)

    col_summary = df_cl.columns
    col_summary = [x for x in col_summary if x not in ['prob_est', 'rank_pred']]
    
    df_summary = df_cl.rename(columns = {'prob_est': 'prob_est_cl',
                                         'rank_pred': 'rank_pred_cl'})    
    df_summary = df_summary.merge(df_cl2, on = col_summary).rename(columns = {'prob_est': 'prob_est_cl2',
                                                                              'rank_pred': 'rank_pred_cl2'})    
    df_summary = df_summary.merge(df_nnmnl, on = col_summary).rename(columns = {'prob_est': 'prob_est_nnmnl',
                                                                                'rank_pred': 'rank_pred_nnmnl'})    
    df_summary = df_summary.merge(df_lmnl, on = col_summary).rename(columns = {'prob_est': 'prob_est_lmnl',
                                                                               'rank_pred': 'rank_pred_lmnl'})

    df_ensem = df_summary
    
    #Least average rank of 4 models is the predicted winner
    df_ensem['avg_rank'] = (df_ensem['rank_pred_cl'] + df_ensem['rank_pred_cl2'] +
                         df_ensem['rank_pred_nnmnl'] + df_ensem['rank_pred_lmnl']) / 4
    df_ensem['rank_pred_avg'] = df_ensem.groupby('Race ID')['avg_rank'].rank(ascending = True)
    
    #simple bets
    df_ensem = df_ensem[df_ensem['rank_pred_avg'] < 2].reset_index(drop = True)
    df_ensem['cost'] = -1
    df_ensem['dividend_win'] = (df_ensem['Final Position'] == 1) * df_ensem['oddfn']
    df_ensem['dividend_plc'] = (df_ensem['Final Position'] <= 3) * df_ensem['plcfn'] / 10
    df_ensem['PL_win'] = df_ensem['dividend_win'] + df_ensem['cost']
    df_ensem['PL_plc'] = df_ensem['dividend_plc'] + df_ensem['cost']
    df_ensem['NetWealth_win'] = df_ensem['PL_win'].cumsum()
    df_ensem['NetWealth_plc'] = df_ensem['PL_plc'].cumsum()
    
    axs[i].set_xlabel('bet count')
    axs[i].set_ylabel('Cumulative Return')
    axs[i].set_title('Ensembled Model on {}ing data'.format(summary_type))
        
    df_ensem['NetWealth_win'].plot(ax = axs[i], legend = True)
    df_ensem['NetWealth_plc'].plot(ax = axs[i], legend = True)
    
    
    print('############################' + summary_type + '#################################')
    print('Bet count: {}'.format(df_ensem.shape[0]))
    print('count Win Pool bets that wins: {} ({:.2f}%)'.format((df_ensem['PL_win'] > 0).sum(), 100 * (df_ensem['PL_win'] > 0).sum() / df_ensem.shape[0]))
    print('count Place Pool bets that wins: {} ({:.2f}%)'.format((df_ensem['PL_plc'] > 0).sum(), 100 * (df_ensem['PL_plc'] > 0).sum() / df_ensem.shape[0]))
    print('% gain of Win pool bets: {:.2f} ({:.2f}%)'.format(df_ensem['NetWealth_win'].iloc[-1], 100 * df_ensem['NetWealth_win'].iloc[-1] / df_ensem.shape[0]))
    print('% gain of Place pool bets: {:.2f} ({:.2f}%)'.format(df_ensem['NetWealth_plc'].iloc[-1], 100 * df_ensem['NetWealth_plc'].iloc[-1] / df_ensem.shape[0]))
    
    i = i + 1
    




# =============================================================================
# DCM Ensembled (avg winning prob)

fig, axs = plt.subplots(nrows = 3, ncols = 2)

i = 0
cap = 10
frac = 0.0235
f_bound = 0
prob_lb = 0
odds_lb = 0.1


        
for summary_type in ['train', 'test']:
    if summary_type == 'train': 
        df_cl = DCM_results(cl, x_train, df_train)
        df_cl2 = DCM_results(cl2, [x_train_func,x_train_tech], df_train)
        df_nnmnl = DCM_results(nn_mnl, x_train, df_train)
        df_lmnl = DCM_results(lmnl, [x_train_l, x_train_nn], df_train)
    
    else:
        df_cl = DCM_results(cl, x_test, df_test)
        df_cl2 = DCM_results(cl2, [x_test_func,x_test_tech], df_test)
        df_nnmnl = DCM_results(nn_mnl, x_test, df_test)
        df_lmnl = DCM_results(lmnl, [x_test_l, x_test_nn], df_test)

    col_summary = df_cl.columns
    col_summary = [x for x in col_summary if x not in ['prob_est', 'rank_pred']]
    
    df_summary = df_cl.rename(columns = {'prob_est': 'prob_est_cl',
                                         'rank_pred': 'rank_pred_cl'})    
    df_summary = df_summary.merge(df_cl2, on = col_summary).rename(columns = {'prob_est': 'prob_est_cl2',
                                                                              'rank_pred': 'rank_pred_cl2'})    
    df_summary = df_summary.merge(df_nnmnl, on = col_summary).rename(columns = {'prob_est': 'prob_est_nnmnl',
                                                                                'rank_pred': 'rank_pred_nnmnl'})    
    df_summary = df_summary.merge(df_lmnl, on = col_summary).rename(columns = {'prob_est': 'prob_est_lmnl',
                                                                               'rank_pred': 'rank_pred_lmnl'})

    df_ensem = df_summary
    
    #Least average rank of 4 models is the predicted winner
    df_ensem['prob_est_avg'] = (df_ensem['prob_est_cl'] + df_ensem['prob_est_cl2'] +
                         df_ensem['prob_est_nnmnl'] + df_ensem['prob_est_lmnl']) / 4
    df_ensem['rank_pred_avg'] = df_ensem.groupby('Race ID')['prob_est_avg'].rank(ascending = False)
    df_ensem['EV_avg'] = df_ensem['oddbr'] * df_ensem['prob_est_avg']
    df_ensem['f_avg'] = ((df_ensem['oddbr'] - 1) * df_ensem['prob_est_avg'] - ( 1 - df_ensem['prob_est_avg'])) / (df_ensem['oddbr'] - 1)
    df_ensem['f_avg'] = (df_ensem['f_avg'] * frac).clip(lower = None, upper = cap)
    
    #simple bets
    df_ensem_simple = df_ensem[df_ensem['rank_pred_avg'] == 1].reset_index(drop = True)
    
    df_ensem_simple['cost'] = -1
    df_ensem_simple['dividend_win'] = (df_ensem_simple['Final Position'] == 1) * df_ensem_simple['oddfn']
    df_ensem_simple['dividend_plc'] = (df_ensem_simple['Final Position'] <= 3) * df_ensem_simple['plcfn']/10
    #Rebate is when the bet amount is larger than $10000 HKD, if losing, HKJC will rebate 10% of the bet
    df_ensem_simple['rebate_win'] = (df_ensem_simple['dividend_win'] == 0) * 0.1
    df_ensem_simple['rebate_plc'] = (df_ensem_simple['dividend_plc'] == 0) * 0.1
    
    df_ensem_simple['PL_win'] = df_ensem_simple['dividend_win'] + df_ensem_simple['cost']
    df_ensem_simple['PL_plc'] = df_ensem_simple['dividend_plc'] + df_ensem_simple['cost']
    
    df_ensem_simple['PL_win_rebate'] = df_ensem_simple['dividend_win'] + df_ensem_simple['cost'] + df_ensem_simple['rebate_win']
    df_ensem_simple['PL_plc_rebate'] = df_ensem_simple['dividend_plc'] + df_ensem_simple['cost'] + df_ensem_simple['rebate_plc']
    
    df_ensem_simple['NetWealth_win'] = df_ensem_simple['PL_win'].cumsum()
    df_ensem_simple['NetWealth_plc'] = df_ensem_simple['PL_plc'].cumsum()
    df_ensem_simple['NetWealth_win_w/Rebate'] = df_ensem_simple['PL_win_rebate'].cumsum()
    df_ensem_simple['NetWealth_plc_w/Rebate'] = df_ensem_simple['PL_plc_rebate'].cumsum()

    axs[0,i].set_xlabel('bet count')
    axs[0,i].set_ylabel('Cumulative P&L')
    
    df_ensem_simple['NetWealth_win'].plot(ax = axs[0, i], legend = True)
    df_ensem_simple['NetWealth_plc'].plot(ax = axs[0, i], legend = True)
    df_ensem_simple['NetWealth_win_w/Rebate'].plot(ax = axs[0, i], legend = True)
    df_ensem_simple['NetWealth_plc_w/Rebate'].plot(ax = axs[0, i], legend = True)
    
    print('\n')
    print('###########___Winner Prediction___#################' + summary_type + '#################################')
    print('Bet count: {}'.format(df_ensem_simple.shape[0]))
    print('count Win Pool bets that wins: {} ({:.2f}%)'.format((df_ensem_simple['PL_win'] > 0).sum(), 100 * (df_ensem_simple['PL_win'] > 0).sum() / df_ensem_simple.shape[0] ))
    print('count Place Pool bets that wins: {} ({:.2f}%)'.format((df_ensem_simple['PL_plc'] > 0).sum(), 100 * (df_ensem_simple['PL_plc'] > 0).sum() / df_ensem_simple.shape[0]))
    print('% gain of Win pool bets: {:.2f} ({:.2f}%)'.format(df_ensem_simple['NetWealth_win'].iloc[-1] , 100 * df_ensem_simple['NetWealth_win'].iloc[-1] / df_ensem_simple.shape[0]))
    print('% gain of Win pool bets with Rebate: {:.2f} ({:.2f}%)'.format(df_ensem_simple['NetWealth_win_w/Rebate'].iloc[-1], 100 * df_ensem_simple['NetWealth_win_w/Rebate'].iloc[-1] / df_ensem_simple.shape[0]))
    print('% gain of Place pool bets: {:.2f} ({:.2f}%)'.format(df_ensem_simple['NetWealth_plc'].iloc[-1], 100 * df_ensem_simple['NetWealth_plc'].iloc[-1] / df_ensem_simple.shape[0]))
    print('% gain of Place pool bets with Rebate: {:.2f} ({:.2f}%)'.format(df_ensem_simple['NetWealth_plc_w/Rebate'].iloc[-1], 100 * df_ensem_simple['NetWealth_plc_w/Rebate'].iloc[-1] / df_ensem_simple.shape[0]))
       
    
    
    
    #EV based bets
    df_ensem_EV = df_ensem[(df_ensem['EV_avg'] > 1) & (df_ensem['oddbr'] > 0.1) & (df_ensem['oddbr'] < 50)].reset_index()

    df_ensem_EV['cost'] = -1
    df_ensem_EV['dividend_win'] = (df_ensem_EV['Final Position'] == 1) * df_ensem_EV['oddfn']
    df_ensem_EV['dividend_plc'] = (df_ensem_EV['Final Position'] <= 3) * df_ensem_EV['plcfn']/10
    #Rebate is when the bet amount is larger than $10000 HKD, if losing, HKJC will rebate 10% of the bet
    df_ensem_EV['rebate_win'] = (df_ensem_EV['dividend_win'] == 0) * 0.1
    df_ensem_EV['rebate_plc'] = (df_ensem_EV['dividend_plc'] == 0) * 0.1
    
    df_ensem_EV['PL_win'] = df_ensem_EV['dividend_win'] + df_ensem_EV['cost']
    df_ensem_EV['PL_plc'] = df_ensem_EV['dividend_plc'] + df_ensem_EV['cost']
    
    df_ensem_EV['PL_win_rebate'] = df_ensem_EV['dividend_win'] + df_ensem_EV['cost'] + df_ensem_EV['rebate_win']
    df_ensem_EV['PL_plc_rebate'] = df_ensem_EV['dividend_plc'] + df_ensem_EV['cost'] + df_ensem_EV['rebate_plc']
    
    df_ensem_EV['NetWealth_win'] = df_ensem_EV['PL_win'].cumsum()
    df_ensem_EV['NetWealth_plc'] = df_ensem_EV['PL_plc'].cumsum()
    df_ensem_EV['NetWealth_win_w/Rebate'] = df_ensem_EV['PL_win_rebate'].cumsum()
    df_ensem_EV['NetWealth_plc_w/Rebate'] = df_ensem_EV['PL_plc_rebate'].cumsum()

    axs[1,i].set_xlabel('bet count')
    axs[1,i].set_ylabel('Cumulative P&L')
    
    df_ensem_EV['NetWealth_win'].plot(ax = axs[1, i], legend = True)
    df_ensem_EV['NetWealth_plc'].plot(ax = axs[1, i], legend = True)
    df_ensem_EV['NetWealth_win_w/Rebate'].plot(ax = axs[1, i], legend = True)
    df_ensem_EV['NetWealth_plc_w/Rebate'].plot(ax = axs[1, i], legend = True)
    
    print('\n')
    print('###########___EV___#################' + summary_type + '#################################')
    print('Bet count: {}'.format(df_ensem_EV.shape[0]))
    print('count Win Pool bets that wins: {} ({:.2f}%)'.format((df_ensem_EV['PL_win'] > 0).sum(), 100 * (df_ensem_EV['PL_win'] > 0).sum() / df_ensem_EV.shape[0] ))
    print('count Place Pool bets that wins: {} ({:.2f}%)'.format((df_ensem_EV['PL_plc'] > 0).sum(), 100 * (df_ensem_EV['PL_plc'] > 0).sum() / df_ensem_EV.shape[0]))
    print('% gain of Win pool bets: {:.2f} ({:.2f}%)'.format(df_ensem_EV['NetWealth_win'].iloc[-1] , 100 * df_ensem_EV['NetWealth_win'].iloc[-1] / df_ensem_EV.shape[0]))
    print('% gain of Win pool bets with Rebate: {:.2f} ({:.2f}%)'.format(df_ensem_EV['NetWealth_win_w/Rebate'].iloc[-1], 100 * df_ensem_EV['NetWealth_win_w/Rebate'].iloc[-1] / df_ensem_EV.shape[0]))
    print('% gain of Place pool bets: {:.2f} ({:.2f}%)'.format(df_ensem_EV['NetWealth_plc'].iloc[-1], 100 * df_ensem_EV['NetWealth_plc'].iloc[-1] / df_ensem_EV.shape[0]))
    print('% gain of Place pool bets with Rebate: {:.2f} ({:.2f}%)'.format(df_ensem_EV['NetWealth_plc_w/Rebate'].iloc[-1], 100 * df_ensem_EV['NetWealth_plc_w/Rebate'].iloc[-1] / df_ensem_EV.shape[0]))
    
    
    
    #Kelly based bets
    df_ensem_Kelly = df_ensem[(df_ensem['f_avg'] > f_bound) & 
                            (df_ensem['oddbr'] > odds_lb) & 
                            (df_ensem['oddbr'] < 50) &
                            (df_ensem['prob_est_avg'] > prob_lb)].reset_index(drop = True)
    
    df_ensem_Kelly['cost'] = -df_ensem_Kelly['f_avg']
    df_ensem_Kelly['dividend_win'] = (df_ensem_Kelly['Final Position'] == 1) * df_ensem_Kelly['f_avg'] * df_ensem_Kelly['oddfn']
    df_ensem_Kelly['dividend_plc'] = (df_ensem_Kelly['Final Position'] <= 3) * df_ensem_Kelly['f_avg'] * df_ensem_Kelly['plcfn']/10
    
    df_ensem_Kelly['rebate_win'] = (df_ensem_Kelly['dividend_win'] == 0) * df_ensem_Kelly['f_avg'] * 0.1
    df_ensem_Kelly['rebate_plc'] = (df_ensem_Kelly['dividend_plc'] == 0) * df_ensem_Kelly['f_avg'] * 0.1
    
    #PL here is in % terms
    df_ensem_Kelly['PL_win'] = 1 + df_ensem_Kelly['dividend_win'] + df_ensem_Kelly['cost']
    df_ensem_Kelly['PL_plc'] = 1 + df_ensem_Kelly['dividend_plc'] + df_ensem_Kelly['cost']    
    df_ensem_Kelly['PL_win_rebate'] = 1 + df_ensem_Kelly['dividend_win'] + df_ensem_Kelly['cost'] + df_ensem_Kelly['rebate_win']
    df_ensem_Kelly['PL_plc_rebate'] = 1 + df_ensem_Kelly['dividend_plc'] + df_ensem_Kelly['cost'] + df_ensem_Kelly['rebate_plc']
        
    df_ensem_Kelly['NetWealth_win_avg'] = df_ensem_Kelly['PL_win'].cumprod()
    df_ensem_Kelly['NetWealth_plc_avg'] = df_ensem_Kelly['PL_plc'].cumprod()
    df_ensem_Kelly['NetWealth_win_avg_w/Rebate'] = df_ensem_Kelly['PL_win_rebate'].cumprod()
    df_ensem_Kelly['NetWealth_plc_avg_w/Rebate'] = df_ensem_Kelly['PL_plc_rebate'].cumprod()
    
    axs[2, i].set_xlabel('bet count')
    axs[2, i].set_ylabel('Asset Value')
    axs[2, i].set_title('Ensemble Kelly strategy on {}ing data'.format(summary_type))
    
    df_ensem_Kelly['NetWealth_win_avg'].plot(ax = axs[2, i], legend = True)
    df_ensem_Kelly['NetWealth_plc_avg'].plot(ax = axs[2, i], legend = True)
    df_ensem_Kelly['NetWealth_win_avg_w/Rebate'].plot(ax = axs[2, i], legend = True)
    df_ensem_Kelly['NetWealth_plc_avg_w/Rebate'].plot(ax = axs[2, i], legend = True)

    print('\n')
    print('################__Kelly__{}__################'.format(summary_type))
    print('Bet count: {}'.format(df_ensem_Kelly.shape[0]))
    print('count Win Pool bets that wins: {} ({:.2f}%)'.format((df_ensem_Kelly['PL_win'] > 1).sum(), 100 * (df_ensem_Kelly['PL_win'] > 1).sum() / df_ensem_Kelly.shape[0] ))
    print('count Place Pool bets that wins: {} ({:.2f}%)'.format((df_ensem_Kelly['PL_plc'] > 1).sum(), 100 * (df_ensem_Kelly['PL_plc'] > 1).sum() / df_ensem_Kelly.shape[0]))
    print('Final Asset of Win pool bets: {:.2f} ({:.2f}%)'.format(df_ensem_Kelly['NetWealth_win_avg'].iloc[-1] , 100 * (df_ensem_Kelly['NetWealth_win_avg'].iloc[-1] - 1)))
    print('Final Asset of Win pool bets with Rebate: {:.2f} ({:.2f}%)'.format(df_ensem_Kelly['NetWealth_win_avg_w/Rebate'].iloc[-1], 100 * (df_ensem_Kelly['NetWealth_win_avg_w/Rebate'].iloc[-1] - 1)))
    print('Final Asset of Place pool bets: {:.2f} ({:.2f}%)'.format(df_ensem_Kelly['NetWealth_plc_avg'].iloc[-1], 100 * (df_ensem_Kelly['NetWealth_plc_avg'].iloc[-1] - 1)))
    print('Final Asset of Place pool bets with Rebate: {:.2f} ({:.2f}%)'.format(df_ensem_Kelly['NetWealth_plc_avg_w/Rebate'].iloc[-1], 100 * (df_ensem_Kelly['NetWealth_plc_avg_w/Rebate'].iloc[-1] - 1)))
    

    i = i + 1
    
plt.show()




# =============================================================================
# Bet level analysis

fig, axs = plt.subplots(nrows = 1, ncols = 3)

df_anl = df_EV_model
df_anl['year'] = df_anl['Race ID'].str[:4].astype(int)
df_anl['win'] = (df_anl['dividend_win'] > 0) * 1

df_pvt = df_anl.groupby(['win', 'year'])[['oddbr', 'oddfn']].mean()
df_pvt['odds_change%'] = df_pvt['oddfn'] / df_pvt['oddbr'] - 1

df_pvt1 = df_pvt.reset_index()

df_pvt2 = df_anl.groupby(['year'])[['win']].mean().reset_index()


sns.regplot(x = 'year', y = 'odds_change%', data = df_pvt1[df_pvt1['win'] == 1], ax = axs[0])
axs[0].set_xlim(1998,2020)
sns.regplot(x = 'year', y = 'oddfn', data = df_pvt1[df_pvt1['win'] == 1], ax = axs[1])
axs[1].set_xlim(1998,2020)
sns.regplot(x = 'year', y = 'win', data = df_pvt2, ax = axs[2])
axs[2].set_ylim(0,0.3)
axs[2].set_xlim(1998,2020)
axs[2].set_ylabel('Accuracy')





# =============================================================================


sns.distplot(df_summary['odds_prob_br'], bins = 60)
sns.distplot(df_summary['prob_est_cl'], bins = 60)



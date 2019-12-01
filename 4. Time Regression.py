# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 01:03:50 2019

@author: hokai
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import pickle
import math

#Neural network regression packages
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.utils import np_utils 

from keras.layers import Input, Dense, Activation, Dropout, Flatten, Concatenate
from keras.layers import Conv2D, Add, Reshape
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import model_from_json
from keras.models import load_model

#standard sk learn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR

from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.exceptions import DataConversionWarning

import statsmodels.api as sm

#ignore data conversion warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.options.mode.chained_assignment = None


path_import = r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Data\\'
path_export = r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Data\\'

df_raw = pd.read_pickle(path_import + 'RaceData3.pkl')

#============================= Processing =============================#
df = df_raw
col_var = ['Distance', 'Rating', 'Draw', 'Loading', 'oddon', 'oddbr', 'avg_prev_rating', 
           'avg_dist_rating', 'avg_going_rating', 'new_dist',
           'LTD_horse_win_rate', 'LTD_jockey_win_cnt', 'LTD_jockey_win_rate',
           'ema_horse_quantile', 'ema_jockey_quantile', 'prev_margin', 'DOK',
           'class_up', 'class_down', 'class_new', 'class_bleeding', 'weight_diff',
           'prev_t_outpf','log_odd_imply_prob']


##============ Function to get accuracy ===================##

def array_to_df(x):
    return pd.DataFrame(x)

def get_accuracy(y_result, y_pred, df):
    y_pred_temp = array_to_df(y_pred)
    y_pred_temp.columns = ['Time_val_pred']
    y_pred_temp.index = y_result.index
    
    comp = pd.concat([y_result, y_pred_temp, df[['Race ID','Final Position', 'win', 'd_win1', 'd_place1', 'd_place2', 'd_place3']]], axis = 1)
    comp['rank_pred'] = comp.groupby('Race ID')['Time_val_pred'].rank(ascending = True)
    
    accuracy = sum(comp[(comp['win']==1) & (comp['rank_pred']==1)]['win']) / sum(comp[(comp['win']==1)]['win'])
    return comp, accuracy

##============ Functions for models ===================##
#Neural Network Regression
def NNR(x_input, num_layer = 8):
    input_size = x_input.shape[1]
    
    main_input = Input(shape = (input_size,))
    x = main_input
    for i in range(num_layer - 1):
        x = Dense(input_size, activation = 'relu', name = 'Dense{}'.format(i))(x)
#        if i in [1,3,5]:
#            x = Dropout(0.2, name = 'Drop{}'.format(i))(x)
    out_node = Dense(1)(x)
    mm1 = Model(inputs = main_input, outputs = out_node)
    mm1.compile(loss = 'mean_squared_error', optimizer = 'adam')
    
    return mm1


#================ Training with K=fold cross validation =============#
    
sc = StandardScaler()
n_folds = 10
early_terminate = False

load_regr, load_nnr, load_rfr, load_knnr = False, False, False, True
vec_regr, vec_nnr, vec_rfr, vec_knnr = [], [], [], []

kf = KFold(n_splits = n_folds, shuffle = True)

race_id = df['Race ID'].unique()
res_accuracy_train = pd.DataFrame(index = range(n_folds), columns = ['regr', 'nnr', 'rfr', 'knnr'])
res_accuracy_test = pd.DataFrame(index = range(n_folds), columns = ['regr', 'nnr', 'rfr', 'knnr'])

i = 0
#Models looping for corss validation
for train_index, test_index in kf.split(race_id):
    df_train, df_test = df[df['Race ID'].isin(race_id[train_index])], df[df['Race ID'].isin(race_id[test_index])]
    
    x_train, x_test = sc.fit_transform(df_train[col_var]), sc.fit_transform(df_test[col_var])
    y_train, y_test = df_train['Time_val'], df_test['Time_val']
    
    #Linear Regression
    if load_regr:
        print('Fitting Linear regression {}...'.format(i))
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)
        y_regr_pred_train = regr.predict(x_train)
        y_regr_pred_test = regr.predict(x_test)
        vec_regr.append(regr)
    else:
        y_regr_pred_train = np.zeros(x_train.shape[0])
        y_regr_pred_test = np.zeros(x_test.shape[0])
    
    
    #Neural Network regression
    if load_nnr:
        print('Fitting NN regression {}...'.format(i))
        nnr = NNR(x_input = x_train, num_layer = 4)
        hist_nnr = nnr.fit(x_train, y_train, validation_split = 0.2, epochs = 80, verbose = 1)
        y_nnr_pred_train = nnr.predict(x_train)
        y_nnr_pred_test = nnr.predict(x_test)
        vec_nnr.append(nnr)
    else:
        y_nnr_pred_train = np.zeros(x_train.shape[0])
        y_nnr_pred_test = np.zeros(x_test.shape[0])
   
    #Random forest regression
    if load_rfr:
        print('Fitting RF regression {}...'.format(i))
        rfr = RandomForestRegressor(max_depth = 10, n_estimators = 20, random_state = 0)
        rfr.fit(x_train, y_train)
        y_rfr_pred_train = rfr.predict(x_train)
        y_rfr_pred_test = rfr.predict(x_test)
        vec_rfr.append(rfr)
    else:
        y_rfr_pred_train = np.zeros(x_train.shape[0])
        y_rfr_pred_test = np.zeros(x_test.shape[0])
        
    #Knn regression
    if load_knnr:
        print('Fitting KNN regression {}...'.format(i))
        # n is chosen from sqrt(n) where n is number of training samples
        knnr = KNeighborsRegressor(n_neighbors = 400)
        knnr.fit(x_train, y_train)
        y_knnr_pred_train = knnr.predict(x_train)
        y_knnr_pred_test = knnr.predict(x_test)
        vec_knnr.append(knnr)
    else:
        y_knnr_pred_train = np.zeros(x_train.shape[0])
        y_knnr_pred_test = np.zeros(x_test.shape[0])

    #Compute the accuracy matrix
    j = 0
    pred_tuple = tuple(zip([y_regr_pred_train, y_nnr_pred_train, y_rfr_pred_train, y_knnr_pred_train],
                           [y_regr_pred_test, y_nnr_pred_test, y_rfr_pred_test, y_knnr_pred_test]))
    
    for y_pred_train, y_pred_test in pred_tuple:
        res_accuracy_train.iloc[i,j] = get_accuracy(y_train, y_pred_train, df_train)[1]
        res_accuracy_test.iloc[i,j] = get_accuracy(y_test, y_pred_test, df_test)[1]
        j = j + 1

    if early_terminate:
        break
    
    i = i + 1
    
#Print the results
print('Training accuracy:')
print(res_accuracy_train)
print('\nTesting accuracy:')
print(res_accuracy_test)


for i in range(len(vec_nnr)):
    #vec_nnr[i].save(r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Python\saved model\20190503_run\time_regression\nnr\nnr{}.h5'.format(i))
    pickle.dump(vec_regr[i], open(r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Python\saved model\20190503_run\time_regression\regr\regr{}.sav'.format(i), 'wb'))

res_accuracy_test.to_csv(r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Python\saved model\20190503_run\time_regression\res_accuracy_test.csv')
res_accuracy_train.to_csv(r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Python\saved model\20190503_run\time_regression\res_accuracy_train.csv')





# =============================================================================
# regr with statsmodels

x_train1 = sm.add_constant(x_train)
model = sm.OLS(y_train, x_train1).fit()
model.summary()


# =============================================================================
df_nnr_save = pd.read_csv(r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Python\saved model\nnr_saved.csv')

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.set_ylim(bottom = 1, top = 3)
ax1.plot(df_nnr_save['epochs'], df_nnr_save['loss'], color = color)

plt.show()



# =============================================================================
# Constant size wagering

''' ------------------------- Training data -------------------------
'''
results_train = df_train[['Race ID', 'Distance', 'Course', 'Race Class', 'Horse Name',
               'Age', 'Draw', 'Rating', 'Loading', 'Jockey', 'Stable',
               'Final Position', 'Time_val','oddbr', 'oddfn',
               'd_place1', 'd_place2', 'd_place3', 'd_quin']].reset_index(drop = True)

results_train = pd.concat([results_train, pd.DataFrame(y_regr_pred_train, columns = ['y_pred_regr']),
                                         pd.DataFrame(y_nnr_pred_train, columns = ['y_pred_nnr']),
                                         pd.DataFrame(y_rfr_pred_train, columns = ['y_pred_rfr']),
                                         pd.DataFrame(y_knnr_pred_train, columns = ['y_pred_knnr'])], axis = 1)

results_train['plcfn'] = 0
results_train.loc[(results_train['Final Position'] == 1), 'plcfn'] = results_train['d_place1']
results_train.loc[(results_train['Final Position'] == 2), 'plcfn'] = results_train['d_place2']
results_train.loc[(results_train['Final Position'] == 3), 'plcfn'] = results_train['d_place3']


fig, axs = plt.subplots(nrows = 2)

axs[0].set_xlabel('bet count')
axs[0].set_ylabel('Cumulative P&L')

axs[1].set_xlabel('bet count')
axs[1].set_ylabel('Cumulative P&L')

for model in ['regr', 'nnr', 'rfr', 'knnr']:
    field_name = 'rank_pred_{}'.format(model)
    results_train[field_name] = results_train.groupby('Race ID')['y_pred_{}'.format(model)].rank(ascending = True)


results_train_regr = results_train[results_train['rank_pred_regr'] == 1].reset_index(drop = True)
results_train_nnr = results_train[results_train['rank_pred_nnr'] == 1].reset_index(drop = True)
results_train_rfr = results_train[results_train['rank_pred_rfr'] == 1].reset_index(drop = True)
results_train_knnr = results_train[results_train['rank_pred_knnr'] == 1].reset_index(drop = True)

i = 0
for res, model in [(results_train_regr, 'regr'), (results_train_nnr, 'nnr'), (results_train_rfr, 'rfr'), (results_train_knnr, 'knnr')]:
    
    print('========= model {} training data ========='.format(model))
    res['cost'] = -1
    res['dividend_win'] = (res['Final Position'] == 1) * res['oddfn']
    res['dividend_plc'] = (res['Final Position'] <= 3) * res['plcfn'] / 10
    res['PL_win'] = res['dividend_win'] + res['cost']
    res['PL_plc'] = res['dividend_plc'] + res['cost']
    res['NetWealth_win_{}'.format(model)] = res['PL_win'].cumsum()
    res['NetWealth_plc_{}'.format(model)] = res['PL_plc'].cumsum()

    print('Bet Count:{}'.format(res.shape[0]))
    print('Correct win Bet: {} ({:.2f}%)'.format((res['dividend_win'] > 0).sum(), 100 * (res['dividend_win'] > 0).sum() / res.shape[0]))
    print('Correct plc Bet: {} ({:.2f}%)'.format((res['dividend_plc'] > 0).sum(), 100 * (res['dividend_plc'] > 0).sum() / res.shape[0]))
    print('Win final net wealth: {:.2f} ({:.2f}%)'.format(res['NetWealth_win_{}'.format(model)].iloc[-1], 100 * res['NetWealth_win_{}'.format(model)].iloc[-1] / res.shape[0] ))
    print('Plc final net wealth: {:.2f} ({:.2f}%)'.format(res['NetWealth_plc_{}'.format(model)].iloc[-1], 100 * res['NetWealth_plc_{}'.format(model)].iloc[-1] / res.shape[0] ))

    res['NetWealth_win_{}'.format(model)].plot(ax = axs[0], legend = True)
    res['NetWealth_plc_{}'.format(model)].plot(ax = axs[1], legend = True)
    
    i = i + 1
    

plt.show()





'''------------------------- Testing data -------------------------
'''
results_test = df_test[['Race ID', 'Distance', 'Course', 'Race Class', 'Horse Name',
               'Age', 'Draw', 'Rating', 'Loading', 'Jockey', 'Stable',
               'Final Position', 'Time_val','oddbr', 'oddfn',
               'd_place1', 'd_place2', 'd_place3', 'd_quin']].reset_index(drop = True)

results_test = pd.concat([results_test, pd.DataFrame(y_regr_pred_test, columns = ['y_pred_regr']),
                                         pd.DataFrame(y_nnr_pred_test, columns = ['y_pred_nnr']),
                                         pd.DataFrame(y_rfr_pred_test, columns = ['y_pred_rfr']),
                                         pd.DataFrame(y_knnr_pred_test, columns = ['y_pred_knnr'])], axis = 1)

results_test['plcfn'] = 0
results_test.loc[(results_test['Final Position'] == 1), 'plcfn'] = results_test['d_place1']
results_test.loc[(results_test['Final Position'] == 2), 'plcfn'] = results_test['d_place2']
results_test.loc[(results_test['Final Position'] == 3), 'plcfn'] = results_test['d_place3']


fig, axs = plt.subplots(nrows = 2)

axs[0].set_xlabel('bet count')
axs[0].set_ylabel('Cumulative P&L')

axs[1].set_xlabel('bet count')
axs[1].set_ylabel('Cumulative P&L')

for model in ['regr', 'nnr', 'rfr', 'knnr']:
    field_name = 'rank_pred_{}'.format(model)
    results_test[field_name] = results_test.groupby('Race ID')['y_pred_{}'.format(model)].rank(ascending = True)


results_test_regr = results_test[results_test['rank_pred_regr'] == 1].reset_index(drop = True)
results_test_nnr = results_test[results_test['rank_pred_nnr'] == 1].reset_index(drop = True)
results_test_rfr = results_test[results_test['rank_pred_rfr'] == 1].reset_index(drop = True)
results_test_knnr = results_test[results_test['rank_pred_knnr'] == 1].reset_index(drop = True)

i = 0
for res, model in [(results_test_regr, 'regr'), (results_test_nnr, 'nnr'), (results_test_rfr, 'rfr'), (results_test_knnr, 'knnr')]:
    
    print('========= model {} testing data ========='.format(model))
    res['cost'] = -1
    res['dividend_win'] = (res['Final Position'] == 1) * res['oddfn']
    res['dividend_plc'] = (res['Final Position'] <= 3) * res['plcfn'] / 10
    res['PL_win'] = res['dividend_win'] + res['cost']
    res['PL_plc'] = res['dividend_plc'] + res['cost']
    res['NetWealth_win_{}'.format(model)] = res['PL_win'].cumsum()
    res['NetWealth_plc_{}'.format(model)] = res['PL_plc'].cumsum()
    
    print('Bet Count:{}'.format(res.shape[0]))
    print('Correct win Bet: {} ({:.2f}%)'.format((res['dividend_win'] > 0).sum(), 100 * (res['dividend_win'] > 0).sum() / res.shape[0]))
    print('Correct plc Bet: {} ({:.2f}%)'.format((res['dividend_plc'] > 0).sum(), 100 * (res['dividend_plc'] > 0).sum() / res.shape[0]))
    print('Win final net wealth: {:.2f} ({:.2f}%)'.format(res['NetWealth_win_{}'.format(model)].iloc[-1], 100 * res['NetWealth_win_{}'.format(model)].iloc[-1] / res.shape[0] ))
    print('Plc final net wealth: {:.2f} ({:.2f}%)'.format(res['NetWealth_plc_{}'.format(model)].iloc[-1], 100 * res['NetWealth_plc_{}'.format(model)].iloc[-1] / res.shape[0] ))

    res['NetWealth_win_{}'.format(model)].plot(ax = axs[0], legend = True)
    res['NetWealth_plc_{}'.format(model)].plot(ax = axs[1], legend = True)
    
    i = i + 1
    

plt.show()





# =============================================================================
# Ensembled model

for tmp in [results_train, results_test]:
    tmp['score_avg'] = 0.25 * (tmp['rank_pred_regr'] + tmp['rank_pred_nnr']+
                                   tmp['rank_pred_rfr'] + tmp['rank_pred_knnr'])
    tmp['rank_pred_avg'] = tmp.groupby('Race ID')['score_avg'].rank(ascending = True)


results_train_avg = results_train[results_train['rank_pred_avg'] == 1].reset_index(drop = True)
results_test_avg = results_test[results_test['rank_pred_avg'] == 1].reset_index(drop = True)

i = 0
rebate = True
fig, axs = plt.subplots(ncols = 2)

for res, model in [(results_train_avg, 'avg'), (results_test_avg, 'avg')]:
    
    print('========= model {} ========='.format(model))
    res['cost'] = -1
    res['dividend_win'] = (res['Final Position'] == 1) * res['oddfn']
    res['dividend_plc'] = (res['Final Position'] <= 3) * res['plcfn'] / 10
    
    if rebate:
        res['rebate_win'] = (res['dividend_win'] == 0)*0.1
        res['rebate_plc'] = (res['dividend_plc'] == 0)*0.1
    else:
        res['rebate_win'] = 0
        res['rebate_plc'] = 0
        
    res['PL_win'] = res['dividend_win'] + res['cost'] + res['rebate_win']
    res['PL_plc'] = res['dividend_plc'] + res['cost'] + res['rebate_plc']
    res['NetWealth_win_{}'.format(model)] = res['PL_win'].cumsum()
    res['NetWealth_plc_{}'.format(model)] = res['PL_plc'].cumsum()
    
    print('Bet Count:{}'.format(res.shape[0]))
    print('Correct win Bet: {} ({:.2f}%)'.format((res['dividend_win'] > 0).sum(), 100 * (res['dividend_win'] > 0).sum() / res.shape[0]))
    print('Correct plc Bet: {} ({:.2f}%)'.format((res['dividend_plc'] > 0).sum(), 100 * (res['dividend_plc'] > 0).sum() / res.shape[0]))
    print('Win final net wealth: {:.2f} ({:.2f}%)'.format(res['NetWealth_win_{}'.format(model)].iloc[-1], 100 * res['NetWealth_win_{}'.format(model)].iloc[-1] / res.shape[0] ))
    print('Plc final net wealth: {:.2f} ({:.2f}%)'.format(res['NetWealth_plc_{}'.format(model)].iloc[-1], 100 * res['NetWealth_plc_{}'.format(model)].iloc[-1] / res.shape[0] ))
    
    axs[i].set_xlabel('bet count')
    axs[i].set_ylabel('Cumulative P&L')
    
    res['NetWealth_win_{}'.format(model)].plot(ax = axs[i], legend = True)
    res['NetWealth_plc_{}'.format(model)].plot(ax = axs[i], legend = True)
    
    i = i + 1





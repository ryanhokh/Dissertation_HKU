# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:02:56 2019

@author: hokai
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import random
from random import randint

#Neural network regression packages
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Concatenate, MaxPooling2D, Lambda
from keras.layers import Conv2D, Add, Reshape, Conv1D, TimeDistributed, Permute
from keras.optimizers import RMSprop, Adam, SGD 

import keras.backend as K

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import DataConversionWarning

from keras.utils import plot_model
 
#============= minmax scaler warning ignore ==============#
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#============= import data and split to training/testing ==============#
path_import = r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Data\\'
path_export = r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Data\\'

df_raw = pd.read_pickle(path_import + 'RaceData3.pkl')



df = df_raw
#============= Functions for models ======================#
def logFunc(x):
    return K.log(x)

def clogit(n_feature, n_choice, logits_activation = 'softmax'):
    main_input = Input((n_feature, n_choice, 1), name = 'input_layer')
    utility = Conv2D(filters = 1, kernel_size = [n_feature, 1], strides = (1,1), 
                     padding = 'valid', trainable = True, use_bias = True, 
                     name = 'utility')(main_input)
    util_flat = Flatten(name = 'utility_flatten')(utility)
    logits = Activation(logits_activation, name = 'activation')(util_flat)
    cl = Model(inputs = main_input, outputs = logits, name = 'choice_layer')
    return cl


def clogit2step(n_feature, n_choice, n_extra, logits_activation = 'softmax'):
    main_input = Input((n_feature, n_choice, 1), name = 'input_layer')
    extra_input = Input((n_extra, n_choice, 1), name = 'extra_input_layer')
    
    utility_main = Conv2D(filters = 1, kernel_size = [n_feature, 1], strides = (1,1), 
                     padding = 'valid', trainable = True, use_bias = True, 
                     name = 'utility_main')(main_input)
    
    utility_extra = Conv2D(filters = 1, kernel_size = [n_extra, 1], strides = (1,1), 
                     padding = 'valid', trainable = True, use_bias = True, 
                     name = 'utility_extra')(extra_input)
    
    utility_main_flatten = Reshape([n_choice], name = 'flatten_main')(utility_main)
    utility_extra_flatten = Reshape([1, n_choice], name = 'flatten_extra')(utility_extra)
    
    logits_main = Activation(logits_activation, name = 'logits_main_input')(utility_main_flatten)
    logits_main_log = Lambda(logFunc)(logits_main)
    logits_main_log = Reshape([1, n_choice], name = 'reshape_logits')(logits_main_log)
    
    utility_concat = Concatenate(axis = -2)([logits_main_log, utility_extra_flatten])
    utility_concat = Reshape([2,n_choice,1])(utility_concat)
    utility_final = Conv2D(filters = 1, kernel_size=[2,1], strides = (1,1),
                           padding = 'valid', trainable = True, use_bias = False,
                           name = 'step2_conv')(utility_concat)
    utility_final = Reshape([n_choice], name = 'util_flat_final')(utility_final)
    logits_final = Activation(logits_activation, name = 'logits_final')(utility_final)
    cl2 = Model(inputs = [main_input, extra_input], outputs = logits_final)
    return cl2

def NN_MNL(n_feature, n_choice, n_layer, logits_activation = 'softmax'):
    main_input = Input((n_feature, n_choice, 1), name = 'input_layer')
    x = Permute((3,2,1))(main_input)
    for i in range(n_layer):
        x = TimeDistributed(Dense(n_feature), name = 'utility_{}'.format(i))(x)
    x = TimeDistributed(Dense(1), name = 'utility_final')(x)
    x = Permute((3,2,1))(x)
    utility = Flatten()(x)
    logits = Activation(logits_activation, name = 'choice')(utility)
    
    nn_mnl = Model(inputs = main_input, outputs = logits)
    return nn_mnl


def L_MNL(n_choice, n_feature, n_extra, n_layer, logits_activation = 'softmax'):
    main_input = Input((n_feature, n_choice, 1), name = 'input_layer')
    extra_input = Input((n_extra, n_choice, 1), name = 'extra_input_layer')
    
    #Linear Kernel
    utility_main = Conv2D(filters = 1, kernel_size = [n_feature, 1], strides = (1,1), 
                     padding = 'valid', trainable = True, use_bias = True, 
                     name = 'utility_main')(main_input)
    utility_main = Reshape([n_choice], name = 'L_util_flatten')(utility_main)
    
    ##NN Kernel
    x = Permute((3,2,1))(extra_input)
    for i in range(n_layer):
        x = TimeDistributed(Dense(n_extra), name = 'NN_utility_{}'.format(i))(x)
    x = TimeDistributed(Dense(1), name = 'NN_utility_final')(x)
    x = Permute((3,2,1))(x)
    utility_extra = Flatten()(x)
    
    #Combining Linear kernal and NN kernal
    final_utilities = Add(name="New_Utility_functions")([utility_main, utility_extra])
    logits = Activation(logits_activation, name='Choice')(final_utilities)   
    
    l_mnl = Model(inputs = [main_input, extra_input], outputs = logits)
    return l_mnl


def denseNN(beta_num, choices_num, networkSize = 16, logits_activation = 'softmax'):
	""" Dense Neural Network (writen with CNN for coding convenience. Connections are Equivalent Here)
	 	- Kernel is size of Input. Numbers of Filters are the size of the neurons ( == DNN )
	"""
	main_input= Input((beta_num, choices_num,1), name='Features')
	dense = Conv2D(filters = networkSize, kernel_size=[beta_num, choices_num], activation = 'relu', padding = 'valid', name = 'Dense_NN_per_frame')(main_input)
	#Dropout successfully prevents overfit. If removed, it is better to run model.fit on full data including a callback.
	dropped = Dropout(0.2, name = 'Regularizer')(dense)
	new_feature = Dense(units = choices_num, name="Output_new_feature")(dropped)
	new_featureR = Reshape([choices_num], name='Remove_Dim')(new_feature)

	logits = Activation(logits_activation, name='Choice')(new_featureR)

	model = Model(inputs = main_input, outputs=logits)
	return model



#=============  utilities ==============#
def Convert3D_X(X, size = 12):
    return X.reshape(int(X.shape[0]/size), size, X.shape[1])

def Convert3D_y(y, size = 12):
    return y.values.reshape(int(y.shape[0]/size), size)

def cal_accuracy(model, X_data, y_data, return_results = False):
    y_pred = model.predict(X_data)
    results_pred = pd.DataFrame(y_pred).idxmax(axis = 1)
    results_true = pd.DataFrame(y_data).idxmax(axis = 1)
    results = pd.DataFrame(zip(results_pred, results_true))
    results = results.rename(columns = {0: 'results_pred', 1: 'results_true'})
    accuracy = results[results['results_pred'] == results['results_true']].shape[0] / y_data.shape[0]
    
    return results, accuracy



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



def DCM_results_multi(models, X_data, df_data):
    #Prepare basic fields
    results = df_data[['Race ID', 'Distance', 'Course', 'Race Class', 'Horse Name',
                       'Age', 'Draw', 'Rating', 'Loading', 'Jockey', 'Stable',
                       'Final Position', 'Time_val','oddbr', 'oddfn',
                       'd_place1', 'd_place2', 'd_place3', 'd_quin']]
    results = results.reset_index(drop = True)
    #Derive Plc odds column
    results['plcfn'] = 0
    results.loc[(results['Final Position'] == 1), 'plcfn'] = results['d_place1']
    results.loc[(results['Final Position'] == 2), 'plcfn'] = results['d_place2']
    results.loc[(results['Final Position'] == 3), 'plcfn'] = results['d_place3']
    
    #Prepare model predictions
    i = 0
    for m in models:
        y_pred = m.predict(X_data)
        y_pred = pd.DataFrame(y_pred.reshape(y_pred.size))
        y_pred = y_pred.rename(columns = {0: 'prob_est{}'.format(i)})
        results = pd.concat([results, y_pred], axis = 1)
        results['rank_pred{}'.format(i)] = results.groupby('Race ID')['prob_est{}'.format(i)].rank(ascending = False)
        i = i + 1

    return results



def plot_metric(acc, loss):
    t = np.arange(len(acc))
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')
    ax1.plot(t, acc, color = color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('loss')
    ax2.plot(t, loss, color = color)
    
    fig.tight_layout()
    plt.show()
    
    
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



#============= Training the model with Cross validation =================#
n_folds = 10
kf = KFold(n_splits = n_folds, shuffle = True)
scaling = True
shuffle_data = True
early_terminate = True


# If shuffle then shuffle, else we sort the data by Draw (to pretend the start of race)
if shuffle_data:
    df = df.groupby('Race ID').apply(lambda x: x.sample(frac=1)).reset_index(drop = True)
else: 
    df = df.sort_values(['Race ID', 'Draw'], ascending = True).reset_index(drop = True)

#============= Get the most frequeny running count ============#

df_temp = df[['Race ID','Rating']].groupby('Race ID').count().reset_index()
df_temp_freq = pd.crosstab(index = df_temp['Rating'], columns = 'count').reset_index()

race_size = 14    #Race size is either 12 or 14
race_id_input = df_temp[df_temp['Rating'] == race_size]['Race ID'].unique()


load_cl, load_dnn, load_cl2, load_nnmnl, load_lmnl = True, False, True, True, True
#load_cl, load_dnn, load_cl2, load_nnmnl, load_lmnl = True, False, True, True, True

vec_cl, vec_dnn, vec_cl2, vec_nnmnl, vec_lmnl = [], [], [], [], []

res_accuract_train = pd.DataFrame(index = range(n_folds), columns = ['cl', 'dnn', 'cl2', 'nn_mnl', 'lmnl'])
res_accuract_test = pd.DataFrame(index = range(n_folds), columns = ['cl', 'dnn', 'cl2', 'nn_mnl', 'lmnl'])

i = 0



for train_index, test_index in kf.split(race_id_input):
    
    print('Loading fold {}...'.format(i))
    
    vec_model = []
    # Data Preparation
    df_train = df[df['Race ID'].isin(race_id_input[train_index])]
    df_test = df[df['Race ID'].isin(race_id_input[test_index])]
    
    sc = StandardScaler()
    
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
    
    # Loading models - Condtional logit
    if load_cl:
        print('Loading conditional logit...')
        cl = clogit(n_feature = x_train.shape[1], n_choice = x_train.shape[2])
        cl.compile(loss = 'categorical_crossentropy', optimizer = 'adam' ,metrics = ['accuracy'])
        hist_cl = cl.fit(x_train, y_train, epochs = 40)
        vec_cl.append(cl)
        
        res_accuract_train.iloc[i,0] = cal_accuracy(cl, x_train, y_train)[1]
        res_accuract_test.iloc[i,0] = cal_accuracy(cl, x_test, y_test)[1]

    
    
    # Loading models - dNN
#    if load_dnn:
#        print('Loading dense NN...')
#        dnn = denseNN(x_train.shape[1], x_train.shape[2], networkSize = 12)
#        dnn.compile(loss = 'categorical_crossentropy', optimizer = 'adam' ,metrics = ['accuracy'])
#        dnn.fit(x_train, y_train, epochs = 80)
#        vec_dnn.append(dnn)
#        
#        res_accuract_train.iloc[i,1] = cal_accuracy(dnn, x_train, y_train)[1]
#        res_accuract_test.iloc[i,1] = cal_accuracy(dnn, x_test, y_test)[1]


    # Loading models - 2step Conditional logit model    
    if load_cl2:
        print('Loading conditional logit 2-step...')
        cl2 = clogit2step(n_feature = x_train_func.shape[1], n_choice = x_train_func.shape[2], n_extra = x_train_tech.shape[1])
        cl2.compile(loss = 'categorical_crossentropy', optimizer = 'adam' ,metrics = ['accuracy'])
        hist_cl2 = cl2.fit([x_train_func, x_train_tech], y_train, epochs = 70)
        vec_cl2.append(cl2)
        
        res_accuract_train.iloc[i,2] = cal_accuracy(cl2, [x_train_func,x_train_tech], y_train)[1]
        res_accuract_test.iloc[i,2] = cal_accuracy(cl2, [x_test_func, x_test_tech], y_test)[1]
        
    # Loading models - neural network multinomial logit
    if load_nnmnl:
        print('Loading neural network MNL...')
        nn_mnl = NN_MNL(x_train.shape[1], x_train.shape[2], n_layer = 8)
        nn_mnl.compile(loss = 'categorical_crossentropy', optimizer = 'adam' ,metrics = ['accuracy'])
        hist_nnmnl = nn_mnl.fit(x_train, y_train, epochs = 100)
        vec_nnmnl.append(nn_mnl)
        
        res_accuract_train.iloc[i,3] = cal_accuracy(nn_mnl, x_train, y_train)[1]
        res_accuract_test.iloc[i,3] = cal_accuracy(nn_mnl, x_test, y_test)[1]
        
    if load_lmnl:
        print('Loading L-MNL model')
        lmnl = L_MNL(n_choice = x_train_l.shape[2], n_feature = x_train_l.shape[1], n_extra = x_train_nn.shape[1], n_layer = 8)
        lmnl.compile(loss = 'categorical_crossentropy', optimizer = 'adam' ,metrics = ['accuracy'])
        hist_lmnl = lmnl.fit([x_train_l, x_train_nn], y_train, epochs = 100)
        vec_lmnl.append(lmnl)
        
        res_accuract_train.iloc[i,4] = cal_accuracy(lmnl, [x_train_l, x_train_nn], y_train)[1]
        res_accuract_test.iloc[i,4] = cal_accuracy(lmnl, [x_test_l, x_test_nn], y_test)[1]
        
        
        
    if early_terminate:
        break
    i = i + 1
    
print(res_accuract_train)
print(res_accuract_test)


# =============================================================================
#  Training progress
print(hist_cl.history.keys())
plot_metric(hist_cl.history['acc'], hist_cl.history['loss'])

print(hist_cl2.history.keys())
plot_metric(hist_cl2.history['acc'], hist_cl2.history['loss'])
plot_model(cl2, to_file = r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Python\cl2.png')

print(hist_nnmnl.history.keys())
plot_metric(hist_nnmnl.history['acc'], hist_nnmnl.history['loss'])

nn_mnl2
nn_mnl8


print(hist_lmnl.history.keys())
plot_metric(hist_lmnl.history['acc'], hist_lmnl.history['loss'])

# =============================================================================
#  L-MNL intermediate

lmnl_Lutil = Model(inputs = lmnl.input, outputs= lmnl.get_layer('L_util_flatten').output)
lmnl_Lutil_output = lmnl_inter.predict([x_train_l, x_train_nn])
lmnl_Lutil_output = pd.DataFrame(lmnl_Lutil_output)
v_L = lmnl_Lutil_output.var(axis = 1)

lmnl_NNutil = Model(inputs = lmnl.input, outputs= lmnl.get_layer('flatten_10').output)
lmnl_NNutil_output = lmnl_NNutil.predict([x_train_l, x_train_nn])
lmnl_NNutil_output = pd.DataFrame(lmnl_NNutil_output)
v_NN = lmnl_NNutil_output.var(axis = 1)
sns.scatterplot(x = v_L, y = v_NN)
plt.xlabel('Linear Kernel Variances')
plt.ylabel('NN Kernel Variances')

lmnl_Addutil = Model(inputs = lmnl.input, outputs= lmnl.get_layer('New_Utility_functions').output)
lmnl_Addutil_output = lmnl_Addutil.predict([x_train_l, x_train_nn])
lmnl_Addutil_output = pd.DataFrame(lmnl_Addutil_output)





# =============================================================================
#  Save and export data & models

path_export_model = r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Python\data_paper_DCM\\'

cl.save(path_export_model + 'cl.h5')
cl2.save(path_export_model + 'cl2.h5')
nn_mnl.save(path_export_model + 'nn_mnl.h5')
lmnl.save(path_export_model + 'lmnl.h5')

df_train.to_pickle(path_export_model + 'df_train.pkl')
df_test.to_pickle(path_export_model + 'df_test.pkl')


# =============================================================================






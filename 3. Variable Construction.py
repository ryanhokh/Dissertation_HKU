# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 20:17:38 2019

@author: hokai
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#======================= Data loading =========================#
path_import = r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Data\\'
path_export = r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Data\\'
df_raw = pd.read_pickle(path_import + 'RaceData1.pkl')
df_stdtime = pd.read_csv(path_import + 'TrackStandardTime.csv')


# ====================== Processing and filtering ====================#
df = df_raw

#Filter for data after 1998
df = df[df['Season_yr'] > 1998]
df = df[df['bodyweight'] > 10]

df = df.reset_index(drop = True)

"""01 var type update and cleaning
"""
#Change margin to float
df['margin'] = df['margin'].apply(float)
#Drop synthetic columns
df = df.drop(columns = ['m', 's', 's1', 's2'])

#Merge with standard time
df['Track Type'] = df['Track'].map({'AWT': 'AWT'})
df['Track Type'] = df['Track Type'].fillna('Turf')
#To preserve the order when merging
df = df.merge(df.merge(df_stdtime, how = 'inner', on = ['Course', 'Track Type', 'Distance', 'Race Class'], sort = False))
df['time_outperformance'] = df['Time_val'] - df['totalstandardtime']


"""02 Significant Variable construction
"""
"""From identifying winners of competitive events: A SVM-based classification model for horserace prediction
"""
#First formula the old variables
#Avg rating of 4 prev races - performance
df['avg_prev_rating'] = 0
n = 4
for i in range(n):
    df['avg_prev_rating'] = df['avg_prev_rating'] + (1/n) * df.groupby('Horse Name')['Rating'].shift(i + 1)

df['avg_prev_rating'] = df['avg_prev_rating'].fillna(0)


#Avg rating of prev races at the same distance - distance preference
df['avg_dist_rating'] = df.groupby(['Horse Name', 'Distance'])['Rating'].apply(lambda x: x.expanding().mean())


#Avg rating of prev races at the same going - going preference
df['avg_going_rating'] = df.groupby(['Horse Name', 'Going'])['Rating'].apply(lambda x: x.expanding().mean())


#New distance - changing distance could have discount the performance (if 3 or 4 of prev race is 80% less distance then 1 else 0)
df['new_dist'] = 0
n = 4
for i in range(n):
    df['new_dist'] = df['new_dist'] + (df.groupby('Horse Name')['Distance'].shift(i + 1) < df['Distance'] * 0.8)

df['new_dist'] = (df['new_dist'] >= 3) * 1


#Horse LTD win rate
df['LTD_horse_win_rate'] = df.groupby('Horse Name')['win'].apply(lambda x: x.expanding().mean())
df['LTD_horse_win_rate'] = df.groupby('Horse Name')['LTD_horse_win_rate'].shift(1)


#Jockey LTD win count
df['LTD_jockey_win_cnt'] = df.groupby('Jockey')['win'].apply(lambda x: x.expanding().sum())
df['LTD_jockey_win_cnt'] = df.groupby('Jockey')['LTD_jockey_win_cnt'].shift(1)


#Jockey win rate
df['LTD_jockey_win_rate'] = df.groupby('Jockey')['win'].apply(lambda x: x.expanding().mean())
df['LTD_jockey_win_rate'] = df.groupby('Jockey')['LTD_jockey_win_rate'].shift(1)

"""Generated idea from identifying...
"""
#quantile is the bigger the better
#Exponential moving average of horse quantile (35% representation of current quantile)
df['ema_horse_quantile'] = df.groupby('Horse Name')['quantile'].apply(lambda x: x.ewm(span = 5).mean())
df['ema_horse_quantile'] = df.groupby('Horse Name')['ema_horse_quantile'].shift(1)


#EMA of jockey quantile
df['ema_jockey_quantile'] = df.groupby('Jockey')['quantile'].apply(lambda x: x.ewm(span = 20).mean())
df['ema_jockey_quantile'] = df.groupby('Jockey')['ema_jockey_quantile'].shift(1)



"""From Computer based horse race handicapping and wagering
"""
#Historical margin 
df['prev_margin'] = df.groupby('Horse Name')['margin'].shift(1)

#DOK - margin & quantile ((winning position > 50th quantile) OR (within 6.25lengths of winner) OR (within 1/16 of a mile of today's distance ##unable to implement))
df['DOK'] = ((df['margin'] < 6.25) | (df['quantile'] < 0.5)) * 1
df['DOK'] = df.groupby('Horse Name')['DOK'].shift(1)


""" Other popular estimates and metrics
"""
#Up/down class as categorical variable
df['class_up'] = (df['Up/Down'] == 'Up')*1
df['class_down'] = (df['Up/Down'] == 'Down')*1
df['class_new'] = (df['Up/Down'] == 'New')*1


#Bleeding as categorical variable
df['class_bleeding'] = df['Bleeding']*1

#Weight difference
df['weight_diff'] = df['bodyweight'] - df.groupby('Horse Name')['bodyweight'].shift(1)
df['weight_diff'] = df['weight_diff'].fillna(0)


#Time outperformance in previous race
df['time_outperformance'] = [5 if abs(x) > 10 else x for x in df['time_outperformance']]
df['prev_t_outpf'] = df.groupby('Horse Name')['time_outperformance'].shift(1)


#log of Odds implied winning probability
df['oddfn-1'] = 1/df['oddfn']
df['oddfn_gp_total'] = df.groupby('Race ID')['oddfn-1'].transform('sum')
df['log_odd_imply_prob'] = np.log(df['oddfn-1']/df['oddfn_gp_total'])
df = df.drop(columns = ['oddfn-1', 'oddfn_gp_total'])



#======================= Filling n/a ======================#
#filling win rate to 0s to N/a because the horse had never won
col_var0 = ['LTD_horse_win_rate', 'LTD_jockey_win_cnt', 'LTD_jockey_win_rate', 'ema_horse_quantile', 'ema_jockey_quantile']
df[col_var0] = df[col_var0].fillna(0)

#margin follows a right fat tail distribution, putting n/a to mean should give no bias to the horse
df['prev_margin'] = df['prev_margin'].fillna(df['prev_margin'].mean())

#DOK should fill to 0 because majority of horses (160k/180k) could make 1. filing DOK to 1 also because the horse is expected to place at a proper race
df['DOK'] = df['DOK'].fillna(1)

#time outperformance also follows a right fat tail distribution, putting n/a to mean should give no bias to the horse
df['prev_t_outpf'] = df['prev_t_outpf'].fillna(df['prev_t_outpf'].mean())


#======================= Export ======================#
df.to_csv(path_export + 'RaceData3.csv')
df.to_pickle(path_export + 'RaceData3.pkl')
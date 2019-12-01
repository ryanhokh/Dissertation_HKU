# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 01:18:18 2019

@author: hokai
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

path_import = r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Data\\'
path_export = r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Visualization\\'
df = pd.read_pickle(path_import + 'RaceData1.pkl')
df_stdtime = pd.read_csv(path_import + 'TrackStandardTime.csv')

#Filter for data after 1998
df = df[df['Season_yr'] > 1998]
df = df[df['bodyweight'] > 10]

df = df.reset_index(drop = True)

"""01 Average speed across class
"""
ax = sns.boxplot(x = 'Race Class', y = 'speed', data = df, order = ['CUP', '1', '2', '3', '4', '5', 'GRIFFIN'])
ax.set(ylim = (13,19))
ax.get_figure().savefig('01_speed_class.png')

df_speed1_summary = df[['Race Class','speed']].groupby(['Race Class'])['speed'].describe().reset_index()
print(df_speed1_summary)


"""02 Speed by class and by distance
"""
df_speed2 = pd.pivot_table(df, values = 'speed', index = ['Distance'], columns = ['Race Class'], aggfunc = np.mean)
df_speed2.columns = ['Class ' + x for x in df_speed2.columns]
ax = sns.lineplot(data = df_speed2[df_speed2.columns[0:6]])
ax.get_figure().savefig('02_speed_class_distance.png')


"""03 Speed SD by class and by distance
"""
df_speed3 = pd.pivot_table(df, values = 'speed', index = ['Distance'], columns = ['Race Class'], aggfunc = np.std)
df_speed3.columns = ['Class ' + x for x in df_speed3.columns]
ax = sns.lineplot(data = df_speed3[df_speed3.columns[0:6]])
ax.get_figure().savefig('03_speed_SD_class_distance.png')


"""04 Speed by Track status and distance (sample by race class 3)
"""
df_speed4 = pd.pivot_table(df[df['Race Class'] == '3'], values = 'speed', index = ['Distance'], columns = ['Going'], aggfunc = np.mean)
ax = sns.lineplot(data = df_speed4[['FA', 'FG', 'G', 'GA', 'GY', 'Y']])
ax.get_figure().savefig('04_speed_going_distance.png')


"""05 Speed by bodyweight and distance
"""
list_dist = [1200, 1600, 1800]
list_class = ['1', '3', '5']

df_speed5 = df[df['Distance'].isin(list_dist) & df['Race Class'].isin(list_class)]
ax = sns.lmplot(x = 'bodyweight', y = 'speed', col = 'Distance', row = 'Race Class', data = df_speed5)
plt.figure()
ax.set(xscale = 'log', xlim = (850, 1400), ylim = (13,19))
ax.get_figure().savefig('05_speed_bodyweight_distance.png')

df_speed5_correl = pd.DataFrame(index = list_dist, columns=list_class)

for dist in list_dist:
    for clas in list_class:
        tmp = df[(df['Distance'] == dist) & (df['Race Class'] == clas)][['bodyweight', 'speed']].corr()
        df_speed5_correl.loc[dist, clas] = tmp.iloc[1,0]

print(df_speed5_correl)


"""06 Win Rate dependent to trainer and jockey
"""
#function used for jockey, where win% is about (win of jockey)/(jockey attendence)
def fn_rate(df_race, cnt_threshold, start, sort_by = 'win%', asc = False ,top_cnt = 10, id_type = 'Jockey'):
    tmp = df_race[df_race['Season_yr'] >= start]
    tmp = tmp.groupby(id_type).agg({'win': ['count', 'sum', 'mean'],
                                    'plc': ['sum', 'mean']})
    tmp = tmp[tmp['win']['count'] > cnt_threshold]
    tmp.columns = ['race count', 'win count', 'win%', 'place count', 'plc%']
    tmp = tmp.sort_values(by = sort_by, ascending = asc)
    return tmp.head(top_cnt)


#Best win rate jockey in 10yrs
print(fn_rate(df, cnt_threshold=50, start=2009))

#Best win rate jockey in 5yrs
print(fn_rate(df, cnt_threshold=50, start=2014))

#Best win rate jockey in current year
print(fn_rate(df, cnt_threshold=50, start=2018))

#Best win rate trainer in 5 yrs
print(fn_rate(df, cnt_threshold=50, start=2014, id_type='Stable'))

#Best win rate trainer in current year
print(fn_rate(df, cnt_threshold=50, start=2018, id_type='Stable'))



"""07 Gate number win rate and dependence to track & course
"""
df_gate = fn_rate(df, cnt_threshold=0, start = 0, id_type='Draw', sort_by='Draw', asc = True, top_cnt=50)
df_gate = df_gate.reset_index()
df_gate = df_gate[['Draw', 'win%', 'plc%']]
df_gate = pd.melt(df_gate, id_vars='Draw', var_name='type', value_name='rates')
plt.figure()
ax = sns.barplot(x = 'Draw', y = 'rates', hue = 'type', data = df_gate)
ax.get_figure().savefig('07_winplc_rate_draw.png')


df_gate = df_gate = fn_rate(df, cnt_threshold=0, start = 0, id_type=['Distance','Draw'], sort_by=['Distance', 'Draw'], asc = True, top_cnt=5000)
df_gate = df_gate.reset_index()
df_gate = df_gate[df_gate['Distance'].isin([1200, 1800, 2200])]
plt.figure()
ax = sns.barplot(x = 'Draw', y = 'win%', hue = 'Distance', data = df_gate)
ax.get_figure().savefig('07_win_rate_draw_distance.png')

#Win rate on all possibilities
df_gate = fn_rate(df, cnt_threshold=20, start = 0, 
                  id_type=['Course', 'Track', 'Distance','Draw'], 
                  sort_by = ['Course', 'Track', 'Distance','Draw'], 
                  top_cnt=10000, asc = True)
df_gate = df_gate.reset_index()
print(df_gate)


"""08 Age win rate dependence
"""

df_age = fn_rate(df, cnt_threshold=0, start = 0, id_type='Age', sort_by='Age', asc = True)
df_age = df_age.reset_index()
df_age = df_age.iloc[1:]
print(df_age)
ax = sns.barplot(x = 'Age', y = 'win count', data = df_age)
ax2 = plt.twinx()
ax2 = sns.lineplot(x = df_age.index - 1, y = 'win%', data = df_age)
ax2.get_figure().savefig('08_win_rate_age.png')


"""09 Class changing dependence
"""
df_class = fn_rate(df, cnt_threshold=0, start = 0, id_type='Up/Down', asc = True)
df_class = df_class.reset_index()
print(df_class)
ax = sns.barplot(x = 'Up/Down', y = 'win%', data = df_class)
ax.get_figure().savefig('09_win_rate_class_change.png')


"""10.1 Horse historic performance - curr quantile vs prior quantile
"""
#General corr - for all distance
df_rank = df[['Horse Name', 'quantile']]
df_rank.columns = ['Name', 'CurQuantile']
for i in range(1,6):
    df_rank['PriorQuantile_' + str(i)] = df_rank.groupby('Name')['CurQuantile'].shift(i)

#Apply exponential moving average 
df_rank['PriorEMAQuantile'] = df_rank.groupby('Name')['CurQuantile'].apply(lambda x: x.ewm(span = 4).mean())
df_rank['PriorEMAQuantile'] = df_rank.groupby('Name')['PriorEMAQuantile'].shift(1)

print(df_rank.corr())
ax = sns.heatmap(df_rank.corr(), annot = True)


#Historic by distance classification
df_rank = df[['Horse Name', 'Distance','quantile']]
dict_dist = {1000: 'Short',
             1150: 'Short',
             1200: 'Short',
             1400: 'Medium',
             1600: 'Medium',
             1650: 'Medium',
             1800: 'Medium',
             2000: 'Long',
             2200: 'Long',
             2400: 'Long'}

df_rank['dist_type'] = df_rank['Distance'].map(dict_dist)
df_rank = df_rank.rename(columns = {'Horse Name': 'Name',
                                    'quantile': 'CurQuantile'})
for i in range(1,6):
    df_rank['PriorQuantile_' + str(i)] = df_rank.groupby(['Name', 'dist_type'])['CurQuantile'].shift(i)

print(df_rank.iloc[:,2:].corr())


"""10.2 Jockey historic performance - curr quantile vs prior quantile - positively correlated
"""
#General corr - for all distance
df_rank = df[['Jockey', 'quantile']]
df_rank.columns = ['Name', 'CurQuantile']
for i in range(1,6):
    df_rank['PriorQuantile_' + str(i)] = df_rank.groupby('Name')['CurQuantile'].shift(i)

print(df_rank.corr())
ax = sns.heatmap(df_rank.corr(), annot = True)



"""10.3 Jockey & Horse historic performance - curr quantile vs prior quantile - positively correlated
"""
#General corr - for all distance
df_rank = df[['Jockey', 'Horse Name', 'quantile']]
df_rank.columns = ['J_Name', 'H_Name', 'CurQuantile']
for i in range(1,6):
    df_rank['PriorQuantile_' + str(i)] = df_rank.groupby(['J_Name', 'H_Name'])['CurQuantile'].shift(i)

print(df_rank.corr())


"""11 Odds before race vs final rank, and odds change impact to win rate
"""
df_odd = df[['Horse Name', 'Race Class', 'oddon', 'oddbr', 'oddfn', 'quantile']]
print(df_odd.corr())

df_odd = df[['Horse Name', 'Race Class', 'oddon', 'oddbr', 'oddfn', 'quantile', 'win', 'plc']]
ax = sns.pairplot(x_vars='oddon', y_vars='oddbr', hue = 'win', data = df_odd, size = 10).fig.suptitle('Win ON vs Win FN in Win Rate')
ax = sns.pairplot(x_vars='oddon', y_vars='oddbr', hue = 'plc', data = df_odd, size = 10).fig.suptitle('Win ON vs Win FN in Plc Rate')

ax = sns.heatmap(df_odd.corr(), annot = True)
ax.set_title('Correlation between odds and quantile')


"""12 Insignificant variables - HKJC Horse Rating and final rank
"""

df_rating = df[['Horse Name', 'Race Class', 'quantile', 'Rating']]
df_rating = df_rating.rename(columns = {'Rating': 'CurRating'})
for i in range(1,4):
    df_rating['PriorRating_' + str(i)] = df_rating.groupby('Horse Name')['CurRating'].shift(i)

print(df_rating[df_rating['Race Class'] == '1'].corr())
print(df_rating[df_rating['Race Class'] == '5'].corr())
print(df_rating.corr())

fig, axs = plt.subplots(nrows = 1, ncols = 1)

axs.set_title('Correlation b/w quantiles & rating (Class 1)')
sns.heatmap(df_rating[df_rating['Race Class'] == '1'].corr(), annot = True, ax = axs)

axs.set_title('Correlation b/w quantiles & rating (Class 5)')
sns.heatmap(df_rating[df_rating['Race Class'] == '5'].corr(), annot = True, ax = axs[1])


"""13  Insignificant variables - Rest days impact
"""
df_rest = df[['Horse Name', 'Race Date', 'Race Class', 'quantile']]
df_rest['PriorDate'] = df_rest.groupby('Horse Name')['Race Date'].shift(1)
df_rest['RestDays'] = (df_rest['Race Date'] - df_rest['PriorDate']).dt.days
df_rest['RestDays'] = [365 if x > 365 else x for x in df_rest['RestDays']]
df_rest = df_rest.dropna()

print(df_rest.corr())
ax = sns.distplot(df_rest['RestDays'], bins = 30)
ax.set(xlim = (0,150))
#most of the horses will race again in 36days
print(df_rest['RestDays'].describe())
#We want to check if extra rest/ too less rest would impact the performance
#Too much rest
print(df_rest[df_rest['RestDays'] > 36 * 1.5].corr())
ax = sns.heatmap(df_rest[df_rest['RestDays'] > 36].corr(), annot = True)
#Too less rest
print(df_rest[df_rest['RestDays'] < 16/2].corr())



"""14 Margin analysis - the winning distance could be significant indicator to next race
"""
df['margin'] = df['margin'].apply(float)
df_margin = df[['Race ID', 'Horse Name', 'Final Position', 'quantile','margin']]
df_margin = df_margin[~((df_margin['Final Position'] > 10) & (df_margin['margin'] < 1))]
#eliminate winner with non-zero margin
df_margin = df_margin[~((df_margin['Final Position'] == 1) & (df_margin['margin'] > 0))]
df_margin['prior_margin'] = df_margin.groupby('Horse Name')['margin'].shift(1)

ax = sns.boxplot(x = 'Final Position', y = 'margin', data = df_margin)
ax.set(ylim = (-3,40))

print(df_margin.corr())
ax = sns.heatmap(df_margin.corr(), annot = True)


"""15 Prev out-performance
"""
#Time outperformance
df['Track Type'] = df['Track'].map({'AWT': 'AWT'})
df['Track Type'] = df['Track Type'].fillna('Truf')

df = df.merge(df_stdtime, how = 'inner', on = ['Course', 'Track Type', 'Distance', 'Race Class'])
df['time_outperformance'] = df['Time_val'] - df['totalstandardtime']
df['prev_time_outperform'] = df.groupby('Horse Name')['time_outperformance'].shift(1)

df_timeoutperform = df[['Horse Name', 'quantile','prev_time_outperform']]
print(df_timeoutperform.corr())





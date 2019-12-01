# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 21:57:46 2019

@author: hokai
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
 

path_import = r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\RACEDATA\racedata split\\'
path_export = r'D:\Google Drive\Learning & Academics\HKU MSc CS\COMP7704 Dissertation\Data\\'

df1 = pd.read_csv(path_import + 'RaceData 01-06.txt' , sep=',' , header=None)
df2 = pd.read_csv(path_import + 'RaceData 07-12.txt' , sep=',' , header=None)
df3 = pd.read_csv(path_import + 'RaceData 13-18.txt' , sep=',' , header=None)
df4 = pd.read_csv(path_import + 'RaceData 19-24.txt' , sep=',' , header=None)
df5 = pd.read_csv(path_import + 'RaceData 25-30.txt' , sep=',' , header=None)
df6 = pd.read_csv(path_import + 'RaceData 31-36.txt' , sep=',' , header=None)
df7 = pd.read_csv(path_import + 'RaceData 37-40.txt' , sep=',' , header=None)

loopList = [df1, df2, df3, df4, df5, df6, df7]

df_2019 = pd.read_csv(path_export + 'RaceData_merged_2017to19.csv')

df = pd.DataFrame()

for x in loopList:
    df = pd.concat([df, x.loc[:,:len(x.columns)-2]], axis = 1)

col_names = ['Season'
              ,'Meeting No'
              ,'Race No in Year'
              ,'Race Date'
              ,'Race No in Mtg'
              ,'Distance'
              ,'horsetime1'
              ,'Course'
              ,'Track'
              ,'Race Class'
              ,'Going'
              ,'Cup Name'
              ,'Horse Name'
              ,'horsemarg1'
              ,'Age'
              ,'Draw'                                       #排位
              ,'Brand No'                                   #Kind of a unique ID for a horse, but will be reused!
              ,'Rating'
              ,'Loading'
              ,'Up/Down'                                    #Up or down of race class
              ,'Bleeding'
              ,'Reserve'
              ,'Jockey'
              ,'Stable'
              ,'Time'
              ,'Final Position'
              ,'Winning Time'
              ,'pos1','pos2','pos3','pos4','pos5'           #posrank in the mid, pos1 > pos2 > ... > pos5 > finish
              ,'margin'                                     #Margin = Distance from the winner
              ,'winticket'                                    #Count of tickets win
              ,'plcticket'                                  #Count of tickets plc
              ,'oddon'                                      #Overnight odds
              ,'secttime1','secttime2','secttime3','secttime4','secttime5','secttime6'   #WINNER Time sent on each section, Sec1 > Sec2 > ...
              ,'oddbr'                                      #***Odds before match starts?
              ,'oddfn'                                      #Final Odds
              ,'d_win1','d_win2'                            #Final dollars of the win
              ,'d_place1','d_place2','d_place3','d_place4'  #Final dollars of placeX if rank is X
              ,'d_quin'                                     #Final dollars of quin (place1 + place2)
              ,'bodyweight'
              ,'horsetime1','horsetime2','horsetime3','horsetime4','horsetime5','horsetime6' #HORSE Time sent on each section, Sec1 > Sec2 > ...
              ,'horsemarg1','horsemarg2','horsemarg3','horsemarg4','horsemarg5','horsemarg6'
              ,'horseno']

df.columns = col_names
df = df.loc[:,~df.columns.duplicated()]

#Reallocate the sequence
df = df[['Season', 'Meeting No', 'Race No in Year', 'Race Date',
       'Race No in Mtg', 'Distance', 'Course', 'Track',
       'Race Class', 'Going', 'Cup Name', 'Horse Name', 'Age',
       'Draw', 'Brand No', 'Rating', 'Loading', 'Up/Down', 'Bleeding',
       'Reserve', 'Jockey', 'Stable','Final Position', 'Time',  'Winning Time',
       'pos1', 'pos2', 'pos3', 'pos4', 'pos5', 'margin', 'winticket',
       'plcticket', 'oddon', 'secttime1', 'secttime2', 'secttime3', 'secttime4',
       'secttime5', 'secttime6', 'oddbr', 'oddfn', 'd_win1', 'd_win2',
       'd_place1', 'd_place2', 'd_place3', 'd_place4', 'd_quin', 'bodyweight',
       'horseno','horsetime1',
       'horsetime2', 'horsetime3', 'horsetime4', 'horsetime5', 'horsetime6',
       'horsemarg1',
       'horsemarg2', 'horsemarg3', 'horsemarg4', 'horsemarg5', 'horsemarg6'
       ]]

df = df[df['Season'] != 17]

#================== df 2019 ============================#

ls_oldname = ['season', 'meetingno', 'raceinyr', 'date', 'raceno', 'distance',
       'course', 'track', 'raceclass', 'going', 'cup', 'horse', 'age',
       'drawing', 'brandno', 'rating', 'netload', 'updn', 'bleeding',
       'reserve', 'jockey', 'stable', 'fp', 'time', 'wintime']

ls_rename = ['Season', 'Meeting No', 'Race No in Year', 'Race Date',
       'Race No in Mtg', 'Distance', 'Course', 'Track', 'Race Class', 'Going',
       'Cup Name', 'Horse Name', 'Age', 'Draw', 'Brand No', 'Rating',
       'Loading', 'Up/Down', 'Bleeding', 'Reserve', 'Jockey', 'Stable',
       'Final Position', 'Time', 'Winning Time'] 

dict_rename = dict(zip(ls_oldname, ls_rename))
df_2019 = df_2019.rename(index = str, columns = dict_rename)

#==================== Combine 2 dataframes =====================#
df_final = pd.concat([df, df_2019], sort = False)

df_final = df_final[['Season', 'Meeting No', 'Race No in Year', 'Race Date',
       'Race No in Mtg', 'Distance', 'Course', 'Track',
       'Race Class', 'Going', 'Cup Name', 'Horse Name', 'Age',
       'Draw', 'Brand No', 'Rating', 'Loading', 'Up/Down', 'Bleeding',
       'Reserve', 'Jockey', 'Stable','Final Position', 'Time',  'Winning Time',
       'pos1', 'pos2', 'pos3', 'pos4', 'pos5', 'margin', 'winticket',
       'plcticket', 'oddon', 'secttime1', 'secttime2', 'secttime3', 'secttime4',
       'secttime5', 'secttime6', 'oddbr', 'oddfn', 'd_win1', 'd_win2',
       'd_place1', 'd_place2', 'd_place3', 'd_place4', 'd_quin', 'bodyweight',
       'horseno','horsetime1',
       'horsetime2', 'horsetime3', 'horsetime4', 'horsetime5', 'horsetime6',
       'horsemarg1',
       'horsemarg2', 'horsemarg3', 'horsemarg4', 'horsemarg5', 'horsemarg6'
       ]]


#================= Manipulate data types ==============================#

df_final['Race Date'] = pd.to_datetime(df_final['Race Date'], format = "%m/%d/%Y")

#Create Race ID
df_final['Race ID'] = df_final['Race Date'].dt.strftime('%Y%m%d').astype(str) + '_' + df_final['Race No in Mtg'].map("{:02}".format)


#Exclude data after 20/3 because the race was not happened
df_final = df_final[~(df_final['Race Date'] > datetime.strptime('03/20/2019', '%m/%d/%Y'))]

#Exclude time that is unknown
df_final['Time'] = df_final['Time'].str.replace(' ', '')
df_final = df_final[~(df_final['Time'] == ':.')]
df_final = df_final[~(df_final['Time'] == '0:00.0')]
df_final = df_final.set_value(12356, 'Time', '01:39.0')

#Reset index
df_final = df_final.reset_index(drop = True)


df_final['Time'] = ['0' + x if len(x) == 6 else x for x in df_final['Time']]

#Convert time to seconds

df_final[['m', 's']] = df_final['Time'].str.split(':', expand = True)
df_final[['s1', 's2']] = df_final['s'].str.split('.', expand = True)
df_final['s2'] = [str(x) + '0' if len(x) == 1 else str(x) for x in df_final['s2']]
df_final['Time_val'] = df_final['m'].apply(int) * 60 + df_final['s1'].apply(int) + df_final['s2'].apply(int)*0.01

df_final.drop(columns = ['m', 's', 's1', 's2'])

#Convert data to strings
ls_ToStr = ['Course', 'Track', 'Race Class', 'Going', 'Cup Name', 'Horse Name', 'Brand No'
           , 'Up/Down', 'Bleeding', 'Jockey', 'Stable']

for x in ls_ToStr:
    df_final[x] = df_final[x].apply(str)
    
    
#Replace Race class to proper ones
dict_class = {'1-2': '2',
              '2-3': '3',
              '3-4': '4',
              '4-5': '5',
              '5-6': '6',
              '6':   '5',
              '7':   '5',
              '0':   'GRIFFIN',
              '9':   'CUP',
              '1-5': '5',
              '2-4': '4',
              '6-7': '5',
              '0-0': '2',
              'nan': '2',
              'P':   'CUP'}


df_final['Race Class'] = df_final['Race Class'].replace(dict_class)
df_final['Race Class'] = df_final['Race Class'].replace({'6': '5'})


#Replace Bleeding to proper ones
dict_bleed = {'.F.': False,
              '.T.': True,
              'False': False,
              'True': True}
df_final['Bleeding'] = df_final['Bleeding'].replace(dict_bleed)


#Replace Up/Down to proper Ones
dict_updown = {'Recls': 'New', 
               'V.Promote': 'Up', 
               'V.Demote': 'Down'}
df_final['Up/Down'] = df_final['Up/Down'].replace(dict_updown)


#Exclude those with margin is '.NULL.'
df_final = df_final[~(df_final['margin'] == '.NULL.')]

#Exlude those with abnormal gates
df_final = df_final[~((df_final['Draw'] == 0)|(df_final['Draw'] > 14))]

#================== Construct assistive variables =====================#

#Create Season variable
df_final['Season_yr'] = [x + 2000 if x < 50 else x + 1900 for x in df_final['Season']]

#Create win indicator
df_final['win'] = (df_final['Final Position'] == 1)*1

#Create place indicator
df_final['plc'] = (df_final['Final Position'].isin([1,2,3]))*1


#Count runners
df_cnt = df_final[['Race ID', 'Rating']].groupby('Race ID').count().reset_index()
df_cnt = df_cnt.rename(columns = {'Rating': 'count_runner'})
df_final = df_final.merge(df_cnt, how = 'left', on = 'Race ID')

#Quantile of the horse in a race
df_final['quantile'] = 1 - df_final['Final Position'] / df_final['count_runner']


#Construct Speed
df_final['speed'] = df_final['Distance'] / df_final['Time_val']
    
#Total Weight
df_final['Total weight'] = df_final['bodyweight'] + df_final['Loading']

#Jockey trainer combination
df_final['Jockey_trainer'] = df_final['Jockey'] + '_' + df_final['Stable']


#================= Ready to export =================================#
df_final = df_final.reset_index(drop = True)
    
df_final.to_csv(path_export + 'RaceData1.csv')
df_final.to_pickle(path_export + 'RaceData1.pkl')



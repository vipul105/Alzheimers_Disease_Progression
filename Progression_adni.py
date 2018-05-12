# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:04:05 2018

@author: Vipul Satone
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score   
from sklearn import mixture
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn import metrics 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.style.use('seaborn-colorblind')
import imageio

sns.set()

#Keeping the directory correctly 
import os
os.chdir('C:\\Users\\Vipul Satone\\health data')

# assessment data
cols = {}


'''
Begin Function
'''

def visualize_unclean_data(Train):
    null_in_cols = Train.isnull().sum()
    null_in_rows = Train.isnull().sum(axis = 1)
    print(Train.isnull().sum().sum())
    if (null_in_cols.sum() != 0):
        fig_1 = plt.figure()
        arr = plt.hist(null_in_rows , bins=np.arange(round(min(null_in_rows)), round(max(null_in_rows)) + null_in_rows.max()/8,null_in_rows.max()/8))
        plt.xlabel('count of features having NULL in them')
        plt.ylabel('Number of observations (that have *bin* \n number of features NULL in them)')
        plt.title('Graph to find out number of most empty observations \n (max null values can be %d)(Rows)' %(Train.shape[1]))
        for i in range(len(arr[0])):
            if (arr[0][i] != 0):
                plt.text(arr[1][i],arr[0][i],str(arr[0][i]))
        fig_1.show()

        fig_2 = plt.figure()
        arr1 = plt.hist(null_in_cols, bins=np.arange(round(min(null_in_cols)), round(max(null_in_cols)) + null_in_cols.max()/8,null_in_cols.max()/8))
        plt.xlabel('count of observations having NULL in them')
        plt.ylabel('Number of features (that have *bin* \n number of observations NULL in them)')
        plt.title('Graph to find out number of most empty features \n (max null values can be %d)(Columns)' %(Train.shape[0]))
        for i in range(len(arr1[0])):
            if (arr1[0][i] != 0):
                plt.text(arr1[1][i],arr1[0][i],str(arr1[0][i]))
        fig_2.show()


def data_info(cdr):
    print('Name of dataset is: ' + cdr.name) 
    print('\n0th level of columns is ')
    print(list(pd.Series(cdr.columns.get_level_values(0)).unique()) )
    print('\n1st level of columns is: ')
    print(list(pd.Series(cdr.columns.get_level_values(1)).unique()) )
    print('\nShape of datset is:')
    print(cdr.shape)
    print('\nTotal number of missing values: ')
    print(cdr.isnull().sum().sum())


'''
Begin Function
'''

# cognitive dementia rating
cols['cdr'] = ['RID','VISCODE2', 'CDMEMORY', 'CDORIENT', 'CDJUDGE' ,'CDCOMMUN' ,'CDHOME' ,'CDCARE']
cdr = pd.read_csv("ADNI\\Raw_Data\\Assessment\\CDR.csv",index_col='RID', usecols=cols['cdr'])
cdr1 = cdr.copy(deep = True)
cdr = cdr[cdr['VISCODE2'].isin(['bl','m12','m06']) ]  
visualize_unclean_data(cdr)
cdr = cdr.reset_index().set_index(['RID','VISCODE2'])
cdr = cdr[~cdr.index.duplicated()].unstack()
cdr = cdr[ (cdr.isnull().sum(axis = 1) <= 4) ]
cdr.name = 'Clinical Dementia Rating'
data_info(cdr)


# Neuropsychiatric Inventory Questionnaire
cols['npiq'] = ['RID', 'VISCODE2', 'NPIASEV','NPIBSEV', 'NPICSEV','NPIDSEV', 'NPIESEV', 'NPIFSEV',  \
    'NPIGSEV',  'NPIHSEV','NPIISEV',  'NPIJSEV',  'NPIKSEV', 'NPILSEV', 'NPIA','NPIB', 'NPIC','NPID'\
    , 'NPIE', 'NPIF','NPIG',  'NPIH','NPII',  'NPIJ',  'NPIK', 'NPIL']
npiq = pd.read_csv('ADNI\\Raw_Data\\Assessment\\NPIQ.csv', usecols=cols['npiq'], index_col = ['RID', 'VISCODE2'])
npiq1 = npiq.copy(deep = True) 
# there are some values which are -1 instead of 2 . replacing them with 2 to indicate readings not taken
npiq = npiq.replace({-1: 2})
#Replcing all -4 to 0 
npiq = npiq[~npiq.index.duplicated()].reset_index()
npiq = npiq[npiq.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
# removing observations with 2 (ie. no readings)
npiq = npiq[ ~( (npiq.NPIA == 2) | (npiq.NPIB == 2) | (npiq.NPIC == 2) | (npiq.NPID == 2) | \
    (npiq.NPIE == 2) | (npiq.NPIF == 2) | (npiq.NPIG == 2) | (npiq.NPIH == 2) | (npiq.NPII == 2) | \
    (npiq.NPIJ == 2) | (npiq.NPIK == 2) | (npiq.NPIL == 2) ) ]
#Multiplying whether of not with the severity ratings
npiq['NPIA_total'] = npiq['NPIA']*npiq['NPIASEV']
npiq['NPIB_total'] = npiq['NPIB']*npiq['NPIBSEV']
npiq['NPIC_total'] = npiq['NPIC']*npiq['NPICSEV']
npiq['NPID_total'] = npiq['NPID']*npiq['NPIDSEV']
npiq['NPIE_total'] = npiq['NPIE']*npiq['NPIESEV']
npiq['NPIF_total'] = npiq['NPIF']*npiq['NPIFSEV']
npiq['NPIG_total'] = npiq['NPIG']*npiq['NPIGSEV']
npiq['NPIH_total'] = npiq['NPIH']*npiq['NPIHSEV']
npiq['NPII_total'] = npiq['NPII']*npiq['NPIISEV']
npiq['NPIJ_total'] = npiq['NPIJ']*npiq['NPIJSEV']
npiq['NPIK_total'] = npiq['NPIK']*npiq['NPIKSEV']
npiq['NPIL_total'] = npiq['NPIL']*npiq['NPILSEV']
npiq = npiq.replace({-0: 0})
#Taking the multiplied columns only 
npiq = npiq[['NPIA_total', 'NPIB_total', 'NPIC_total', 'NPID_total', 'NPIE_total', 'NPIF_total', 'NPIG_total',\
             'NPIH_total', 'NPII_total', 'NPIJ_total', 'NPIK_total', 'NPIL_total']] # drop extra
new_col_list = ['NPIA_total', 'NPIB_total', 'NPIC_total', 'NPID_total', 'NPIE_total', 'NPIF_total',\
                'NPIG_total','NPIH_total', 'NPII_total', 'NPIJ_total', 'NPIK_total', 'NPIL_total']
npiq = npiq.unstack()
visualize_unclean_data(npiq)
npiq = npiq[ (npiq.isnull().sum(axis = 1) < 15) ]
for a in new_col_list: 
    npiq[a] = npiq[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
npiq.name = 'Neuropsychiatric Inventory Q'
data_info(npiq)


#NEUROBAT - Just using the total scores CLCOKSCOR, COPYSCOR, BNTTOTAL
cols['neurobat'] = ['RID', 'VISCODE2', 'CLOCKSCOR', 'COPYSCOR', 'LMSTORY', 'LIMMTOTAL', 'LIMMEND',
       'AVTOT1', 'AVERR1', 'AVTOT2', 'AVERR2', 'AVTOT3', 'AVERR3', 'AVTOT4',
       'AVERR4', 'AVTOT5', 'AVERR5', 'AVTOT6', 'AVERR6', 'AVTOTB', 'AVERRB',
       'AVENDED', 'DSPANFOR', 'DSPANFLTH', 'DSPANBAC', 'DSPANBLTH',
       'CATANIMSC', 'CATANPERS', 'CATANINTR', 'CATVEGESC', 'CATVGPERS',
       'CATVGINTR', 'TRAASCOR', 'TRAAERRCOM', 'TRAAERROM', 'TRABSCOR',
       'TRABERRCOM', 'TRABERROM', 'DIGITSCOR', 'LDELBEGIN', 'LDELTOTAL',
       'LDELCUE','BNTTOTAL', 'AVDELBEGAN', 'AVDEL30MIN', 'AVDELERR1',
       'AVDELTOT', 'AVDELERR2', 'ANARTND', 'ANARTERR']
neurobat_1 = pd.read_csv('ADNI\\Raw_Data\\Assessment\\NEUROBAT.csv', usecols=cols['neurobat'], index_col = ['RID', 'VISCODE2'])
# neurobat Clock
cols['neurobat_clock'] = ['RID', 'VISCODE2', 'CLOCKSCOR']
neurobat_clock = pd.read_csv('ADNI\\Raw_Data\\Assessment\\NEUROBAT.csv', usecols=cols['neurobat_clock'], index_col = ['RID', 'VISCODE2'])
neurobat_clock1 = neurobat_clock.copy(deep = True) 
neurobat_clock = neurobat_clock[~neurobat_clock.index.duplicated()].reset_index()
neurobat_clock = neurobat_clock[neurobat_clock.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
visualize_unclean_data(neurobat_clock)
neurobat_clock = neurobat_clock.unstack()
neurobat_clock = neurobat_clock[ (neurobat_clock.isnull().sum(axis = 1) <= 1) ]
new_col_list_neurobat_clock = neurobat_clock.columns.levels[0]
for a in new_col_list_neurobat_clock: 
    neurobat_clock[a] = neurobat_clock[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
neurobat_clock.name = 'Neuropsychological Battery (subdata - clock)'
data_info(neurobat_clock)
# neurobat Copy
cols['neurobat_copy'] = ['RID', 'VISCODE2', 'COPYSCOR']
neurobat_copy = pd.read_csv('ADNI\\Raw_Data\\Assessment\\NEUROBAT.csv', usecols=cols['neurobat_copy'], index_col = ['RID', 'VISCODE2'])
neurobat_copy1 = neurobat_copy.copy(deep = True) 
neurobat_copy = neurobat_copy[~neurobat_copy.index.duplicated()].reset_index()
neurobat_copy = neurobat_copy[neurobat_copy.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
visualize_unclean_data(neurobat_copy)
neurobat_copy = neurobat_copy.unstack()
neurobat_copy = neurobat_copy[ (neurobat_copy.isnull().sum(axis = 1) <= 1) ]
new_col_list_neurobat_copy = neurobat_copy.columns.levels[0]
for a in new_col_list_neurobat_copy: 
    neurobat_copy[a] = neurobat_copy[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
neurobat_copy.name = 'Neuropsychological Battery (subdata - copy)'
data_info(neurobat_copy)
# neurobat Story
cols['neurobat_limm_story'] = ['RID', 'VISCODE2', 'LIMMTOTAL']
neurobat_limm_story = pd.read_csv('ADNI\\Raw_Data\\Assessment\\NEUROBAT.csv', usecols=cols['neurobat_limm_story'], index_col = ['RID', 'VISCODE2'])
neurobat_limm_story1 = neurobat_limm_story.copy(deep = True) 
neurobat_limm_story = neurobat_limm_story[~neurobat_limm_story.index.duplicated()].reset_index()
neurobat_limm_story = neurobat_limm_story[neurobat_limm_story.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
visualize_unclean_data(neurobat_limm_story)
neurobat_limm_story = neurobat_limm_story.unstack()
neurobat_limm_story = neurobat_limm_story.drop(['m06','bl'], axis=1, level=1)
neurobat_limm_story = neurobat_limm_story[ (neurobat_limm_story.isnull().sum(axis = 1) < 1) ]
neurobat_limm_story.name = 'Neuropsychological Battery (subdata - story)'
data_info(neurobat_limm_story)
# neurobat digit span
cols['neurobat_dspan'] = ['RID', 'VISCODE2','DSPANFOR', 'DSPANFLTH', 'DSPANBAC', 'DSPANBLTH']
neurobat_dspan = pd.read_csv('ADNI\\Raw_Data\\Assessment\\NEUROBAT.csv', usecols=cols['neurobat_dspan'], index_col = ['RID', 'VISCODE2'])
neurobat_dspan1 = neurobat_dspan.copy(deep = True) 
neurobat_dspan = neurobat_dspan[~neurobat_dspan.index.duplicated()].reset_index()
neurobat_dspan = neurobat_dspan[neurobat_dspan.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
visualize_unclean_data(neurobat_dspan)
neurobat_dspan = neurobat_dspan[ (neurobat_dspan.isnull().sum(axis = 1) < 4) ]
neurobat_dspan = neurobat_dspan.unstack()
neurobat_dspan = neurobat_dspan[ (neurobat_dspan.isnull().sum(axis = 1) <6) ]
new_col_list_neurobat_dspan = neurobat_dspan.columns.levels[0]
for a in new_col_list_neurobat_dspan: 
    neurobat_dspan[a] = neurobat_dspan[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
neurobat_dspan.name = 'Neuropsychological Battery (subdata - digit span)'
data_info(neurobat_dspan)
# Neurobat Categorical Fluency
cols['neurobat_cat_flu'] = ['RID', 'VISCODE2','CATANIMSC', 'CATANPERS', 'CATANINTR', 'CATVEGESC', 'CATVGPERS','CATVGINTR']
neurobat_cat_flu = pd.read_csv('ADNI\\Raw_Data\\Assessment\\NEUROBAT.csv', usecols=cols['neurobat_cat_flu'], index_col = ['RID', 'VISCODE2'])
neurobat_cat_flu1 = neurobat_cat_flu.copy(deep = True) 
neurobat_cat_flu = neurobat_cat_flu[~neurobat_cat_flu.index.duplicated()].reset_index()
neurobat_cat_flu = neurobat_cat_flu[neurobat_cat_flu.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
neurobat_cat_flu = neurobat_cat_flu.replace({-1: np.NAN})
visualize_unclean_data(neurobat_cat_flu)
neurobat_cat_flu = neurobat_cat_flu[ (neurobat_cat_flu.isnull().sum(axis = 1) < 4) ]
del neurobat_cat_flu['CATVEGESC']
del neurobat_cat_flu['CATVGPERS']
del neurobat_cat_flu['CATVGINTR']
neurobat_cat_flu = neurobat_cat_flu.unstack()
neurobat_cat_flu = neurobat_cat_flu[ (neurobat_cat_flu.isnull().sum(axis = 1) <4) ]
new_col_list_neurobat_cat_flu = neurobat_cat_flu.columns.levels[0]
for a in new_col_list_neurobat_cat_flu: 
    neurobat_cat_flu[a] = neurobat_cat_flu[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
neurobat_cat_flu.name = 'Neuropsychological Battery (subdata - category fluency : only animal examples)'
data_info(neurobat_cat_flu)
# Neurobat trail making test
cols['neurobat_trail'] = ['RID', 'VISCODE2', 'TRAASCOR', 'TRAAERRCOM', 'TRAAERROM', 'TRABSCOR','TRABERRCOM', 'TRABERROM']
neurobat_trail = pd.read_csv('ADNI\\Raw_Data\\Assessment\\NEUROBAT.csv', usecols=cols['neurobat_trail'], index_col = ['RID', 'VISCODE2'])
neurobat_trail1 = neurobat_trail.copy(deep = True) 
neurobat_trail = neurobat_trail[~neurobat_trail.index.duplicated()].reset_index()
neurobat_trail = neurobat_trail[neurobat_trail.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
visualize_unclean_data(neurobat_trail)
neurobat_trail = neurobat_trail[ (neurobat_trail.isnull().sum(axis = 1) < 3) ]
neurobat_trail = neurobat_trail.unstack()
neurobat_trail = neurobat_trail[ (neurobat_trail.isnull().sum(axis = 1) <=6) ]
new_col_list_neurobat_trail = neurobat_trail.columns.levels[0]
for a in new_col_list_neurobat_trail: 
    neurobat_trail[a] = neurobat_trail[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
neurobat_trail.name = 'Neuropsychological Battery (subdata - Trail making)'
data_info(neurobat_trail)
# Neurobat Rey Auditory Verbal Learning Test
cols['neurobat_av'] = ['RID', 'VISCODE2','AVTOT1', 'AVDELERR1','AVDELTOT', 'AVERR1', 'AVTOT2', 'AVERR2', 'AVTOT3', \
    'AVERR3','AVDELERR2', 'AVTOT4','AVERR4', 'AVTOT5', 'AVERR5', 'AVTOT6', 'AVERR6', 'AVTOTB', 'AVERRB','AVDEL30MIN']
neurobat_av = pd.read_csv('ADNI\\Raw_Data\\Assessment\\NEUROBAT.csv', usecols=cols['neurobat_av'], index_col = ['RID', 'VISCODE2'])
neurobat_av1 = neurobat_av.copy(deep = True) 
neurobat_av = neurobat_av[~neurobat_av.index.duplicated()].reset_index()
neurobat_av = neurobat_av[neurobat_av.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
visualize_unclean_data(neurobat_av)
neurobat_av = neurobat_av.unstack()
neurobat_av = neurobat_av[ (neurobat_av.isnull().sum(axis = 1) <25) ]
new_col_list_neurobat_av = neurobat_av.columns.levels[0]
for a in new_col_list_neurobat_av: 
    neurobat_av[a] = neurobat_av[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
neurobat_av.name = 'Neuropsychological Battery (subdata - av)'
data_info(neurobat_av)
# neurobat digit score
cols['neurobat_digit_score'] = ['RID', 'VISCODE2','DIGITSCOR']
neurobat_digit_score = pd.read_csv('ADNI\\Raw_Data\\Assessment\\NEUROBAT.csv', usecols=cols['neurobat_digit_score'], index_col = ['RID', 'VISCODE2'])
neurobat_digit_score1 = neurobat_digit_score.copy(deep = True) 
neurobat_digit_score = neurobat_digit_score[~neurobat_digit_score.index.duplicated()].reset_index()
neurobat_digit_score = neurobat_digit_score[neurobat_digit_score.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
visualize_unclean_data(neurobat_digit_score)
neurobat_digit_score = neurobat_digit_score[ (neurobat_digit_score.isnull().sum(axis = 1) < 1) ]
neurobat_digit_score = neurobat_digit_score.unstack()
neurobat_digit_score = neurobat_digit_score[ (neurobat_digit_score.isnull().sum(axis = 1) <=1) ]
new_col_list_neurobat_digit_score = neurobat_digit_score.columns.levels[0]
for a in new_col_list_neurobat_digit_score: 
    neurobat_digit_score[a] = neurobat_digit_score[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
neurobat_digit_score.name = 'Neuropsychological Battery (subdata - digit score)'
data_info(neurobat_digit_score)
# neurobat Logical Memory delayed recall
cols['neurobat_logical_memory'] = ['RID', 'VISCODE2','LDELTOTAL','LDELCUE']
neurobat_logical_memory = pd.read_csv('ADNI\\Raw_Data\\Assessment\\NEUROBAT.csv', usecols=cols['neurobat_logical_memory'], index_col = ['RID', 'VISCODE2'])
neurobat_logical_memory1 = neurobat_logical_memory.copy(deep = True) 
neurobat_logical_memory = neurobat_logical_memory[~neurobat_logical_memory.index.duplicated()].reset_index()
neurobat_logical_memory = neurobat_logical_memory[neurobat_logical_memory.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
visualize_unclean_data(neurobat_logical_memory)
neurobat_logical_memory = neurobat_logical_memory[ (neurobat_logical_memory.isnull().sum(axis = 1) < 1) ]
neurobat_logical_memory = neurobat_logical_memory.unstack()
neurobat_logical_memory.name = 'Neuropsychological Battery (subdata - logical memeory test)'
data_info(neurobat_logical_memory)
# neurobat Boston naming test
cols['neurobat_boston_naming_test'] = ['RID', 'VISCODE2', 'BNTSPONT','BNTSTIM','BNTCSTIM','BNTPHON','BNTCPHON']
neurobat_boston_naming_test = pd.read_csv('ADNI\\Raw_Data\\Assessment\\NEUROBAT.csv', usecols=cols['neurobat_boston_naming_test'], index_col = ['RID', 'VISCODE2'])
neurobat_boston_naming_test1 = neurobat_boston_naming_test.copy(deep = True) 
neurobat_boston_naming_test = neurobat_boston_naming_test[~neurobat_boston_naming_test.index.duplicated()].reset_index()
neurobat_boston_naming_test = neurobat_boston_naming_test[neurobat_boston_naming_test.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
visualize_unclean_data(neurobat_boston_naming_test)
neurobat_boston_naming_test = neurobat_boston_naming_test[ (neurobat_boston_naming_test.isnull().sum(axis = 1) < 5) ]
neurobat_boston_naming_test = neurobat_boston_naming_test.unstack()
neurobat_boston_naming_test = neurobat_boston_naming_test[ (neurobat_boston_naming_test.isnull().sum(axis = 1) <6) ]
new_col_list_neurobat_boston_naming_test = neurobat_boston_naming_test.columns.levels[0]
for a in new_col_list_neurobat_boston_naming_test: 
    neurobat_boston_naming_test[a] = neurobat_boston_naming_test[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
neurobat_boston_naming_test.name = 'Neuropsychological Battery (subdata - Boston naming test)'
data_info(neurobat_boston_naming_test)
# neurobat American National Adult Reading Test
cols['neurobat_anrt'] = ['RID', 'VISCODE2', 'ANARTND']
neurobat_anrt = pd.read_csv('ADNI\\Raw_Data\\Assessment\\NEUROBAT.csv', usecols=cols['neurobat_anrt'], index_col = ['RID', 'VISCODE2'])
neurobat_anrt1 = neurobat_anrt.copy(deep = True) 
neurobat_anrt = neurobat_anrt[~neurobat_anrt.index.duplicated()].reset_index()
neurobat_anrt = neurobat_anrt[neurobat_anrt.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
visualize_unclean_data(neurobat_anrt)
neurobat_anrt = neurobat_anrt[ (neurobat_anrt.isnull().sum(axis = 1) < 1) ]
#neurobat_clock = neurobat_clock[ (neurobat_clock.isnull().sum(axis = 1) <= 1) ]
neurobat_anrt = neurobat_anrt.unstack()
neurobat_anrt = neurobat_anrt[ (neurobat_anrt.isnull().sum(axis = 1) <=1) ]
new_col_list_neurobat_anrt = neurobat_anrt.columns.levels[0]
for a in new_col_list_neurobat_anrt: 
    neurobat_anrt[a] = neurobat_anrt[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
neurobat_anrt.name = 'Neuropsychological Battery (subdata - American national reading test)'
data_info(neurobat_anrt)

neurobat1 = pd.merge(neurobat_clock,neurobat_copy , left_index = True, right_index = True, how='inner')
neurobat1 = pd.merge(neurobat1,neurobat_limm_story , left_index = True, right_index = True, how='inner')
neurobat1 = pd.merge(neurobat1,neurobat_av , left_index = True, right_index = True, how='inner')
neurobat1 = pd.merge(neurobat1,neurobat_cat_flu , left_index = True, right_index = True, how='inner')
neurobat1 = pd.merge(neurobat1,neurobat_trail , left_index = True, right_index = True, how='inner')
neurobat1 = pd.merge(neurobat1,neurobat_logical_memory , left_index = True, right_index = True, how='inner')
neurobat1 = pd.merge(neurobat1,neurobat_boston_naming_test , left_index = True, right_index = True, how='inner')

#Alzheimer’s Disease Assessment Scale
cols['adascores'] = ['RID', 'TOTALMOD', 'VISCODE','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10',\
    'Q11','Q12','Q14']
adascores = pd.read_csv("ADNI\\Raw_Data\\Assessment\\ADASSCORES.csv",index_col='RID', usecols=cols['adascores'])
adascores1 = adascores.copy(deep = True)
adascores1['VISCODE2'] = adascores1['VISCODE']
del adascores1['VISCODE']
adascores['VISCODE2'] = adascores['VISCODE']
del adascores['VISCODE']
adascores = adascores[adascores['VISCODE2'].isin(['m06','m12','bl']) ]  
adascores = adascores.reset_index().set_index(['RID','VISCODE2'])
visualize_unclean_data(adascores)
adascores = adascores.replace({-4:np.NAN})
adascores = adascores[~adascores.index.duplicated()].unstack()
adascores = adascores[ (adascores.isnull().sum(axis = 1) <25) ]
new_col_list_adascores = adascores.columns.levels[0]
for a in new_col_list_adascores: 
    adascores[a] = adascores[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
adascores = adascores[ (adascores.isnull().sum(axis = 1) <1) ]
adascores.name = 'Neuropsychological Battery (subdata - American national reading test)'
data_info(adascores)

# baseline symptoms
cols['recbllog'] = ['RID','VISCODE2', 'BSXSYMNO', 'BSXSEVER', 'BSXCHRON','BSXCONTD']
recbllog = pd.read_csv('ADNI\\Raw_Data\\medical\\RECBLLOG.csv', usecols=cols['recbllog'], index_col = ['RID', 'VISCODE2'])
recbllog1 = recbllog.copy(deep = True) 
recbllog = recbllog[~recbllog.index.duplicated()].reset_index()
recbllog = recbllog[recbllog.VISCODE2.isin(['bl'])]
del recbllog['VISCODE2']
recbllog = recbllog[~recbllog.index.duplicated()]
recbllog = recbllog[ (recbllog.isnull().sum(axis = 1) < 2) ]
recbllog['BSXSYMNO']= recbllog.BSXSYMNO.replace(list(range(1,29)),['Nausea','Vomiting','Diarrhea','Constipation','Abdominal discomfort','Sweating','Dizziness','Low energy','Drowsiness','Blurred vision','Headache','Dry mouth','Shortness of breath','Coughing','Palpitations','Chest pain','Urinary discomfort','Urinary frequency', 'Ankle swelling','Musculoskeletal pain','Rash','Insomnia','Depressed mood','Crying','Elevated mood','Wandering','Fall','Other'])
recbllog =recbllog.set_index(['RID','BSXSYMNO'])
recbllog['BSXCONTD']= recbllog.BSXCONTD.replace([0],[2])
#Unstacking 
recbllog = recbllog.unstack()
recbllog = recbllog.fillna(0)
recbllog = recbllog.T.swaplevel(i=0, j=1, axis=0)
recbllog = recbllog.T
recbllog2 = recbllog.copy(deep = True)
recbllog2['VISCODE2'] = 'bl'
recbllog2.name = 'Baseline Symptome'
data_info(recbllog2)

# Mini Mental State Exam
cols['mmse'] = ['RID', 'VISCODE2','MMSCORE','MMDATE','MMYEAR','MMMONTH','MMDAY','MMSEASON','MMHOSPIT',\
    'MMFLOOR','MMCITY','MMAREA','MMSTATE','MMBALL','MMFLAG','MMTREE','MMD','MML','MMR','MMO','MMW',\
    'MMBALLDL','MMFLAGDL','MMTREEDL','MMWATCH','MMPENCIL','MMREPEAT','MMHAND','MMFOLD','MMONFLR','MMREAD',\
    'MMWRITE','MMDRAW']
mmse = pd.read_csv('ADNI\\Raw_Data\\Assessment\\MMSE.csv', usecols=cols['mmse'], index_col = ['RID', 'VISCODE2'])
mmse1 = mmse.copy(deep = True)
mmse = mmse[~mmse.index.duplicated()].reset_index()
mmse = mmse[mmse.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
mmse = mmse.replace({-4:np.NAN})
mmse = mmse.replace({-1:np.NAN})
visualize_unclean_data(mmse)
mmse = mmse[ (mmse.isnull().sum(axis = 1) < 10) ]
mmse = mmse.unstack()
mmse = mmse[ (mmse.isnull().sum(axis = 1) < 20) ]
mmse = mmse[~mmse.index.duplicated()]
mmse.name = 'Mini Mental State Exam'
data_info(mmse)

# Geriatric Depression Scale
cols['geriatric'] = ['VISCODE2', 'RID', 'GDTOTAL']
geriatric = pd.read_csv("ADNI\\Raw_Data\\Assessment\\GDSCALE.csv", index_col='RID', usecols=cols['geriatric'])
geriatric1 = geriatric.copy(deep = True)
geriatric = geriatric.replace({-4:np.NAN})
geriatric = geriatric.replace({-1:np.NAN})
visualize_unclean_data(geriatric)
geriatric = geriatric[geriatric['VISCODE2'].isin(['bl','m12','m06']) ]  
geriatric = geriatric.reset_index().set_index(['RID','VISCODE2'])
geriatric = geriatric[ (geriatric.isnull().sum(axis = 1) ==0) ]
geriatric = geriatric[~geriatric.index.duplicated()].unstack()
geriatric = geriatric[ (geriatric.isnull().sum(axis = 1) ==0) ]
geriatric.name = 'Geriatric depression scale'
data_info(geriatric)

# Crane Lab (UW) Neuropsych Summary Score
cols['UWNPSYCHSUM_10_27_17'] = ['RID', 'VISCODE2', 'ADNI_MEM', 'ADNI_EF']
uwn = pd.DataFrame(pd.read_csv("ADNI\\Raw_Data\\Assessment\\UWNPSYCHSUM_10_27_17.csv",index_col= ['RID','VISCODE2'], usecols=cols['UWNPSYCHSUM_10_27_17']))
uwn1 = uwn.copy(deep = True)
uwn = uwn[~uwn.index.duplicated()].reset_index()
uwn = uwn[uwn.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
uwn = uwn[~uwn.index.duplicated()].unstack()
uwn = uwn[ (uwn.isnull().sum(axis = 1) < 3) ]
visualize_unclean_data(uwn)
new_col_list_uwn = uwn.columns.levels[0]
for a in new_col_list_uwn: 
    uwn[a] = uwn[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
uwn = uwn[ (uwn.isnull().sum(axis = 1) <1) ]
uwn.name = 'Crane Lab (UW) Neuropsych Summary Score'
data_info(uwn)

# baseline symptomes 
cols['blscheck'] = ['RID', 'VISCODE2', 'BCNAUSEA', 'BCVOMIT', 'BCDIARRH', 'BCCONSTP',
       'BCABDOMN', 'BCSWEATN', 'BCDIZZY', 'BCENERGY', 'BCDROWSY', 'BCVISION',
       'BCHDACHE', 'BCDRYMTH', 'BCBREATH', 'BCCOUGH', 'BCPALPIT', 'BCCHEST',
       'BCURNDIS', 'BCURNFRQ', 'BCANKLE', 'BCMUSCLE', 'BCRASH', 'BCINSOMN',
       'BCDPMOOD', 'BCCRYING', 'BCELMOOD', 'BCWANDER', 'BCFALL']
blscheck = pd.read_csv("ADNI\\Raw_Data\\medical\\BLSCHECK.csv", index_col='RID', usecols=cols['blscheck'])
blscheck1 = blscheck.copy(deep = True)
# for adni2
blscheck_adni2 = blscheck[blscheck['VISCODE2'].isin(['sc']) ]  
blscheck_adni2 = blscheck_adni2.reset_index().set_index(['RID','VISCODE2'])
visualize_unclean_data(blscheck_adni2)
blscheck_adni2 = blscheck_adni2[ (blscheck_adni2.isnull().sum(axis = 1) <15) ]
blscheck_adni2 = blscheck_adni2[~blscheck_adni2.index.duplicated()].unstack()
blscheck_adni2.name = 'baseline symptoms for adni2'
data_info(blscheck_adni2)
# for adni1
blscheck_adni1 = blscheck[blscheck['VISCODE2'].isin(['bl']) ]  
blscheck_adni1 = blscheck_adni1.reset_index().set_index(['RID','VISCODE2'])
visualize_unclean_data(blscheck_adni1)
blscheck_adni1 = blscheck_adni1[~blscheck_adni1.index.duplicated()].unstack()
blscheck_adni1.name = 'baseline symptoms for adni1'
data_info(blscheck_adni1)

# Neuropsychiatric Inventory
cols['npi'] = ['RID', 'VISCODE2','NPIATOT', 'NPIBTOT', 'NPICTOT',  'NPIDTOT', \
    'NPIETOT', 'NPIFTOT', 'NPIGTOT', 'NPIHTOT', 'NPIITOT', 'NPIJTOT','NPIKTOT', 'NPILTOT']
npi = pd.read_csv('ADNI\\Raw_Data\\Assessment\\NPI.csv', usecols=cols['npi'], index_col = ['RID', 'VISCODE2'])
npi1 = npi.copy(deep = True) 
npi = npi[~npi.index.duplicated()].reset_index()
# m12 only
npi_m12 = npi[npi.VISCODE2.isin(['m12'])].set_index(['RID','VISCODE2'])
npi_m12 = npi_m12[~npi_m12.index.duplicated()].unstack()
visualize_unclean_data(npi_m12)
npi_m12 = npi_m12[ (npi_m12.isnull().sum(axis = 1) < 12) ]
# baseline
npi_bl = npi[npi.VISCODE2.isin(['bl'])].set_index(['RID','VISCODE2'])
npi_bl = npi_bl[~npi_bl.index.duplicated()].unstack()
visualize_unclean_data(npi_bl)
npi_bl = npi_bl[ (npi_bl.isnull().sum(axis = 1) < 12) ]
# both baseline and m12
npi_all = npi[npi.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
npi_all = npi_all[~npi_all.index.duplicated()].unstack()
visualize_unclean_data(npi_all)
npi_all = npi_all[ (npi_all.isnull().sum(axis = 1) < 10) ]
npi_all.name = 'Neuropsychiatric Inventory'
data_info(npi_all)

# everyday cognition by study partner
# not considering VISSPAT5
cols['ecogsp'] =  ['RID', 'VISCODE2', \
       'MEMORY1', 'MEMORY2', 'MEMORY3', 'MEMORY4', 'MEMORY5',\
       'MEMORY6', 'MEMORY7', 'MEMORY8', 'LANG1', 'LANG2', 'LANG3', 'LANG4',\
       'LANG5', 'LANG6', 'LANG7', 'LANG8', 'LANG9', 'VISSPAT1', 'VISSPAT2',\
       'VISSPAT3', 'VISSPAT4', 'VISSPAT6', 'VISSPAT7', 'VISSPAT8',\
       'PLAN1', 'PLAN2', 'PLAN3', 'PLAN4', 'PLAN5', 'ORGAN1', 'ORGAN2',\
       'ORGAN3', 'ORGAN4', 'ORGAN5', 'ORGAN6', 'DIVATT1', 'DIVATT2', 'DIVATT3',\
       'DIVATT4']
ecogsp = pd.read_csv("ADNI\\Raw_Data\\Assessment\\ECOGSP.csv",index_col='RID', usecols=cols['ecogsp'])
ecogsp1 = ecogsp.copy(deep = True)
ecogsp = ecogsp[ecogsp['VISCODE2'].isin(['bl','m12','m06']) ]  
ecogsp = ecogsp.reset_index().set_index(['RID','VISCODE2'])
ecogsp = ecogsp[~ecogsp.index.duplicated()]
ecogsp = ecogsp[ (ecogsp.isnull().sum(axis = 1) <= 30) ]
visualize_unclean_data(ecogsp)
ecogsp = ecogsp.unstack()
ecogsp = ecogsp[ (ecogsp.isnull().sum(axis = 1) < 41) ]
new_col_list_ecogsp = ecogsp.columns.levels[0]
for a in new_col_list_ecogsp: 
    ecogsp[a] = ecogsp[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
ecogsp = ecogsp[ (ecogsp.isnull().sum(axis = 1) <1) ]
ecogsp.name = 'Everyday cognition - study partner'
data_info(ecogsp)

# everyday cognition by participant
cols['ecogpt']  = ['RID', 'VISCODE2', 'MEMORY1', 'MEMORY2', 'MEMORY3', 'MEMORY4', 'MEMORY5',\
       'MEMORY6', 'MEMORY7', 'MEMORY8', 'LANG1', 'LANG2', 'LANG3', 'LANG4',\
       'LANG5', 'LANG6', 'LANG7', 'LANG8', 'LANG9', 'VISSPAT1', 'VISSPAT2',\
       'VISSPAT3', 'VISSPAT4', 'VISSPAT6', 'VISSPAT7', 'VISSPAT8',\
       'PLAN1', 'PLAN2', 'PLAN3', 'PLAN4', 'PLAN5', 'ORGAN1', 'ORGAN2',\
       'ORGAN3', 'ORGAN4', 'ORGAN5', 'ORGAN6', 'DIVATT1', 'DIVATT2', 'DIVATT3',\
       'DIVATT4']
ecogpt = pd.read_csv("ADNI\\Raw_Data\\Assessment\\ecogpt.csv",index_col='RID', usecols=cols['ecogpt'])
ecogpt1 = ecogpt.copy(deep = True)
ecogpt = ecogpt[ecogpt['VISCODE2'].isin(['bl','m12','m06']) ]  
ecogpt = ecogpt.reset_index().set_index(['RID','VISCODE2'])
ecogpt = ecogpt[~ecogpt.index.duplicated()]
ecogpt = ecogpt[ (ecogpt.isnull().sum(axis = 1) <= 30) ]
visualize_unclean_data(ecogpt)
ecogpt = ecogpt.unstack()
ecogpt = ecogpt[ (ecogpt.isnull().sum(axis = 1) < 41) ]
new_col_list_ecogpt = ecogpt.columns.levels[0]
for a in new_col_list_ecogpt: 
    ecogpt[a] = ecogpt[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
ecogpt = ecogpt[ (ecogpt.isnull().sum(axis = 1) <1) ]
ecogpt.name = 'Everyday cognition - participant'
data_info(ecogpt)

# Montreal Cognitive Assessment
cols['moca'] = ['RID','VISCODE2', 'TRAILS', 'CUBE', 'CLOCKCON', 'CLOCKNO', 'CLOCKHAN','LION', 'RHINO', 'CAMEL', 'IMMT1W1', 'IMMT1W2', 'IMMT1W3', 'IMMT1W4',
       'IMMT1W5', 'IMMT2W1', 'IMMT2W2', 'IMMT2W3', 'IMMT2W4', 'IMMT2W5','DIGFOR', 'DIGBACK', 'LETTERS', 'SERIAL1', 'SERIAL2', 'SERIAL3',
       'SERIAL4', 'SERIAL5', 'REPEAT1', 'REPEAT2', 'FFLUENCY', 'ABSTRAN','ABSMEAS', 'DELW1', 'DELW2', 'DELW3', 'DELW4', 'DELW5', 'DATE', 'MONTH',
       'YEAR', 'DAY', 'PLACE', 'CITY']
cols['moca_trail_making'] = ['TRAILS']
cols['moca_visuosoconstructional'] = ['CUBE', 'CLOCKCON', 'CLOCKNO', 'CLOCKHAN']
cols['moca_naming'] = [ 'LION', 'RHINO', 'CAMEL']
cols['moca_immediate_recall'] = [ 'IMMT2W1', 'IMMT2W2', 'IMMT2W3', 'IMMT2W4', 'IMMT2W5','IMMT1W1', 'IMMT1W2', 'IMMT1W3', 'IMMT1W4', 'IMMT1W5']
cols['moca_attention'] = [ 'DIGFOR', 'DIGBACK', 'LETTERS', 'SERIAL1', 'SERIAL2', 'SERIAL3','SERIAL4', 'SERIAL5']
cols['moca_sen_repetetion'] = ['REPEAT1','REPEAT2']
cols['moca_fluency'] = ['FFLUENCY']
cols['moca_abstraction'] = ['ABSTRAN','ABSMEAS']
cols['moca_delayed_word_recall'] = [ 'DELW1', 'DELW2', 'DELW3', 'DELW4', 'DELW5']
cols['moca_orientation'] = [ 'DATE', 'MONTH', 'YEAR', 'DAY', 'PLACE', 'CITY' ]
moca = pd.read_csv('ADNI\\Raw_Data\\Assessment\\MOCA.csv', usecols=cols['moca'], index_col = ['RID', 'VISCODE2'])
moca['moca_trail_making'] = moca[cols['moca_trail_making']].sum(axis=1)
moca['moca_visuosoconstructional'] = moca[cols['moca_visuosoconstructional']].sum(axis=1)
moca['moca_naming'] = moca[cols['moca_naming']].sum(axis = 1)
moca['moca_immediate_recall'] = moca[cols['moca_immediate_recall']].sum(axis=1)
moca['moca_attention'] = moca[cols['moca_attention']].sum(axis=1)
moca['moca_sen_repetetion'] = moca[cols['moca_sen_repetetion']].sum(axis=1)
moca['moca_fluency'] = moca[cols['moca_fluency']].sum(axis=1)
moca['moca_abstraction'] = moca[cols['moca_abstraction']].sum(axis=1)
moca['moca_delayed_word_recall'] = moca[cols['moca_delayed_word_recall']].sum(axis=1)
moca['moca_orientation'] = moca[cols['moca_orientation']].sum(axis=1)
moca = moca[['moca_trail_making', 'moca_visuosoconstructional', 'moca_naming', 'moca_attention', 'moca_immediate_recall',\
             'moca_sen_repetetion', 'moca_fluency','moca_abstraction','moca_delayed_word_recall','moca_orientation']] # drop extra
moca1 = moca.copy(deep = True) 
#Dropping the Duplicated Index (Only 1)
moca = moca[~moca.index.duplicated()].reset_index()
moca = moca[moca.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
moca = moca.unstack()
visualize_unclean_data(moca)
moca = moca[ (moca.isnull().sum(axis = 1) < 15) ]
new_col_list_moca = moca.columns.levels[0]
for a in new_col_list_moca: 
    moca[a] = moca[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
moca.name = 'Montreal Cognitive Assessment'
data_info(moca)

# Functional Assesment Questionaire
cols['faq'] = ['RID', 'VISCODE2', 'FAQTOTAL']
faq = pd.read_csv('ADNI\\Raw_Data\\Assessment\\FAQ.csv', usecols=cols['faq'], index_col = ['RID', 'VISCODE2'])
faq1 = faq.copy(deep = True) 
faq = faq[~faq.index.duplicated()].reset_index()
faq = faq[faq.VISCODE2.isin(['bl','m06','m12'])].set_index(['RID','VISCODE2'])
faq = faq[~faq.index.duplicated()]
#Unstacking 
faq = faq.unstack()
faq = faq[ (faq.isnull().sum(axis = 1) < 2) ]
new_col_list_faq = faq.columns.levels[0]
for a in new_col_list_faq: 
    faq[a] = faq[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
visualize_unclean_data(faq)
faq.name = 'Functional Assessment Questionnaire'
data_info(faq)   

# Diagnosis and Symptoms Checklist
cols['adsxlist'] = ['RID' , 'VISCODE' , 'AXNAUSEA' , 'AXVOMIT' , 'AXDIARRH' , 'AXCONSTP' , 'AXABDOMN', 'AXSWEATN' ,\
 'AXDIZZY' , 'AXENERGY' , 'AXDROWSY' , 'AXVISION' , 'AXHDACHE' , 'AXDRYMTH' , 'AXBREATH' , 'AXCOUGH'\
 , 'AXPALPIT' , 'AXCHEST' , 'AXURNDIS' , 'AXURNFRQ' , 'AXANKLE' , 'AXMUSCLE' , 'AXRASH' , 'AXINSOMN' ,\
 'AXDPMOOD' , 'AXCRYING' , 'AXELMOOD' , 'AXWANDER' , 'AXFALL'   ]
adsxlist = pd.read_csv("ADNI\\Raw_Data\\Assessment\\ADSXLIST.csv",index_col='RID', usecols=cols['adsxlist'])
adsxlist1 = adsxlist.copy(deep = True)
adsxlist = adsxlist.replace({-4:np.NAN})
adsxlist = adsxlist.replace({-1:np.NAN})
adsxlist1['VISCODE2'] = adsxlist1['VISCODE']
adsxlist['VISCODE2'] = adsxlist['VISCODE']
adsxlist['VISCODE'].value_counts()
del adsxlist['VISCODE']
del adsxlist1['VISCODE']
adsxlist = adsxlist[adsxlist['VISCODE2'].isin(['m06','m12','bl']) ]  
adsxlist = adsxlist.reset_index().set_index(['RID','VISCODE2'])
adsxlist = adsxlist[~adsxlist.index.duplicated()].unstack()
visualize_unclean_data(adsxlist)
adsxlist = adsxlist.T[ (adsxlist.T.isnull().sum(axis = 1) <= 500) ].T
adsxlist = adsxlist[ (adsxlist.isnull().sum(axis = 1) <= 20) ]
adsxlist.name = 'Diagnosis and Symptoms Checklist'
data_info(adsxlist)

# Vital Signs
cols['VITALS'] = ['RID', 'VISCODE2', 'VSWEIGHT','VSWTUNIT','VSHEIGHT','VSHTUNIT','VSBPSYS','VSBPDIA','VSPULSE','VSRESP','VSTEMP', 'VSTMPUNT']
vitals = pd.read_csv("ADNI\\Raw_Data\\medical\\VITALS.csv",index_col='RID', usecols=cols['VITALS'])
vitals1 = vitals.copy(deep = True)
vitals = vitals.replace({-4:np.NAN})
vitals = vitals.replace({-1:np.NAN})
vitals = vitals[vitals['VISCODE2'].isin(['m06','m12','bl']) ]  
vitals.loc[ vitals['VSWTUNIT'] == 1 , 'VSWEIGHT' ] = vitals[ vitals['VSWTUNIT'] == 1  ].VSWEIGHT /2.20462
del vitals['VSHTUNIT']
del vitals['VSHEIGHT']
del vitals['VSWTUNIT']
vitals.loc[ vitals['VSTMPUNT'] == 1 , 'VSTEMP' ] = (vitals[ vitals['VSTMPUNT'] == 1  ].VSTEMP - 32 )*0.556
del vitals['VSTMPUNT']
vitals = vitals.reset_index().set_index(['RID','VISCODE2'])
vitals = vitals[~vitals.index.duplicated()].unstack()
visualize_unclean_data(vitals)
vitals = vitals[ (vitals.isnull().sum(axis = 1) <7) ]
new_col_list_vitals = vitals.columns.levels[0]
for a in new_col_list_vitals: 
    vitals[a] = vitals[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
vitals.name = 'Vitals'
data_info(vitals)



'''
Visualizations
'''
# Data available without imputation
a = moca1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'moca'})
c = neurobat_1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'neurobat'})
d = npi1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'npi'})
e = npiq1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'npiq'})
f = mmse1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'mmse'})
g = faq1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'faq'})
h = recbllog1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'recbllog'})
#k = neuroexm1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'neuroexm'})
i = uwn1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'uwn'})
#m = item_new.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'item'})
j = geriatric1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'geriatric'})
k = ecogsp1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'ecogsp'})
l = ecogpt1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'ecogpt'})
m = cdr1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'cdr'})
#r = cci1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'cci'})
#s = adsxlist1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'adsxlist'})
n = adascores1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'adascores'})
#u = backmeds1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'backmeds'})
o = blscheck1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'blscheck'})

plot_data = a.merge(c, on='VISCODE2', how='outer').merge(d, on='VISCODE2', how='outer')\
.merge(e, on='VISCODE2', how='outer').merge(f, on='VISCODE2', how='outer').merge(g, on='VISCODE2', how='outer')\
.merge(h, on='VISCODE2', how='outer').merge(i, on='VISCODE2', how='outer').merge(j, on='VISCODE2', how='outer')\
.merge(k, on='VISCODE2', how='outer').merge(l, on='VISCODE2', how='outer').merge(m, on='VISCODE2', how='outer')\
.merge(n, on='VISCODE2', how='outer').merge(o, on='VISCODE2', how='outer')

plot_data = plot_data[plot_data.VISCODE2.isin(['bl','m06','m12','m18','m36','m48','m24'])]


# Data available after imputation
a1 = moca.stack().reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'moca'})
c1 = neurobat1.stack().reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'neurobat'})
d1 = npi_all.stack().reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'npi'})
e1 = npiq.stack().reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'npiq'})
f1 = mmse.stack().reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'mmse'})
g1 = faq.stack().reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'faq'})
h1 = recbllog2.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'recbllog'})
#k1 = neuroexm.stack().reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'neuroexm'})
i1 = uwn.stack().reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'uwn'})
#m1 = item_f1.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'item'})
j1 = geriatric.stack().reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'geriatric'})
k1 = ecogsp.stack().reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'ecogsp'})
l1 = ecogpt.stack().reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'ecogpt'})
m1 = cdr.stack().reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'cdr'})
#r1 = cci.stack().reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'cci'})
#s1 = adsxlist_imp.reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'adsxlist'})
n1 = adascores.stack().reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'adascores'})
#u1 = backmeds.stack().reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'backmeds'})
o1 = blscheck_adni2.stack().reset_index().groupby('VISCODE2').size().reset_index().rename(columns={0:'blscheck'})

plot_data_imp = a1.merge(c1, on='VISCODE2', how='outer').merge(d1, on='VISCODE2', how='outer')\
.merge(e1, on='VISCODE2', how='outer').merge(f1, on='VISCODE2', how='outer').merge(g1, on='VISCODE2', how='outer')\
.merge(h1, on='VISCODE2', how='outer').merge(i1, on='VISCODE2', how='outer').merge(j1, on='VISCODE2', how='outer')\
.merge(k1, on='VISCODE2', how='outer').merge(l1, on='VISCODE2', how='outer').merge(m1, on='VISCODE2', how='outer')\
.merge(n1, on='VISCODE2', how='outer').merge(o1, on='VISCODE2', how='outer')
plot_data_imp = plot_data_imp[plot_data_imp.VISCODE2.isin(['bl','m06','m12','m18','m36','m48','m24'])]


f1, ax1 = plt.subplots(1)
plot_data.set_index('VISCODE2').plot(kind='bar', title="Number of participants and type of collected data for each visit (before imputation)", figsize=(20, 6), ax = ax1)
patches, labels = ax1.get_legend_handles_labels()
ax1.legend(patches, labels, bbox_to_anchor=(1.1, 1), loc='upper right')
ax1.set_ylabel("Number of Participants");
ax1.set_xlabel("Scheduled Visit ID");
f1.savefig('C:\\Users\\Vipul Satone\\health data\\ADNI\\Processed_data\\plots\\distribution.jpg')
f1.show()


f, (ax1, ax2) = plt.subplots(2)
plot_data = plot_data[plot_data.VISCODE2.isin(['bl','m06','m12'])]
plot_data.set_index('VISCODE2').plot(kind='bar', title="Number of participants and type of collected data for each visit (before imputation)", figsize=(20, 12), ax = ax1, legend=False)
ax1.set_ylabel("Number of Participants");
ax1.set_xlabel("Scheduled Visit ID");

plot_data_imp = plot_data_imp[plot_data_imp.VISCODE2.isin(['bl','m06','m12','m24'])]
p = plot_data_imp.set_index('VISCODE2').plot(kind='bar', title="Number of participants and type of collected data for each visit (after imputation)", figsize=(20, 12), ax = ax2)
patches, labels = ax2.get_legend_handles_labels()
ax2.legend(patches, labels, bbox_to_anchor=(1.1, 1.25), loc='upper right')
ax2.set_ylabel("Number of Participants");
ax2.set_xlabel("Scheduled Visit ID");
ax2.set(ylim=(0, 5000))
#ax2.legend
#f1.savefig('C:\\Users\\Vipul Satone\\health data\\ADNI\\Processed_data\\plots\\distribution2.jpg')
f.show()
 


'''
defining all functions
'''

    #return null_in_cols,null_in_rows
def join_dataframes(item0,item1):
    a = pd.concat([(item0), (item1)], axis = 1 )
    a = a[a.iloc[:,2] == 0]
    return a

def numeric(data):
    for i in range(data.shape[1]):
        try:
            data.iloc[:,i] =  data.iloc[:,i].map( lambda x: pd.to_numeric(x, errors='ignore'))
        except:
            pass
    return data

# function used to normalize
def normalize(Train1,b):
    col_names = list(Train1.columns)
    '''
    if (Train1[col_names[i]] == Train1[col_names[i]][0] ).sum() = len(Train1):
        del Train1[col_names[i]]
    '''
    if (b == 'z'):
        for i in range(Train1.shape[1]):
            Train1[col_names[i]] = (Train1[col_names[i]] - Train1[col_names[i]].mean(skipna = True)) / Train1[col_names[i]].std(skipna = True)
    else:
        for i in range(Train1.shape[1]):
            Train1[col_names[i]] = (Train1[col_names[i]] - min(Train1[col_names[i]]) )/ ( max(Train1[col_names[i]] ) - min(Train1[col_names[i]]) )
    return Train1




def project_data(Max_intersection_dataset, visit):
    patno_filtered_visited = dict_datasets[Max_intersection_dataset[0]]
    #patno_filtered_visited = patno_filtered_visited[patno_filtered_visited.isin(['bl','m06','m12'])]
    for t in range(len(Max_intersection_dataset)):
        patients = dict_datasets[Max_intersection_dataset[t]]
        #patients = patients[ patients.isin(['bl','m06','m12']) ]
        patno_filtered_visited = pd.merge(patno_filtered_visited, patients, left_index = True, right_index = True, how='inner')
        a = patno_filtered_visited.T 
        a = a [a.index.get_level_values(1).isin(['bl','m06','m12'])]
        patno_filtered_visited = a.T
        
    patno_filtered_visited.iloc[:,:] = patno_filtered_visited.iloc[:,:].interpolate(method='linear', axis=1, limit=5, limit_direction='both')
      

    M_chosen = normalize(patno_filtered_visited,'m')



    M_chosen = M_chosen.T[ M_chosen.T.isnull().sum(axis = 1)== 0 ].T

    M_W_columns = ['PCA_1', 'PCA_2', 'PCA_3', 'PCA_2_1', 'PCA_2_2','ICA_1', 'ICA_2', 'NMF_2_1', 'NMF_2_2', 
               'NMF_3_1', 'NMF_3_2', 'NMF_3_3','ICA_3_1', 'ICA_3_2', 'ICA_3_3']
    M_W = pd.DataFrame(index=M_chosen.index, columns=M_W_columns)

    from sklearn.decomposition import PCA as sklearnPCA
    model_pca = sklearnPCA(n_components=3)
    M_W[['PCA_1', 'PCA_2', 'PCA_3']] = model_pca.fit_transform(M_chosen)

    model_pca = sklearnPCA(n_components=2)
    M_W[['PCA_2_1', 'PCA_2_2']] = model_pca.fit_transform(M_chosen)
    
    # ICA
    from sklearn import decomposition
    model_ICA = decomposition.FastICA(n_components=2)
    M_W[['ICA_1', 'ICA_2']] = model_ICA.fit_transform(M_chosen)
    model_ICA = decomposition.FastICA(n_components=3)
    M_W[['ICA_3_1', 'ICA_3_2', 'ICA_3_3']] = model_ICA.fit_transform(M_chosen)


    # NMF
    from sklearn import decomposition
    model_NMF = decomposition.NMF(n_components=2, init='nndsvda', max_iter=200)
    model_NMF3 = decomposition.NMF(n_components=3, init='nndsvda', max_iter=200)
    M_W[['NMF_2_1', 'NMF_2_2']] = model_NMF.fit_transform(M_chosen)
    M_W[['NMF_3_1', 'NMF_3_2', 'NMF_3_3']] = model_NMF3.fit_transform(M_chosen)
    
    '''
    deleting 1 outlier point 
    '''
    
    M_W =M_W.T
    del M_W[4393]
    M_W =M_W.T
    

    redued_data = pd.DataFrame(M_W)

    # plot the dimension reduction color makrked with participants' "categories", and "gender"
    %matplotlib auto

    '''
    dignosis = pd.read_csv("ADNI\\Raw_Data\\Assessment\\DXSUM_PDXCONV_ADNIALL.csv",  usecols= ['RID','DIAGNOSIS','DXCURREN','DXCHANGE','VISCODE2'])

    #dignosis = dignosis.set_index('RID')
    dignosis = dignosis[dignosis['RID'].isin(redued_data.index)]
    dignosis = dignosis.set_index('RID')
    dignosis = dignosis[dignosis['VISCODE2'] == visit]


    redued = redued_data.merge(dignosis, how = 'inner', left_index = True, right_index = True)
    redued['new'] = 0
    #reduced = pd.concat([redued_data, dignosis], axis = 0)
    redued['new'] = redued[['DXCHANGE','DXCURREN','DIAGNOSIS']].sum(skipna = True, numeric_only = True, axis = 1)

    redued = redued[ ~(redued['new'].isnull())]
    #redued.new = redued.new.replace([1,2,3,4,5,6,7,8,9],[1,2,3,2,3,3,1,2,1])
    colors_categories = redued.new.replace([1,2,3,4,5,6,7,8,9], ['red', 'blue', 'green', 'yellow','purple','pink','black','cyan','magenta'])
    
    # new labels
    dignosis = pd.read_csv("ADNI\\Raw_Data\\Assessment\\DXSUM_PDXCONV_ADNIALL.csv",  usecols= ['RID','DXCURREN','VISCODE2'])
    dignosis = dignosis[ ~(dignosis['DXCURREN'].isnull())]
    #dignosis = dignosis.set_index('RID')
    dignosis = dignosis[dignosis['RID'].isin(redued_data.index)]
    dignosis = dignosis.set_index('RID')
    dignosis = dignosis[dignosis['VISCODE2'] == visit]
    dignosis = dignosis[ ~(dignosis['DXCURREN'].isnull())]
    redued = redued_data.merge(dignosis, how = 'inner', left_index = True, right_index = True)
    redued = redued[ ~(redued['DXCURREN'].isnull())]
    #redued.new = redued.new.replace([1,2,3,4,5,6,7,8,9],[1,2,3,2,3,3,1,2,1])
    colors_categories = redued.DXCURREN.replace([1,2,3,4,5,6,7,8,9], ['red', 'blue', 'green', 'yellow','purple','pink','black','cyan','magenta'])
    '''
    dignosis = pd.read_csv("ADNI\\Raw_Data\\Assessment\\dxsum.csv",  usecols= ['RID','DXCHANGE','DXMDUE','DXCONFID','VISCODE'])
    dignosis = dignosis[ ~(dignosis['DXCHANGE'].isnull())]
    dignosis = dignosis[ ~(dignosis['DXMDUE'] == 'MCI due to other etiology')]
    dignosis = dignosis[ ~(dignosis['DXCONFID'] == 'Mildly Confident')]
    dignosis = dignosis[ ~(dignosis['DXCONFID'] == 'Uncertain')]
    dignosis = dignosis[dignosis['RID'].isin(redued_data.index)]
    dignosis = dignosis.set_index('RID')
    dignosis = dignosis[dignosis['VISCODE'] == visit]
    redued = redued_data.merge(dignosis, how = 'inner', left_index = True, right_index = True)
    redued = redued[ ~(redued['DXCHANGE'].isnull())]
    redued.DXCHANGE = redued.DXCHANGE.replace(['Stable: NL to NL', 'Stable: NL','Stable: MCI','Stable: MCI to MCI',\
                                               'Stable: Dementia', 'Stable: Dementia to Dementia',\
                                               'Conversion: NL to MCI','Conversion: MCI to Dementia','Conversion: NL to Dementia',\
                                               'Reversion: MCI to NL', 'Reversion: Dementia to MCI'],[1,1,2,2,3,3,4,5,6,7,8])
    #redued.DXCHANGE = redued.DXCHANGE.replace([1,2,3,4,5,6,7,8],[1,2,3,2,3,3,1,2])
    colors_categories = redued.DXCHANGE.replace([1,2,3,4,5,6,7,8,9], ['red', 'blue', 'green', 'yellow','purple','pink','black','cyan','magenta'])

    return redued, colors_categories, M_chosen,dignosis



def projections(redued,colors_categories,t):
    plt.figure(3, figsize=(18, 24))
    plt.scatter(redued[['PCA_2_1']], redued[['PCA_2_2']], c = colors_categories)
    plt.title('PCA')
    p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc='red')
    p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc='blue')
    p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc='green')
    p4 = plt.Rectangle((0, 0), 0.1, 0.1, fc='yellow')
    p5 = plt.Rectangle((0, 0), 0.1, 0.1, fc='purple')
    p6 = plt.Rectangle((0, 0), 0.1, 0.1, fc='pink')
    p7 = plt.Rectangle((0, 0), 0.1, 0.1, fc='black')
    p8 = plt.Rectangle((0, 0), 0.1, 0.1, fc='cyan')
    p9 = plt.Rectangle((0, 0), 0.1, 0.1, fc='magenta')
    plt.legend((p1, p2, p3, p4, p5, p6, p7,p8,p9), ('Stable: NL', 'Stable: MCI', 'Stable: Dementia', 'Conversion: NL to MCI', 'Conversion: MCI to Dementia', 'Conversion: NL to Dementia', 'Reversion: MCI to NL','Reversion: Dementia to MCI','Reversion: Dementia to NL'), loc='best');
    #plt.savefig('C:\\Users\\Vipul Satone\\health data\\ADNI\\Processed_data\\plots\\PCA_item_prog.jpg')
    plt.show()


    plt.figure(4, figsize=(9, 12))
    plt.scatter(redued[['NMF_2_1']], redued[['NMF_2_2']], c = colors_categories)
    plt.title('NMF')
    p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc='red')
    p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc='blue')
    p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc='green')
    p4 = plt.Rectangle((0, 0), 0.1, 0.1, fc='yellow')
    p5 = plt.Rectangle((0, 0), 0.1, 0.1, fc='purple')
    p6 = plt.Rectangle((0, 0), 0.1, 0.1, fc='pink')
    p7 = plt.Rectangle((0, 0), 0.1, 0.1, fc='black')
    p8 = plt.Rectangle((0, 0), 0.1, 0.1, fc='cyan')
    p9 = plt.Rectangle((0, 0), 0.1, 0.1, fc='magenta')
    plt.legend((p1, p2, p3, p4, p5, p6, p7,p8,p9), ('Stable: NL', 'Stable: MCI', 'Stable: Dementia', 'Conversion: NL to MCI', 'Conversion: MCI to Dementia', 'Conversion: NL to Dementia', 'Reversion: MCI to NL','Reversion: Dementia to MCI','Reversion: Dementia to NL'), loc=1);
    plt.xlim((-.01, 0.7))
    plt.ylim((-.02, 1.0)) 
    plt.title(t)
    filename = 'C:\\Users\\Vipul Satone\\health data\\ADNI\\Processed_data\\plots\\NMF_item_prog_' + t + '_.jpg'
    plt.savefig(filename)
    plt.show()

    plt.figure(5, figsize=(18, 24))
    plt.scatter(redued[['ICA_1']], redued[['ICA_2']], c = colors_categories)
    plt.title('ICA')
    p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc='red')
    p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc='blue')
    p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc='green')
    p4 = plt.Rectangle((0, 0), 0.1, 0.1, fc='yellow')
    p5 = plt.Rectangle((0, 0), 0.1, 0.1, fc='purple')
    p6 = plt.Rectangle((0, 0), 0.1, 0.1, fc='pink')
    p7 = plt.Rectangle((0, 0), 0.1, 0.1, fc='black')
    p8 = plt.Rectangle((0, 0), 0.1, 0.1, fc='cyan')
    p9 = plt.Rectangle((0, 0), 0.1, 0.1, fc='magenta')
    plt.legend((p1, p2, p3, p4, p5, p6, p7,p8,p9), ('Stable: NL', 'Stable: MCI', 'Stable: Dementia', 'Conversion: NL to MCI', 'Conversion: MCI to Dementia', 'Conversion: NL to Dementia', 'Reversion: MCI to NL','Reversion: Dementia to MCI','Reversion: Dementia to NL'), loc=3);
    #plt.savefig('C:\\Users\\Vipul Satone\\health data\\ADNI\\Processed_data\\plots\\ICA_item_prog.jpg')
    plt.show()



def plot_side_by_side(M_mci_dem,M_mci_dem_nmf_proj,Predict, case,algorithm):
    trial = 'trial3'
    Predict1 = Predict.copy(deep = True)
    f9 = plt.figure(figsize=(20,10))
    f9.suptitle(case + "_" +algorithm)
    ax3 = f9.add_subplot(1, 2, 1, projection='3d')
    ax4 = f9.add_subplot(1, 2, 2, projection='3d')
    colors = pd.DataFrame(Predict1.replace([0,1,2,4],['yellow','hotpink','blue','cyan']))
    ax3.scatter(M_mci_dem_nmf_proj[['NMF_3_1']], M_mci_dem_nmf_proj[['NMF_3_2']], M_mci_dem_nmf_proj[['NMF_3_3']], c = colors.iloc[:,0], alpha=0.8)
    ax3.grid(True)
    if (case == 'moca24' or case == 'moca36'):
        ax3.set_xlabel('moca')
        ax3.set_ylabel('cognition')
        ax3.set_zlabel('dementia')
    elif (case == 'item24' or case == 'item36'):
        ax3.set_xlabel('base line and symptoms ')
        ax3.set_ylabel('Memory and mental exam')
        ax3.set_zlabel('Dementia and cognition')
    else:
        ax3.set_xlabel('Memory + Mental state')
        ax3.set_ylabel('cognition + dementia + depression')
        ax3.set_zlabel('only baseline symptoms')
    ax3.set_title('Predicted progression on \n' + case + ' by ' + algorithm + ' .\n (I have only included MCI \n and Dementia observations \n here no Healthy obsevations)')
    ax3.view_init(30, 30)
    p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc= 'yellow')
    p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc='hotpink')
    p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc='blue')
    ax3.legend((p1,p2,p3), ('low MCI','High MCI+Low Dementia','High Dementia'), loc='best');
    M_mci_dem.DXCHANGE = M_mci_dem.DXCHANGE.replace([1,2,3,4,5,6,7,8],[1,2,3,2,3,3,1,2])
    colors_categories = M_mci_dem.DXCHANGE.replace([1,2,3,4,5,6,7,8,9], ['red', 'blue', 'green', 'yellow','purple','pink','black','cyan','magenta'])
    ax4.scatter(M_mci_dem[['NMF_3_1']], M_mci_dem[['NMF_3_2']], M_mci_dem[['NMF_3_3']], c = colors_categories, alpha=0.8)
    p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc='red')
    p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc='blue')
    p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc='green')
    ax4.legend((p1,p2,p3), ('Control','MCI','Dementia'), loc='best');
    ax4.grid(True)
    if (case == 'moca24' or case == 'moca36'):
        ax4.set_xlabel('moca')
        ax4.set_ylabel('cognition')
        ax4.set_zlabel('dementia')
    elif (case == 'item24' or case == 'item36'):
        ax4.set_xlabel('base line and symptoms ')
        ax4.set_ylabel('Memory and mental exam')
        ax4.set_zlabel('Dementia and cognition')
    else:
        ax4.set_xlabel('Memory + Mental state')
        ax4.set_ylabel('cognition + dementia + depression')
        ax4.set_zlabel('only baseline symptoms')
    ax4.set_title('Actual on ' + case + ' by ' + algorithm + ' .')
    ax4.view_init(30, 30)
    # rotate the axes and update
    for angle in range(0, 360,60):
        ax3.view_init(30, angle)
        ax4.view_init(30, angle)
        plt.draw()
        plt.pause(.0001)
        filename = "C:\\Users\\Vipul Satone\\health data\\ADNI\\Processed_data\\" + trial + "\\clustering_plots\\" + case+"\\"+ algorithm + "\\" + case + algorithm +'_'+ str(angle) + ".png"
        #print(filename)
        f9.savefig(filename)
    ax4.view_init(30, 20)
    ax3.view_init(30, 20)


def gif_convert(moca24, gmm):
    trial = 'trial3'
    with imageio.get_writer("C:\\Users\\Vipul Satone\\health data\\ADNI\\Processed_data\\" + trial + "\\clustering_plots\\"+ moca24 + "\\" + gmm +"\\" + moca24 + gmm + "_movie.gif", mode='I') as writer:
        for angle in range(0, 360,60):
            filename = "C:\\Users\\Vipul Satone\\health data\\ADNI\\Processed_data\\" + trial + "\\clustering_plots\\"+ moca24 + "\\" + gmm  + "\\"+ moca24 + gmm +'_'+ str(angle) + ".png"
            image = imageio.imread(filename)
            writer.append_data(image)
            #print(angle)




def PCA_NMF_ICA_projections_3d(redued_item_24,colors_categories_item_24,nmf,pca,ica):
    if (nmf == 'yes'):
        f6, ax = plt.subplots(1)
        ax = plt.axes(projection='3d')
        ax.scatter(redued_item_24[['NMF_3_1']], redued_item_24[['NMF_3_2']], redued_item_24[['NMF_3_3']], c = colors_categories_item_24)
        a = round(redued_item_24[['NMF_3_1']].min())
        b = round(redued_item_24[['NMF_3_1']].max())
        ax.grid(True)
        ax.set_title('NMF3d')
        ax.set_xlabel('axis x')
        ax.set_ylabel('axis y')
        ax.set_zlabel('axis z')
        ax.view_init(30, 30)
        f6.show()

    if (pca == 'yes'):
        f7, ax1 = plt.subplots(1)
        ax1 = plt.axes(projection='3d')
        ax1.scatter(redued_item_24[['PCA_1']], redued_item_24[['PCA_2']], redued_item_24[['PCA_3']], c = colors_categories_item_24)
        ax1.grid(True)
        ax1.set_title('PCA3d')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])
        ax1.set_xlabel('axis x')
        ax1.set_ylabel('axis y')
        ax1.set_zlabel('axis z')
        ax1.view_init(30, 30)
        f7.show()
    if (ica == 'yes'):
        f8, ax2 = plt.subplots(1)
        ax2 = plt.axes(projection='3d')
        ax2.scatter(redued_item_24[['ICA_3_1']], redued_item_24[['ICA_3_2']], redued_item_24[['ICA_3_3']], c = colors_categories_item_24)
        ax2.grid(True)
        ax2.set_title('ICA3d')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_zticklabels([])
        ax2.set_xlabel('axis x')
        ax2.set_ylabel('axis y')
        ax2.set_zlabel('axis z')
        ax2.view_init(30, 30)
        f8.show()


def organize_prediction_no_moca_item(M_mci_dem_nmf_proj,Predict_gmm):
    M_mci_dem_nmf_proj['predicted'] = Predict_gmm
    a = list(pd.unique(Predict_gmm.iloc[:,0]))
    srt = np.empty((len(a),2))
    for i in a:
        srt[i,1] = M_mci_dem_nmf_proj[M_mci_dem_nmf_proj.predicted == i].iloc[:,1].sum() / len(M_mci_dem_nmf_proj[M_mci_dem_nmf_proj.predicted == i])
        srt[i,0] = i
    srt = pd.DataFrame(srt).sort([1])
    Predict_gmm.replace([srt.iloc[0,0],srt.iloc[1,0], srt.iloc[2,0] ],[0, 1, 2], inplace=True)   
    return pd.DataFrame(Predict_gmm)


def organize_prediction_moca(M_mci_dem_nmf_proj,Predict_gmm):
    M_mci_dem_nmf_proj['predicted'] = Predict_gmm
    a = list(pd.unique(Predict_gmm.iloc[:,0]))
    srt = np.empty((len(a),2))
    for i in a:
        a = M_mci_dem_nmf_proj[M_mci_dem_nmf_proj.predicted == i].iloc[:,1].sum() / len(M_mci_dem_nmf_proj[M_mci_dem_nmf_proj.predicted == i])
        b =  M_mci_dem_nmf_proj[M_mci_dem_nmf_proj.predicted == i].iloc[:,2].sum() / len(M_mci_dem_nmf_proj[M_mci_dem_nmf_proj.predicted == i])
        srt[i,1] = a+b
        srt[i,0] = i
    srt = pd.DataFrame(srt).sort([1])
    Predict_gmm.replace([srt.iloc[0,0],srt.iloc[1,0], srt.iloc[2,0] ],[0, 1, 2], inplace=True)   
    return pd.DataFrame(Predict_gmm)


def organize_prediction_item(M_mci_dem_nmf_proj,Predict_gmm):
    M_mci_dem_nmf_proj['predicted'] = Predict_gmm
    a = list(pd.unique(Predict_gmm.iloc[:,0]))
    srt = np.empty((len(a),2))
    for i in a:
        srt[i,1] = M_mci_dem_nmf_proj[M_mci_dem_nmf_proj.predicted == i].iloc[:,2].sum() / len(M_mci_dem_nmf_proj[M_mci_dem_nmf_proj.predicted == i])
        srt[i,0] = i
    srt = pd.DataFrame(srt).sort([1])
    Predict_gmm.replace([srt.iloc[0,0],srt.iloc[1,0], srt.iloc[2,0] ],[0, 1, 2], inplace=True)   
    return pd.DataFrame(Predict_gmm)

# ploting prediction on test data and comparing it with original data
def plot_predicted(M_mci_dem_nmf_proj,Predict,input1, case,algorithm):
    trial = 'trial3'
    f9 = plt.figure(figsize=(20,10))
    f9.suptitle(case + "_" +algorithm)
    ax3 = f9.add_subplot(1, 2, 1, projection='3d')
    ax4 = f9.add_subplot(1, 2, 2, projection='3d')
    Predict = pd.DataFrame(Predict).replace([0,1,2,3],['yellow','hotpink','blue','red'])
    M_mci_dem_nmf_proj1 = M_mci_dem_nmf_proj.copy(deep = True)
    M_mci_dem_nmf_proj2 = M_mci_dem_nmf_proj.copy(deep = True)
    #del M_mci_dem_nmf_proj1['predicted']
    Predict.index = input1.index
    a = pd.merge(M_mci_dem_nmf_proj1, (Predict) , left_index = True, right_index = True, how='inner')
    ax3.scatter(a[['NMF_3_1']], a[['NMF_3_2']], a[['NMF_3_3']], c = a.iloc[:,-1], alpha=0.8)
    ax3.grid(True)
    if (case == 'moca24' or case == 'moca36'):
        ax3.set_xlabel('moca')
        ax3.set_ylabel('cognition')
        ax3.set_zlabel('dementia')
    elif (case == 'item24' or case == 'item36'):
        ax3.set_xlabel('base line and symptoms ')
        ax3.set_ylabel('Memory and mental exam')
        ax3.set_zlabel('Dementia and cognition')
    else:
        ax3.set_xlabel('Memory + Mental state')
        ax3.set_ylabel('cognition + dementia + depression')
        ax3.set_zlabel('only baseline symptoms')
    ax3.set_title('clf predicted')
    ax3.view_init(30, 30)
    p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc= 'yellow')
    p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc='hotpink')
    p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc='blue')
    p4 = plt.Rectangle((0, 0), 0.1, 0.1, fc='red')
    ax3.legend((p1,p2,p3,p4), ('low MCI','High MCI+Low Dementia','High Dementia','control'), loc='best');
    input1 = pd.DataFrame(input1).replace([0,1,2,3],['yellow','hotpink','blue','red'])
    #del M_mci_dem_nmf_proj2['predicted']
    b = pd.merge(M_mci_dem_nmf_proj1, (input1) , left_index = True, right_index = True, how='inner')
    ax4.scatter(b[['NMF_3_1']], b[['NMF_3_2']], b[['NMF_3_3']], c = b.iloc[:,-1], alpha=0.8)
    ax4.grid(True)
    if (case == 'moca24' or case == 'moca36'):
        ax4.set_xlabel('moca')
        ax4.set_ylabel('cognition')
        ax4.set_zlabel('dementia')
    elif (case == 'item24' or case == 'item36'):
        ax4.set_xlabel('base line and symptoms ')
        ax4.set_ylabel('Memory and mental exam')
        ax4.set_zlabel('Dementia and cognition')
    else:
        ax4.set_xlabel('Memory + Mental state')
        ax4.set_ylabel('cognition + dementia + depression')
        ax4.set_zlabel('only baseline symptoms')
    ax4.set_title('clf input')
    ax4.view_init(30, 30)
    p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc= 'yellow')
    p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc='hotpink')
    p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc='blue')
    p4 = plt.Rectangle((0, 0), 0.1, 0.1, fc='red')
    ax4.legend((p1,p2,p3,p4), ('low MCI','High MCI+Low Dementia','High Dementia','control'), loc='best');   
    
'''
function end
'''

# creating a dictionary of all the datasets
datasets_of_interest = ['moca', 'neurobat', 'npi_m12', 'npi_bl','npi_all', 'npiq', 'mmse', \
                        'geriatric', 'ecogsp', 'UWNPSYCHSUM_10_27_17', 'recbllog','vitals','adsxlist',\
                      'ecogpt', 'cdr', 'adascores' ,'blscheck_adni2', 'blscheck_adni1', 'faq' ]
dict_datasets = {}
dict_datasets['moca'] = moca    
dict_datasets['neurobat'] = neurobat1       
dict_datasets['npi_m12'] = npi_m12
dict_datasets['npi_bl'] = npi_bl
dict_datasets['npi_all'] = npi_all
dict_datasets['npiq'] = npiq
dict_datasets['mmse'] = mmse             
dict_datasets['geriatric'] = geriatric   
dict_datasets['ecogsp'] = ecogsp
dict_datasets['UWNPSYCHSUM_10_27_17'] = uwn
dict_datasets['recbllog'] = recbllog
dict_datasets['ecogpt'] = ecogpt
dict_datasets['cdr'] = cdr
dict_datasets['adascores'] = adascores
dict_datasets['blscheck_adni2'] = blscheck_adni2       
dict_datasets['blscheck_adni1'] = blscheck_adni1     
dict_datasets['faq'] = faq 
dict_datasets['vitals'] = vitals 
dict_datasets['adsxlist'] = adsxlist 

size_matrix = pd.DataFrame(np.zeros((len(dict_datasets),2)) )
for r in range(len(dict_datasets) ):
    size_matrix.iloc[r,0] = datasets_of_interest[r]
    size_matrix.iloc[r,1] = len(dict_datasets[datasets_of_interest[r]])

size_matrix.columns = ['dataset','count']
size_matrix =size_matrix.set_index('dataset')
size_matrix = size_matrix.sort_values(by = ['count'],ascending= False )
             
sorted_cols = list(size_matrix.index)
common_rids = pd.DataFrame(np.zeros((len(dict_datasets),len(dict_datasets))))
common_rids.columns = sorted_cols
common_rids.index = sorted_cols

for i in range(len(dict_datasets)):
    for u in range(len(dict_datasets)):
        if (u>=i):
            a = list(dict_datasets[sorted_cols[i]].index)
            b = list(dict_datasets[sorted_cols[u]].index)
        
            common = list(set(a).intersection(b))
            common_rids.iloc[i,u] = len(common)
            
            
# datasets and visits of interest
Max_intersection_dataset_24 = ['moca', 'neurobat','npi_all', 'mmse', \
                        'geriatric', 'ecogsp', 'UWNPSYCHSUM_10_27_17',\
                      'ecogpt', 'cdr' ,'blscheck_adni2', 'faq','vitals' ]
print('************************')
print('Datasets considered :')
print(Max_intersection_dataset_24)
print('************************')

# Making prediction for 'm24' (Dataset considered was till 'm12')
redued_24, colors_categories_24, M_chosen_24, dignosis = project_data(Max_intersection_dataset_24,'m24')
print('************************')
print('Number of observations :')
print(len(redued_24))
print('************************')

projections(redued_24,colors_categories_24,'m24')
M_mci_dem = redued_24
M_mci_dem_nmf = M_mci_dem[['NMF_3_1', 'NMF_3_2','NMF_3_3']]
M_mci_dem_nmf_proj = M_mci_dem_nmf[~(redued_24.DXCHANGE.isin([1]) )]
PCA_NMF_ICA_projections_3d(redued_24,colors_categories_24,'yes','yes','yes')
               

model_gmm = mixture.GaussianMixture(n_components=3, covariance_type='tied')
model_gmm.fit(M_mci_dem_nmf_proj)
Predict_gmm = pd.DataFrame(model_gmm.predict(M_mci_dem_nmf_proj))

Predict_gmm.index = M_mci_dem_nmf_proj.index
Predict_gmm = organize_prediction_moca(M_mci_dem_nmf_proj,Predict_gmm)

plot_side_by_side(M_mci_dem,M_mci_dem_nmf_proj,Predict_gmm, 'prediction_24','gmm')       
gif_convert("item24", "gmm")        
        
nl_data = M_mci_dem_nmf[(redued_24.DXCHANGE.isin([1]) )]
data_prediction_labels = pd.concat([nl_data,M_mci_dem_nmf_proj]).fillna(3)

data_prediction = pd.merge(M_chosen_24,pd.DataFrame(data_prediction_labels['predicted']),left_index = True, right_index = True, how='inner')
visualize_unclean_data(data_prediction)            
            
# predictive modelling            
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest_1", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis", "Logistic Regression"
         , "Random Forest_2"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    RandomForestClassifier(n_estimators = 40)]

data_prediction1 = data_prediction.copy(deep = True)
data_prediction1['predicted'] = data_prediction1['predicted'].replace([3],[3])
del data_prediction1['predicted'] 

Predict_gmm = Predict_gmm.rename(columns={0: "predicted"})
data_prediction1 = pd.merge(data_prediction1,Predict_gmm,left_index = True, right_index = True, how='outer')
data_prediction1 = data_prediction1.fillna(3)


X_train1, X_test1, y_train1, y_test1 = train_test_split(data_prediction1.iloc[:,0:-1], data_prediction1['predicted'], test_size=0.2, random_state=42)
scores1 = []
score_entry = {}
y_pred = {}

# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train1, y_train1)
    score1 = clf.score(X_test1, y_test1)
    y_pred[name] = (clf.predict(X_test1))
    scores1.append(score1)
    score_entry[name] = score1

# figure = plt.figure(figsize=(27, 9))
plt.figure(2, figsize=(8, 4))
imp, names = zip(*sorted(zip(scores1, names)))

plt.barh(range(len(names)), imp, align = 'center')
plt.yticks(range(len(names)), names)

plt.xlabel('Classifier performance')
plt.ylabel('Classifiers')
plt.title('Comparision of different classifiers for prediction \n at m24 considering baseline and m12')
plt.show()            
            
# randomforest performs better so fine tuning it

X_train, X_test, y_train, y_test = train_test_split(data_prediction1.iloc[:,0:-1], data_prediction1['predicted'], test_size=0.2, random_state=42)

'''
# Fine tuning
'''
'low MCI','High MCI+Low Dementia','High Dementia'

# Cross validationon old rf parameters
pipeline = Pipeline([('classifier', RandomForestClassifier())])
scores_old = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=5)
mean_old = scores_old.mean()
std_old = scores_old.std()
print(mean_old)
print(std_old)
print(pipeline.get_params())

grid = {
    'classifier__n_estimators': [5,10,15,20,25,30,35,40,45,50,55],\
    'classifier__max_depth' : [5,10,15,20,25,30,35,40,45,50,55],\
    'classifier__class_weight': [None, 'balanced'],\
    'classifier__max_features': ['auto','sqrt','log2', None],\
    'classifier__random_state' : [0]  
}
grid_search = GridSearchCV(pipeline, param_grid=grid, scoring='accuracy', n_jobs=1, cv=5)
grid_search.fit(X=X_train, y=y_train)

print("-----------")
print(grid_search.best_score_)
print(grid_search.best_params_)            
            
pipeline = Pipeline([('classifier', RandomForestClassifier())])
scores_old = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=5)
mean_old = scores_old.mean()
std_old = scores_old.std()
print(mean_old)
print(std_old)
print(pipeline.get_params())         

'''
# Roc Curve         
'''
#y_train1 = y_train1.replace([1,2,3],['High MCI+Low Dementia', 'High Dementia', 'low MCI + control'])
rf = RandomForestClassifier(max_features=None, n_estimators=45, max_depth= 10,class_weight='balanced', random_state= 0)
rf.fit(X_train1, y_train1)
predictions  = rf.predict_proba(X_test1)

y_true = y_test1 # ground truth labels
y_probas = predictions# predicted probabilities generated by sklearn classifier
skplt.metrics.plot_roc_curve(y_true, y_probas, title= 'm24 3 classes')
plt.text(0.8, 0.6, '0= low | 1= medium |\n2= high | 3 = control |', fontsize=12)
plt.show()

y_predicted = rf.predict(X_test1)
s = y_test1.replace([1,2,0,3],['High MCI+Low Dementia', 'High Dementia', 'low MCI','control'])
c = pd.DataFrame(y_predicted).replace([1,2,0,3],['High MCI+Low Dementia', 'High Dementia', 'low MCI','control'])
skplt.metrics.plot_confusion_matrix(s,c, title = 'm24 4 classes',x_tick_rotation =15)
plt.show()

# new cv accuracy
pipeline1 = Pipeline([('classifier', RandomForestClassifier(max_features=None, n_estimators=45, max_depth= 10,class_weight='balanced', random_state= 0))])
scores_new = cross_val_score(pipeline1, X_train, y_train, scoring='accuracy', cv=5)
mean_new = scores_new.mean()
std_new = scores_new.std()
print(mean_new)
print(std_new)

            
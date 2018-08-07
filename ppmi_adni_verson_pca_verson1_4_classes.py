# -*- coding: utf-8 -*-
"""
Created on Mon May 14 21:52:07 2018

@author: Vipul Satone, Rachneet Kaur
"""
# importing packages
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
from sklearn.metrics import accuracy_score,explained_variance_score
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
ppmi_cols = {}




'''
begin function
following are three important function used to visualize and save proojected space
'''

def project_reduced_3d(reduced_datafame,colors_categories,nmf,pca,ica):
    
    # Description
    #This function will plot data in 3 dim axis and save different views of the plot.
    #input:
    #reduced_datafame - It contains 2d and 3d values of NMF, PCA and ICA.
    #colors_categories - This are the labels, it will help color the projected data based on lables desired (like AD_diseased, AD_non_diseased, PD_diseased, PD_non_diseased)
    #nmf, pca, ica - These are flags which define what needs to be ploted
    #output:
    #Save the diffeeent views of the plot on the hard drive.
    #note: change address to save in different folder
    
    if (nmf == 'yes'):
        f6, ax = plt.subplots(1)
        ax = plt.axes(projection='3d')
        ax.scatter(reduced_datafame[['NMF_3_1']], reduced_datafame[['NMF_3_2']], reduced_datafame[['NMF_3_3']], c = colors_categories)
        ax.grid(True)
        ax.set_title('NMF3d')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel('axis x')
        ax.set_ylabel('axis y')
        ax.set_zlabel('axis z')
        ax.view_init(30, 30)
        f6.savefig('NMF_3D')
        f6.show()

    if (pca == 'yes'):
        f7, ax1 = plt.subplots(1)
        ax1 = plt.axes(projection='3d')
        ax1.scatter(reduced_datafame[['PCA_1']], reduced_datafame[['PCA_2']], reduced_datafame[['PCA_3']], c = colors_categories)
        ax1.grid(True)
        ax1.set_title('PCA3d')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])
        ax1.set_xlabel('axis x')
        ax1.set_ylabel('axis y')
        ax1.set_zlabel('axis z')
        ax1.view_init(30, 30)
        f7.savefig('PCA_3D')
        f7.show()
    
    if (ica == 'yes'):
        f8, ax2 = plt.subplots(1)
        ax2 = plt.axes(projection='3d')
        ax2.scatter(reduced_datafame[['ICA_3_1']], reduced_datafame[['ICA_3_2']], reduced_datafame[['ICA_3_3']], c = colors_categories)
        ax2.grid(True)
        ax2.set_title('ICA3d')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_zticklabels([])
        ax2.set_xlabel('axis x')
        ax2.set_ylabel('axis y')
        ax2.set_zlabel('axis z')
        ax2.view_init(30, 30)
        f8.savefig('ICA_3D')
        f8.show()
        
    for angle in range(0, 360,6):
        # ica
        ax2.view_init(30, angle)
        # pca
        ax1.view_init(30, angle)
        # nmf
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.0001)
        address = "C:\\Users\\Vipul Satone\\health data\\ADNI\\Processed_data\\ppmi_adni\\pca\\gifs\\"
        filename1 = address + "pca" +'_'+ str(angle) + ".png"
        filename2 = address + "ica" +'_'+ str(angle) + ".png"
        filename = address + "nmf" +'_'+ str(angle) + ".png"
        #print(filename)
        f6.savefig(filename)
        f7.savefig(filename1)
        f8.savefig(filename2)


 
def project_reduced_2d(redued,colors_categories):
    
    # Description
    #This function will plot data in 2 dim axis and save different views of the plot.
    #input:
    #reduced_datafame - It contains 2d and 3d values of NMF, PCA and ICA.
    #colors_categories - This are the labels, it will help color the projected data based on lables desired (like AD_diseased, AD_non_diseased, PD_diseased, PD_non_diseased)
    #output:
    #Save the diffeeent views of the plot on the hard drive.
    #note: change address to save in different folder
    
    address = 'C:\\Users\\Vipul Satone\\health data\\ADNI\\Processed_data\\2d plots\\'
    #pca
    plt.figure(figsize=(18, 24))
    plt.scatter(redued[['PCA_2_1']], redued[['PCA_2_2']], c = colors_categories)
    plt.title('PCA')
    p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc='g')
    p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc='r')
    p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc='k')
    p4 = plt.Rectangle((0, 0), 0.1, 0.1, fc='b')
    plt.legend((p1, p2,p3,p4), ('ADNI_control', 'ADNI_patients','PPMI_pd_patients', 'PPMI_pd_healthy'))
    plt.savefig(address +'PCA_2D')
    
    # NMF
    plt.figure(figsize=(9, 12))
    plt.scatter(redued[['NMF_2_1']], redued[['NMF_2_2']], c = colors_categories)
    plt.title('NMF')
    p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc='g')
    p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc='r')
    p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc='k')
    p4 = plt.Rectangle((0, 0), 0.1, 0.1, fc='b')
    plt.legend((p1, p2,p3,p4), ('ADNI_control', 'ADNI_patients','PPMI_pd_patients', 'PPMI_pd_healthy'))
    plt.savefig(address + 'NMF_2D')
    
    #ica
    plt.figure(figsize=(18, 24))
    plt.scatter(redued[['ICA_1']], redued[['ICA_2']], c = colors_categories)
    plt.title('ICA')
    p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc='g')
    p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc='r')
    p3 = plt.Rectangle((0, 0), 0.1, 0.1, fc='k')
    p4 = plt.Rectangle((0, 0), 0.1, 0.1, fc='b')
    plt.legend((p1, p2,p3,p4), ('ADNI_control', 'ADNI_patients','PPMI_pd_patients', 'PPMI_pd_healthy'))
    plt.savefig(address + 'ICA_2D')
    
 
def gif_convert(year):
    # Description
    #This function will generate gifs of 3d plot to give better understanding of distribution.
    #input:
    # year = it can be m48, m24 etc depending on the year for which visualizations are made
    #output:
    #Save the gifs on hard drive.
    #note: change address to save in different folder
    address = "C:\\Users\\Vipul Satone\\health data\\ADNI\\Processed_data\\ppmi_adni\\pca\\gifs\\"
    trial = 'trial3'
    with imageio.get_writer( address + "ica" + '_' + str(year) +  "_movie.gif", mode='I') as writer:
        for angle in range(0, 360,6):
            filename = address + "ica" +'_'+ str(angle) + ".png"
            image = imageio.imread(filename)
            writer.append_data(image)    
    with imageio.get_writer( address + "pca" + '_' + str(year) +  "_movie.gif", mode='I') as writer:
        for angle in range(0, 360,6):
            filename = address + "pca" +'_'+ str(angle) + ".png"
            image = imageio.imread(filename)
            writer.append_data(image)               
    with imageio.get_writer( address + "nmf" + '_' + str(year) +  "_movie.gif", mode='I') as writer:
        for angle in range(0, 360,6):
            filename = address + "nmf" +'_'+ str(angle) + ".png"
            image = imageio.imread(filename)
            writer.append_data(image)   
    
'''
end function
'''







'''
Begin Imputation functions
some functions which will help in impuataions 
'''

# assessment data
cols = {}
ppmi_cols = {}

def data_info(cdr):
#    description 
#    This function will pring some basisc atatics of imputed data to check imputations
#    input: datset
#    output: stats
    print('Name of dataset is: ' + cdr.name) 
    print('\n0th level of columns is ')
    print(list(pd.Series(cdr.columns.get_level_values(0)).unique()) )
    print('\n1st level of columns is: ')
    print(list(pd.Series(cdr.columns.get_level_values(1)).unique()) )
    print('\nShape of datset is:')
    print(cdr.shape)
    print('\nTotal number of missing values: ')
    print(cdr.isnull().sum().sum())



def normalize(Train1,z_norm):
#    Description
#    This function will Normalize the data
#    inputs: 
#    Tran1 - dataset to be normalized
#    z_norm - if 'z' it will to z normalization otherwise min-max normalization 

    col_names = list(Train1.columns)
    '''
    if (Train1[col_names[i]] == Train1[col_names[i]][0] ).sum() = len(Train1):
        del Train1[col_names[i]]
    '''
    if (z_norm == 'z'):
        for i in range(Train1.shape[1]):
            Train1[col_names[i]] = (Train1[col_names[i]] - Train1[col_names[i]].mean(skipna = True)) / Train1[col_names[i]].std(skipna = True)
    else:
        for i in range(Train1.shape[1]):
            Train1[col_names[i]] = (Train1[col_names[i]] - min(Train1[col_names[i]]) )/ ( max(Train1[col_names[i]] ) - min(Train1[col_names[i]]) )
    return Train1


def visualize_unclean_data(Train):
#    Description
#    This function will plot the nans in the data
#    input :
#    Train = dataset
    
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

'''
End Imputation functions
'''

'''
Begin Imputation
'''
# ADNI: Geriatric Depression Scale data imputation
cols['geriatric'] = ['VISCODE2', 'RID', 'GDTOTAL']
geriatric = pd.read_csv("ADNI\\Raw_Data\\Assessment\\GDSCALE.csv", index_col='RID', usecols=cols['geriatric'])
geriatric1 = geriatric.copy(deep = True)
geriatric = geriatric.replace({-4:np.NAN})
geriatric = geriatric.replace({-1:np.NAN})
#len(geriatric)
geriatric = geriatric[geriatric['VISCODE2'].isin(['bl','m12','m06']) ]  
geriatric = geriatric.reset_index().set_index(['RID','VISCODE2'])
geriatric = geriatric[ (geriatric.isnull().sum(axis = 1) ==0) ]
geriatric = geriatric[~geriatric.index.duplicated()].unstack()
geriatric = geriatric[ (geriatric.isnull().sum(axis = 1) ==0) ]
geriatric.name = 'Geriatric depression scale'

#geriatric['adni_or_ppmi','bl'] = 'r'

data_info(geriatric)





# PPMI Geriatric Depression Scale imputation
ppmi_cols['ppmi_geriatric'] = ['PATNO', 'EVENT_ID','GDSSATIS', 'GDSDROPD', 'GDSEMPTY', 'GDSBORED', 'GDSGSPIR', 'GDSAFRAD',\
       'GDSHAPPY', 'GDSHLPLS', 'GDSHOME', 'GDSMEMRY', 'GDSALIVE', 'GDSWRTLS',\
       'GDSENRGY', 'GDSHOPLS', 'GDSBETER']

ppmi_geriatric = pd.read_csv("ADNI\\Raw_Data\\ppmi\\Geriatric_Depression_Scale__Short_.csv", usecols=ppmi_cols['ppmi_geriatric'])
ppmi_geriatric['GDTOTAL'] = ppmi_geriatric.iloc[:,2:].sum(axis=1)
ppmi_geriatric1 = ppmi_geriatric[['PATNO','EVENT_ID','GDTOTAL']]
ppmi_geriatric1.columns = ['RID', 'VISCODE2', 'GDTOTAL']
ppmi_geriatric1['VISCODE2'] = ppmi_geriatric1['VISCODE2'].replace(['BL','V02','V04'],['bl','m06','m12'])
ppmi_geriatric1 = ppmi_geriatric1[(ppmi_geriatric1.VISCODE2.isin(['m06','m12']) )]
ppmi_geriatric1 = ppmi_geriatric1.set_index(['RID', 'VISCODE2']).sort_index(level='RID')
ppmi_geriatric1 = ppmi_geriatric1.unstack()
#ppmi_geriatric1 = ppmi_geriatric1[ (ppmi_geriatric1.isnull().sum(axis = 1) <2) ]
ppmi_geriatric1_cols = ppmi_geriatric1.columns.levels[0]
for a in ppmi_geriatric1_cols: 
    ppmi_geriatric1[a] = ppmi_geriatric1[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')

#ppmi_geriatric1['adni_or_ppmi','bl'] = 'b'



# ADNI: Neurobat Categorical Fluency imputation
cols['neurobat_cat_flu'] = ['RID', 'VISCODE2','CATANIMSC']
neurobat_cat_flu = pd.read_csv('ADNI\\Raw_Data\\Assessment\\NEUROBAT.csv', usecols=cols['neurobat_cat_flu'], index_col = ['RID', 'VISCODE2'])
neurobat_cat_flu1 = neurobat_cat_flu.copy(deep = True) 
neurobat_cat_flu = neurobat_cat_flu[~neurobat_cat_flu.index.duplicated()].reset_index()
neurobat_cat_flu = neurobat_cat_flu[neurobat_cat_flu.VISCODE2.isin(['bl','m12'])].set_index(['RID','VISCODE2'])
neurobat_cat_flu = neurobat_cat_flu.replace({-1: np.NAN})
neurobat_cat_flu = neurobat_cat_flu.unstack()
#len(neurobat_cat_flu)
#neurobat_cat_flu = neurobat_cat_flu[ (neurobat_cat_flu.isnull().sum(axis = 1) <1) ]
neurobat_cat_flu_cols = neurobat_cat_flu.columns.levels[0]
for a in neurobat_cat_flu_cols: 
    neurobat_cat_flu[a] = neurobat_cat_flu[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
#neurobat_cat_flu['adni_or_ppmi','bl'] = 'r'

# PPMI semantic_flu imputation
ppmi_cols['semantic_flu']= ['PATNO', 'EVENT_ID', 'VLTANIM']
ppmi_semantic = pd.read_csv('ADNI\\Raw_Data\\ppmi\\Semantic_Fluency.csv', usecols=ppmi_cols['semantic_flu'])
ppmi_semantic.columns = ['PATNO', 'EVENT_ID', 'CATANIMSC']
ppmi_semantic = ppmi_semantic.set_index(['PATNO'])
ppmi_semantic['EVENT_ID'] = ppmi_semantic['EVENT_ID'].replace(['BL','V02','V04'],['bl','m06','m12'])
ppmi_semantic = ppmi_semantic[(ppmi_semantic.EVENT_ID.isin(['bl','m06','m12']) )]
ppmi_semantic = ppmi_semantic.reset_index().set_index(['PATNO', 'EVENT_ID'])
ppmi_semantic = ppmi_semantic.unstack()
#len(ppmi_semantic)
#ppmi_semantic = ppmi_semantic[ (ppmi_semantic.isnull().sum(axis = 1) <1) ]
ppmi_semantic_cols = ppmi_semantic.columns.levels[0]
for a in ppmi_semantic_cols: 
    ppmi_semantic[a] = ppmi_semantic[a].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
#ppmi_semantic['adni_or_ppmi','bl'] = 'b'


'''
Code for Moca:
'''
# ADNI Montreal Cognitive Assessment imputation 
cols['moca'] = ['RID','VISCODE2', 'TRAILS', 'CUBE', 'CLOCKCON', 'CLOCKNO', 'CLOCKHAN','LION', 'RHINO', 'CAMEL', 'IMMT1W1', 'IMMT1W2', 'IMMT1W3', 'IMMT1W4',
       'IMMT1W5', 'IMMT2W1', 'IMMT2W2', 'IMMT2W3', 'IMMT2W4', 'IMMT2W5','DIGFOR', 'DIGBACK', 'LETTERS', 'SERIAL1', 'SERIAL2', 'SERIAL3',
       'SERIAL4', 'SERIAL5', 'REPEAT1', 'REPEAT2', 'FFLUENCY', 'ABSTRAN','ABSMEAS', 'DELW1', 'DELW2', 'DELW3', 'DELW4', 'DELW5', 'DATE', 'MONTH',
       'YEAR', 'DAY', 'PLACE', 'CITY']
moca_adni = pd.read_csv('ADNI\\Raw_Data\\Assessment\\MOCA.csv', usecols=cols['moca'], index_col = ['RID','VISCODE2'])

cols['moca_trail_making'] = ['TRAILS']
moca_adni['moca_trail_making'] = moca_adni[cols['moca_trail_making']].sum(axis=1)

cols['moca_visuosoconstructional'] = ['CUBE', 'CLOCKCON', 'CLOCKNO', 'CLOCKHAN']
moca_adni['moca_visuosoconstructional'] = moca_adni[cols['moca_visuosoconstructional']].sum(axis=1)

cols['moca_naming'] = [ 'LION', 'RHINO', 'CAMEL']
moca_adni['moca_naming'] = moca_adni[cols['moca_naming']].sum(axis = 1)

map_ppmi = {5:3, 4:3, 3:2, 2:2, 1:1, 0:0 }
moca_adni['SUM_SERIAL'] = moca_adni[['SERIAL1', 'SERIAL2', 'SERIAL3','SERIAL4', 'SERIAL5']].sum(axis = 1).map(map_ppmi)
moca_adni['LETTERS_SCORE'] = moca_adni['LETTERS']<2
moca_adni['LETTERS_SCORE'] = moca_adni['LETTERS_SCORE']*1
cols['moca_attention'] = [ 'DIGFOR', 'DIGBACK', 'LETTERS_SCORE', 'SUM_SERIAL']
moca_adni['moca_attention'] = moca_adni[cols['moca_attention']].sum(axis=1)

moca_adni['FLUENCY'] = moca_adni['FFLUENCY']>=11
moca_adni['FLUENCY'] = moca_adni['FLUENCY']*1
moca_adni['REPEAT'] = moca_adni[['REPEAT1','REPEAT2']].sum(axis=1)         
cols['moca_language'] = ['FLUENCY', 'REPEAT' ]
moca_adni['moca_language'] = moca_adni[cols['moca_language']].sum(axis=1)
     
moca_adni['moca_abstraction'] = moca_adni[['ABSTRAN','ABSMEAS']].sum(axis=1)    

map_recall_ppmi = {1:1, 2:0, 3:0, 0:0 }
delayed = [ 'DELW1', 'DELW2', 'DELW3', 'DELW4', 'DELW5']
for col in delayed:
    moca_adni[col] = moca_adni[col].map(map_recall_ppmi)
moca_adni['moca_delayed_recall'] = moca_adni[delayed].sum(axis = 1)

cols['moca_orientation'] = [ 'DATE', 'MONTH', 'YEAR', 'DAY', 'PLACE', 'CITY' ]
moca_adni['moca_orientation'] = moca_adni[cols['moca_orientation']].sum(axis=1)

moca_adni = moca_adni[['moca_trail_making', 'moca_visuosoconstructional', 'moca_naming', 'moca_attention', 'moca_language','moca_abstraction','moca_delayed_recall','moca_orientation']] # drop extra
moca_adni1 = moca_adni.copy(deep = True) 
#Dropping the Duplicated Index (Only 1)
moca_adni = moca_adni[~moca_adni.index.duplicated()].reset_index()
moca_adni = moca_adni[moca_adni.VISCODE2.isin(['bl','m12','m24'])].set_index(['RID','VISCODE2'])
moca_adni = moca_adni.unstack()
#len(moca_adni)

moca_adni_list = [ 'moca_visuosoconstructional', 'moca_naming', 'moca_attention', 'moca_language', 'moca_abstraction','moca_delayed_recall','moca_orientation']
s1 = moca_adni[ (moca_adni.loc[:, 'moca_trail_making'].isnull().sum(axis = 1) ) <= 1]
a1 = s1.loc[:, 'moca_trail_making'].interpolate(method='linear', axis=1, limit=1, limit_direction='both')


a1.columns = pd.MultiIndex.from_product(['moca_trail_making', list(a1.columns)])

for i in range(len(moca_adni_list)):
    b1 = s1.loc[:, moca_adni_list[i]].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
    b1.columns = pd.MultiIndex.from_product([moca_adni_list[i], list(b1.columns)])
    a1 = pd.merge(a1, b1, left_index = True, right_index = True, how='inner')
moca_adni= a1



# PPMI Montreal Cognitive Assessment  
ppmi_cols['moca'] = ['PATNO', 'EVENT_ID', "MCAALTTM", "MCACUBE", "MCACLCKC", "MCACLCKN", "MCACLCKH", "MCALION", "MCARHINO", "MCACAMEL", "MCAFDS", "MCABDS", "MCAVIGIL", "MCASER7", "MCASNTNC", "MCAVFNUM", "MCAVF", "MCAABSTR", "MCAREC1", "MCAREC2", "MCAREC3", "MCAREC4", "MCAREC5", "MCADATE", "MCAMONTH", "MCAYR", "MCADAY", "MCAPLACE", "MCACITY", "MCATOT"]
moca_ppmi = pd.read_csv('ADNI\\Raw_Data\\ppmi\\Montreal_Cognitive_Assessment__MoCA_.csv', usecols=ppmi_cols['moca'], index_col = ['PATNO'])

#Mapping to the visit codes of ADNI
map_visitcode = {'SC': 'sc', 'BL': 'bl', 'V01': 'm03', 'V02': 'm06', 'V03': 'm09', 'V04': 'm12', 'V05': 'm18', 'V06': 'm24', 'V07': 'm30', 'V08': 'm36', 'V09': 'm42', 'V10': 'm48', 'V11': 'm54', 'V12': 'm60'}
moca_ppmi['VISIT_CODE'] = moca_ppmi['EVENT_ID'].map(map_visitcode)

ppmi_cols['moca_trail_making'] = ["MCAALTTM"]
moca_ppmi['moca_trail_making'] = moca_ppmi[ppmi_cols['moca_trail_making']].sum(axis=1)

ppmi_cols["moca_visuosoconstructional"] = [ "MCACUBE", "MCACLCKC", "MCACLCKN", "MCACLCKH"]
moca_ppmi['moca_visuosoconstructional'] = moca_ppmi[ppmi_cols['moca_visuosoconstructional']].sum(axis=1)

ppmi_cols["moca_naming"] = [ "MCALION", "MCARHINO", "MCACAMEL"]
moca_ppmi['moca_naming'] = moca_ppmi[ppmi_cols['moca_naming']].sum(axis = 1)

ppmi_cols["moca_attention"] = [ "MCAFDS", "MCABDS", "MCAVIGIL", "MCASER7"]
moca_ppmi['moca_attention'] = moca_ppmi[ppmi_cols['moca_attention']].sum(axis=1)

ppmi_cols["moca_language"] = [ "MCASNTNC", "MCAVF"]
moca_ppmi['moca_language'] = moca_ppmi[ppmi_cols['moca_language']].sum(axis=1)

moca_ppmi['moca_abstraction'] = moca_ppmi['MCAABSTR']

ppmi_cols["moca_delayed_recall"] = [ "MCAREC1", "MCAREC2", "MCAREC3", "MCAREC4", "MCAREC5"]
moca_ppmi['moca_delayed_recall'] = moca_ppmi[ppmi_cols['moca_delayed_recall']].sum(axis=1)

ppmi_cols["moca_orientation"] = [ "MCADATE", "MCAMONTH", "MCAYR", "MCADAY", "MCAPLACE", "MCACITY"]
moca_ppmi['moca_orientation'] = moca_ppmi[ppmi_cols['moca_orientation']].sum(axis=1)

moca_ppmi = moca_ppmi[['VISIT_CODE', 'moca_trail_making', 'moca_visuosoconstructional', 'moca_naming', 'moca_attention', 'moca_language', 'moca_abstraction','moca_delayed_recall','moca_orientation']] # drop extra
moca_ppmi1 = moca_ppmi.copy(deep = True) 
#Dropping the Duplicated Index if any
moca_ppmi = moca_ppmi.reset_index()
visualize_unclean_data(moca_ppmi)
moca_ppmi = moca_ppmi[moca_ppmi['VISIT_CODE'].isin(['bl', 'm06','m12','m24'])].set_index(['PATNO','VISIT_CODE'])
moca_ppmi = moca_ppmi.unstack()

moca_ppmi_list = [ 'moca_visuosoconstructional', 'moca_naming', 'moca_attention', 'moca_language', 'moca_abstraction','moca_delayed_recall','moca_orientation']
s = moca_ppmi[ (moca_ppmi.loc[:, 'moca_trail_making'].isnull().sum(axis = 1) ) <= 1]
a = s.loc[:, 'moca_trail_making'].interpolate(method='linear', axis=1, limit=1, limit_direction='both')


a.columns = pd.MultiIndex.from_product(['moca_trail_making', list(a.columns)])

for i in range(len(moca_ppmi_list)):
    b = s.loc[:, moca_ppmi_list[i]].interpolate(method='linear', axis=1, limit=1, limit_direction='both')
    b.columns = pd.MultiIndex.from_product([moca_ppmi_list[i], list(b.columns)])
    a = pd.merge(a, b, left_index = True, right_index = True, how='inner')
moca_ppmi = a


# =============================================================================
# Following are the names of imputated datasets 
# PPMI:
# moca_ppmi
# ppmi_semantic
# ppmi_geriatric1
# 
# ADNI:
# moca_adni
# neurobat_cat_flu
# geriatric
# =============================================================================

# mearging the ADNI datsets and saving them
adni_datasets = {}
adni_list = ['moca_adni', 'neurobat_cat_flu','geriatric']
adni_datasets['moca_adni'] = moca_adni   
adni_datasets['neurobat_cat_flu'] = neurobat_cat_flu   
adni_datasets['geriatric'] = geriatric   
adni_patno_filtered_visited = adni_datasets[adni_list[0]]
for t in range(len(adni_list)-1):
    patients = adni_datasets[adni_list[t+1]]
    adni_patno_filtered_visited = pd.merge(adni_patno_filtered_visited, patients, left_index = True, right_index = True, how='inner')
    a = adni_patno_filtered_visited.T 
    a = a [a.index.get_level_values(1).isin(['bl','m06','m12','m24'])]
    adni_patno_filtered_visited = a.T
    
#Getting labels for ADNI Dataset
# here 0 refers to low || 1 to medium || 2 to high  || and 3 is control

#Predict_gmm = pd.read_csv('C:\\Users\\Vipul Satone\\health data\\ADNI\\Processed_data\\ppmi_adni\\Predict_gmm.csv',index_col = ['RID'])
adni_labels = pd.read_csv('C:\\Users\\Vipul Satone\\health data\\ADNI\\Processed_data\\ppmi_adni\\data_labels_with_controls48.csv',index_col = ['RID'])
adni_labels =adni_labels.iloc[:,-1].reset_index()
#(Predict1.replace([0,1,2,4],['yellow','hotpink','blue','cyan'])   
adni_labels = adni_labels[adni_labels['RID'].isin(adni_patno_filtered_visited.index)]
adni_labels = adni_labels.set_index('RID')
adni_labels.columns = ['adni_or_ppmi']
adni_patno_filtered_visited = pd.merge(adni_patno_filtered_visited, adni_labels, left_index = True, right_index = True, how='inner')
adni_patno_filtered_visited['adni_or_ppmi'] =  adni_patno_filtered_visited['adni_or_ppmi'].replace([0,1,2,3],['r','r','r','g'])
adni_patno_filtered_visited['adni_or_ppmi','bl'] =adni_patno_filtered_visited['adni_or_ppmi']
del adni_patno_filtered_visited['adni_or_ppmi']



# mearging the PPMI datsets and saving them
ppmi_datasets = {}
ppmi_list = ['moca_ppmi', 'ppmi_semantic','ppmi_geriatric1']
ppmi_datasets['moca_ppmi'] = moca_ppmi   
ppmi_datasets['ppmi_semantic'] = ppmi_semantic   
ppmi_datasets['ppmi_geriatric1'] = ppmi_geriatric1   
ppmi_patno_filtered_visited = ppmi_datasets[ppmi_list[0]]
for t in range(len(ppmi_list)-1):
    patients = ppmi_datasets[ppmi_list[t+1]]
    ppmi_patno_filtered_visited = pd.merge(ppmi_patno_filtered_visited, patients, left_index = True, right_index = True, how='inner')
    a = ppmi_patno_filtered_visited.T 
    a = a [a.index.get_level_values(1).isin(['bl','m06','m12','m24'])]
    ppmi_patno_filtered_visited = a.T
visualize_unclean_data(ppmi_patno_filtered_visited)
ppmi_patno_filtered_visited.index = ppmi_patno_filtered_visited.index+100000



#Getting labels for ADNI Dataset
# here red means control || green MCi || Yellow Dementia
ppmi_cols["status"] = ['PATNO', 'ENROLL_CAT']
dignosis_ppmi = pd.read_csv("ADNI\\Raw_Data\\ppmi\\Patient_Status.csv", index_col=["PATNO"], usecols=ppmi_cols["status"])
dignosis_ppmi.index = dignosis_ppmi.index+100000
dignosis_ppmi = dignosis_ppmi[ ~(dignosis_ppmi['ENROLL_CAT'].isnull())]
dignosis_ppmi = dignosis_ppmi[dignosis_ppmi['ENROLL_CAT'].isin(['HC', 'PD'])]
dignosis_ppmi = dignosis_ppmi[dignosis_ppmi.index.isin(ppmi_patno_filtered_visited.index)]
ppmi_patno_filtered_visited = ppmi_patno_filtered_visited.merge(dignosis_ppmi, how = 'inner', left_index = True, right_index = True)

ppmi_patno_filtered_visited.ENROLL_CAT = ppmi_patno_filtered_visited.ENROLL_CAT.replace(['HC', 'PD'],[4, 5])

ppmi_patno_filtered_visited['DXCHANGE',''] = ppmi_patno_filtered_visited['ENROLL_CAT']
del ppmi_patno_filtered_visited['ENROLL_CAT']
#del ppmi_patno_filtered_visited['adni_or_ppmi', 'bl']
ppmi_patno_filtered_visited['DXCHANGE',''] = ppmi_patno_filtered_visited['DXCHANGE',''].replace([4, 5],['b', 'k'])
ppmi_patno_filtered_visited['adni_or_ppmi', 'bl'] = ppmi_patno_filtered_visited['DXCHANGE','']
del ppmi_patno_filtered_visited['DXCHANGE','']

# Joining 2 datasets
M_chosen = pd.concat([adni_patno_filtered_visited, ppmi_patno_filtered_visited])
visualize_unclean_data(M_chosen)
M_chosen = M_chosen.T[ (M_chosen.T.isnull().sum(axis = 1) ) <=150 ].T
labels = M_chosen['adni_or_ppmi', 'bl']
del M_chosen['adni_or_ppmi', 'bl']
M_chosen = normalize(M_chosen,'m')

# creating a new dataframe to store vales of pca, nmf and ica
M_W_columns = ['PCA_1', 'PCA_2', 'PCA_3', 'PCA_2_1', 'PCA_2_2','ICA_1', 'ICA_2', 'NMF_2_1', 'NMF_2_2', 
               'NMF_3_1', 'NMF_3_2', 'NMF_3_3','ICA_3_1', 'ICA_3_2', 'ICA_3_3']
M_W = pd.DataFrame(index=M_chosen.index, columns=M_W_columns)
#M_W_select = M_W.loc[ M_cat.ENROLL_CAT.isin(['HC', 'PD']) ]
#plt.scatter(M_W_select[['NMF_2_1']], M_W_select[['NMF_2_2']], c = colors_categories[ M_cat.ENROLL_CAT.isin(['HC', 'PD']) ])
#M_cat = pd.concat([M_chosen, data_visits["info"].ENROLL_CAT], axis=1) # labels of selected subjects


'''
Data imputation done
'''

'''
Begin - projecting data on reduced space
'''
from sklearn.decomposition import PCA as sklearnPCA
model_pca3 = sklearnPCA(n_components=3)
M_W[['PCA_1', 'PCA_2', 'PCA_3']] = model_pca3.fit_transform(M_chosen)
model_pca = sklearnPCA(n_components=2)
M_W[['PCA_2_1', 'PCA_2_2']] = model_pca.fit_transform(M_chosen)

# ICA
from sklearn import decomposition
model_ICA = decomposition.FastICA(n_components=2)
M_W[['ICA_1', 'ICA_2']] = model_ICA.fit_transform(M_chosen)
model_ICA3 = decomposition.FastICA(n_components=3)
M_W[['ICA_3_1', 'ICA_3_2', 'ICA_3_3']] = model_ICA3.fit_transform(M_chosen)

# NMF
from sklearn import decomposition
model_NMF = decomposition.NMF(n_components=2, init='nndsvda', max_iter=200)
model_NMF3 = decomposition.NMF(n_components=3, init='nndsvda', max_iter=200)
M_W[['NMF_2_1', 'NMF_2_2']] = model_NMF.fit_transform(M_chosen)
M_W[['NMF_3_1', 'NMF_3_2', 'NMF_3_3']] = model_NMF3.fit_transform(M_chosen)
reduced_data = pd.DataFrame(M_W)
reduced = pd.concat([reduced_data, labels], axis = 1)
#reduced['adni_or_ppmi', 'bl'] = reduced['adni_or_ppmi', 'bl'].replace(['r', 'b'],[1,2])
colors = colors_categories = reduced['adni_or_ppmi', 'bl']
#.replace([1,2,], ['red', 'blue'])

H = model_NMF.components_
#print ('H is', H)
H_columns = M_chosen.columns
#print ('H_columns is', H_columns)
M_H = pd.DataFrame(data=H, columns=H_columns)
M_H.loc[0] = H[0,:]
M_H.loc[1] = H[1,:]
#M_H.loc[2] = H[2,:]
#M_H = M_H.iloc[:, M_H.columns.get_level_values(1)=='m12'] 
M_H_T = M_H.T.sort_values(by=[1],ascending=False)
cg = sns.clustermap(data=M_H_T.fillna(0), col_cluster=False, figsize=(30, 30), standard_scale=3)
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=20)
#plt.savefig("C:\\Users\\Vipul Satone\\health data\ADNI\\Processed_data\\ppmi_adni\\axis\\2axis.png")
cg_list = list(cg.dendrogram_row.reordered_ind)
custom_dict_sc  = pd.DataFrame(np.zeros((len(M_H_T),1)))
list_col_M_H_T = list(M_H_T.index.get_level_values(0))
#for i in range(len(M_H_T)) :
#    custom_dict_sc.iloc[i,0] = list_col_M_H_T[cg_list[i]]
#custom_dict_sc.to_csv("C:\\Users\\Vipul Satone\\health data\ADNI\\Processed_data\\ppmi_adni\\axis\\custom_dict_sc_2d.csv")


H3 = model_NMF3.components_
#print ('H is', H)
H_columns3 = M_chosen.columns
#print ('H_columns is', H_columns)
M_H3 = pd.DataFrame(data=H3, columns=H_columns3)
M_H3.loc[0] = H3[0,:]
M_H3.loc[1] = H3[1,:]
M_H3.loc[2] = H3[2,:]
#M_H = M_H.iloc[:, M_H.columns.get_level_values(1)=='m12'] 
M_H_T3 = M_H3.T.sort_values(by=[2],ascending=False)
cg3 = sns.clustermap(data=M_H_T3.fillna(0), col_cluster=False, figsize=(30, 40), standard_scale=3)
plt.setp(cg3.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=20)
#plt.savefig("C:\\Users\\Vipul Satone\\health data\ADNI\\Processed_data\\ppmi_adni\\axis\\axis3.png")
cg_list3 = list(cg3.dendrogram_row.reordered_ind)
custom_dict_sc3  = pd.DataFrame(np.zeros((len(M_H_T3),2)))
list_col_M_H_T3 = list(M_H_T3.index.get_level_values(0))
#for i in range(len(M_H_T3)) :
#    custom_dict_sc3.iloc[i,0] = list_col_M_H_T3[cg_list3[i]]
#custom_dict_sc3.to_csv("C:\\Users\\Vipul Satone\\health data\ADNI\\Processed_data\\ppmi_adni\\axis\\custom_dict_sc_3d.csv")




PCA_NMF_ICA_projections_3d(reduced, colors, 'yes', 'yes', 'yes')
project_reduced_2d(reduced,colors)
gif_convert(48)
'''
End - projecting data on reduced space
'''

#pd.DataFrame(model_pca3.components_,columns=M_chosen.columns,index = ['PC-1','PC-2','PC-3'])
'''
weight3 = pd.DataFrame(model_pca3.components_,columns=M_chosen.columns,index = ['PC-1','PC-2','PC-3'])
model_pca3.explained_variance_  
weight2 = pd.DataFrame(model_pca.components_,columns=M_chosen.columns,index = ['PC-1','PC-2'])
model_pca.explained_variance_ 
weight2 = abs(weight2)
weight3 = abs(weight3)
weight3 = weight3 / weight3.sum(axis = 0)
weight2 = weight2 / weight2.sum(axis = 0)
weight2.to_csv("C:\\Users\\Vipul Satone\\health data\ADNI\\Processed_data\\ppmi_adni\\pca\\pca_2d.csv")
weight3.to_csv("C:\\Users\\Vipul Satone\\health data\ADNI\\Processed_data\\ppmi_adni\\pca\\pca_3d.csv")
'''

'''
Begin - axis lables and finding explained variance
'''
weight3 = pd.DataFrame(model_pca3.components_,columns=M_chosen.columns,index = ['PC-1','PC-2','PC-3'])
print('Explained Variance in 3D ')
print(model_pca3.explained_variance_)  
weight2 = pd.DataFrame(model_pca.components_,columns=M_chosen.columns,index = ['PC-1','PC-2'])
model_pca.explained_variance_ 
weight2 = abs(weight2)
weight3 = abs(weight3)
weight3 = (weight3 / weight3.sum(axis = 0)).T
weight2 = (weight2 / weight2.sum(axis = 0)).T

weight2['axis']  = 0
for i in range((weight2.shape[0])):
    if ( (weight2.iloc[i,0] - weight2.iloc[i,1]) > 0.1 ):
        weight2.iloc[i,2] = 'axis 1'
    elif( (weight2.iloc[i,1] - weight2.iloc[i,0]) > 0.1 ):
        weight2.iloc[i,2]  = 'axis 2'
    else:
        weight2.iloc[i,2] = 'both'


weight3['axis']  = 0
for i in range((weight3.shape[0])):
    if ( (weight3.iloc[i,0] - ( 0.75 * (weight3.iloc[i,2] + weight3.iloc[i,1]) ) ) > 0.01 ):
        weight3.iloc[i,3] = 'axis 1'
    elif( (weight3.iloc[i,1] - ( 0.75 * (weight3.iloc[i,0] + weight3.iloc[i,2]) ) ) > 0.01 ):
        weight3.iloc[i,3]  = 'axis 2'
    elif( (weight3.iloc[i,2] - ( 0.75 * (weight3.iloc[i,0] + weight3.iloc[i,1]) ) ) > 0.01 ):
        weight3.iloc[i,3] = 'axis 3'
    else:
        weight3.iloc[i,3] = 'ambigious'

'''
End - axis lables
'''


# =============================================================================
# 
# weight2.to_csv("C:\\Users\\Vipul Satone\\health data\ADNI\\Processed_data\\ppmi_adni\\pca\\pca_2d.csv")
# weight3.to_csv("C:\\Users\\Vipul Satone\\health data\ADNI\\Processed_data\\ppmi_adni\\pca\\pca_3d.csv")
# 
# 
# =============================================================================

'''
color coding the lables
'''
data_prediction = pd.merge(M_chosen,pd.DataFrame(reduced['adni_or_ppmi', 'bl']),left_index = True, right_index = True, how='inner')
visualize_unclean_data(data_prediction)      
data_prediction['predicted'] = data_prediction['adni_or_ppmi', 'bl']
del data_prediction['adni_or_ppmi', 'bl']
data_prediction1 = data_prediction.copy(deep = True)
data_prediction1['predicted'] = data_prediction1['predicted'].replace(['r','g','b','k'],[1,2,3,4])
      
'''
Making predictions
'''
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


X_train1, X_test1, y_train1, y_test1 = train_test_split(data_prediction1.iloc[:,0:-1], data_prediction1['predicted'], test_size=0.2, random_state=42)
scores1 = []
score_entry = {}
y_pred = {}


# iterate over classifiers
for name, clf in zip(names, classifiers):
    pipeline = Pipeline([('classifier', clf)])
    scores_old = cross_val_score(pipeline, X_train1, y_train1, scoring='accuracy', cv=5)
    score1 = scores_old.mean()
    clf.fit(X_train1, y_train1)
    scores1.append(score1)
    score_entry[name] = score1
'''

# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train1, y_train1)
    score1 = clf.score(X_test1, y_test1)
    y_pred[name] = (clf.predict(X_test1))
    scores1.append(score1)
    score_entry[name] = score1
'''    
    
    
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
Fine tuning RF as it performns best
'''

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

'''
0.651006711409
{'classifier__class_weight': None, 'classifier__max_depth': 5, 'classifier__random_state': 0,
 'classifier__n_estimators': 10, 'classifier__max_features': 'log2'}
'''

pipeline = Pipeline([('classifier', RandomForestClassifier(n_estimators = 10, max_depth=5 ,class_weight=None,random_state =0,max_features='log2'))])
scores_old = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=5)
mean_old = scores_old.mean()
std_old = scores_old.std()
print(mean_old)
print(std_old)
print(pipeline.get_params())      

#y_train1 = y_train1.replace([1,2,3],['High MCI+Low Dementia', 'High Dementia', 'low MCI + control'])
rf = RandomForestClassifier(n_estimators = 10, max_depth=5 ,class_weight=None,random_state =0,max_features='log2')
rf.fit(X_train, y_train)
predictions  = rf.predict_proba(X_test)

'''
Generating roc curve
'''
y_true = y_test # ground truth labels
y_probas = predictions# predicted probabilities generated by sklearn classifier
skplt.metrics.plot_roc_curve(y_true, y_probas, title= 'm24 4 classes')
plt.text(0.8, 0.6, '1 = low | 2 = medium |\n3= high | 4 = control |', fontsize=12)
plt.show()

'''
Generate confusion matrix
'''
y_predicted = rf.predict(X_test1)
s = y_test1#.replace([1,2,0,3],['High MCI+Low Dementia', 'High Dementia', 'low MCI','control'])
c = pd.DataFrame(y_predicted)#.replace([1,2,0,3],['High MCI+Low Dementia', 'High Dementia', 'low MCI','control'])
skplt.metrics.plot_confusion_matrix(s,c, title = 'm24 4 classes',x_tick_rotation =15)
plt.show()


'''
.replace(['r','g','b','k'],[1,2,3,4])
k    329 - pd patienst
b    170 - pd healthy
r    141 - ADNI patients
g    106 - ADNI controls
'''

'''
# Fine tuning log reg
'''

# Cross validationon old rf parameters
pipeline = Pipeline([('classifier', LogisticRegression())])
scores_old = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=5)
mean_old = scores_old.mean()
std_old = scores_old.std()
print(mean_old)
print(std_old)
print(pipeline.get_params())

grid = {
    'classifier__C': [0.05,0.25,0.5,0.75,0.85,0.95,1.0],\
    'classifier__penalty' : ['l1','l2'],\
    'classifier__class_weight': [None, 'balanced'],'classifier__random_state' : [0]  
}
grid_search = GridSearchCV(pipeline, param_grid=grid, scoring='accuracy', n_jobs=1, cv=5)
grid_search.fit(X=X_train, y=y_train)

print("-----------")
print(grid_search.best_score_)
print(grid_search.best_params_)     

'''
0.635906040268
{'classifier__class_weight': None, 'classifier__C': 0.75, 'classifier__random_state': 0, 'classifier__penalty': 'l1'}
'''      

pipeline = Pipeline([('classifier', LogisticRegression(C = 0.75, class_weight = None , random_state = 0, penalty = 'l1'))])
scores_old = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=5)
mean_old = scores_old.mean()
std_old = scores_old.std()
print(mean_old)
print(std_old)
print(pipeline.get_params())      

#y_train1 = y_train1.replace([1,2,3],['High MCI+Low Dementia', 'High Dementia', 'low MCI + control'])
rf = LogisticRegression(C = 0.75, class_weight = None , random_state = 0, penalty = 'l1')
rf.fit(X_train, y_train)
predictions  = rf.predict_proba(X_test)

'''
Generating roc curve
'''
y_true = y_test # ground truth labels
y_probas = predictions# predicted probabilities generated by sklearn classifier
skplt.metrics.plot_roc_curve(y_true, y_probas, title= 'm24 3 classes')
plt.text(0.8, 0.6, '0= low | 1= medium |\n2= high | 3 = control |', fontsize=12)
plt.show()

'''
Generating confusion matrix
'''
y_predicted = rf.predict(X_test1)
s = y_test1#.replace([1,2,0,3],['High MCI+Low Dementia', 'High Dementia', 'low MCI','control'])
c = pd.DataFrame(y_predicted)#.replace([1,2,0,3],['High MCI+Low Dementia', 'High Dementia', 'low MCI','control'])
skplt.metrics.plot_confusion_matrix(s,c, title = 'm24 4 classes',x_tick_rotation =15)
plt.show()
'''
Explained Variance
'''
explained_variance_score(y_true, y_probas) 





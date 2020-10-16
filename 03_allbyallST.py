"""
cantin ST all by all LASSO (and linear regression)
copied from 2_allbyallST.ipynb

Shaina Lu
Zador & Gillis Labs
April 2020"""

from __future__ import division
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split #stratify train/test split
import random

#infiles
ST_CANTIN_FILT_PATH = "/home/slu/spatial/data/cantin_ST_filt_v2.h5"
ONTOLOGY_PATH = "/data/slu/allen_adult_mouse_ISH/ontologyABA.csv"
#outfiles
OUT_PATH_TEST = "allbyallST0p1permute_test0628.csv"
OUT_PATH_TRAIN = "allbyallST0p1permute_train0628.csv"

######################################################################################
#preprocessing functions
def read_data():
    """read in all datasets needed using pandas
       this is new from before"""
    STspotsmeta = pd.read_hdf(ST_CANTIN_FILT_PATH, key='STspotsmeta', mode='r')
    STspots = pd.read_hdf(ST_CANTIN_FILT_PATH, key='STspots', mode='r')
    STpropont = pd.read_hdf(ST_CANTIN_FILT_PATH, key='STpropont', mode='r')

    ontology = pd.read_csv(ONTOLOGY_PATH)
    ontology = ontology.drop([ontology.columns[5], ontology.columns[6]], axis=1)
    ontology = ontology.fillna(-1)  #make root's parent -1

    return STspotsmeta, STspots, STpropont, ontology

def zscore(voxbrain):
    """zscore voxbrain or subsets of voxbrain (rows: voxels, cols: genes)"""
    #z-score 
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(voxbrain)
    z_voxbrain = scaler.transform(voxbrain, copy=True)
    
    #store z-scored voxbrain as pandas dataframe
    z_voxbrain = pd.DataFrame(z_voxbrain)
    z_voxbrain.columns = voxbrain.columns
    z_voxbrain.index = voxbrain.index
    
    return z_voxbrain

def filterproponto(sampleonto):
    """pre-processing for propogated ontology for ST data"""
    #remove brain areas that don't have any samples
    sampleonto_sums = sampleonto.apply(lambda col: col.sum(), axis=0)
    sampleonto = sampleonto.loc[:,sampleonto_sums > 5] #greater than 5 becuase less is not enough for train/test split to have non-zero areas
    
    return sampleonto

def getleaves(propontvox, ontology):
    #leaves are brain areas in the ontology that never show up in the parent column
    allareas = list(propontvox)
    parents = list(ontology.parent)
    for i in range(len(parents)): #convert parents from float to int, ids are ints
        parents[i] = int(parents[i])
    
    #remove parents from all areas
    leaves = []
    for area in allareas:
        if int(area) not in parents:
            leaves.append(area)
    
    return leaves

def analytical_auroc(featurevector, binarylabels):
    """analytical calculation of auroc
       inputs: feature (mean rank of expression level), binary label
       returns: auroc
    """
    #sort ctxnotctx binary labels by mean rank, aescending
    s = sorted(zip(featurevector, binarylabels))
    feature_sort, binarylabels_sort = map(list, zip(*s))
    #print feature_sort
    #print binarylabels_sort

    #get the sum of the ranks in feature vector corresponding to 1's in binary vector
    sumranks = 0
    for i in range(len(binarylabels_sort)):
        if binarylabels_sort[i] == 1:
            sumranks = sumranks + feature_sort[i]
    
    poslabels = binarylabels.sum()
    neglabels = (len(binarylabels) - poslabels) #- (len(binarylabels) - binarylabels.count())  #trying to subtract out 
    
    auroc = ((sumranks/(neglabels*poslabels)) - ((poslabels+1)/(2*neglabels)))
    
    return auroc

#########################################################################################
#LASSO functions

def applyLASSO(Xtrain, Xtest, ytrain, ytest):
    """apply LASSO regression"""
    lasso_reg = Lasso(alpha=0.1, max_iter=10000)
    #lasso_reg = LinearRegression()
    lasso_reg.fit(Xtrain, ytrain)
    
    #train
    predictions_train = lasso_reg.predict(Xtrain)
    auroc_train = analytical_auroc(sp.stats.mstats.rankdata(predictions_train), ytrain)
    #test
    predictions_test = lasso_reg.predict(Xtest)
    auroc_test = analytical_auroc(sp.stats.mstats.rankdata(predictions_test), ytest)
    
    return auroc_train, auroc_test

def getallbyall(data, propont):
    #initialize zeros dataframe to store entries
    allbyall_test = pd.DataFrame(index=list(propont), columns=list(propont))
    allbyall_test = allbyall_test.fillna(0)
    allbyall_train = pd.DataFrame(index=list(propont), columns=list(propont))
    allbyall_train = allbyall_train.fillna(0)
    
    areas = list(propont)
    #for each column, brain area
    for i in range(propont.shape[1]):
        print("col %d" %i)
        #for each row in each column
        for j in range(i+1,propont.shape[1]): #upper triangular!
            area1 = areas[i]
            area2 = areas[j]
            #get binary label vectors
            ylabels1 = propont.loc[propont[area1]+propont[area2] != 0, area1]
            ylabels = pd.Series(np.random.permutation(ylabels1),index=ylabels1.index) #try permuting
            #subset train and test sets for only samples in the two areas
            Xcurr = data.loc[propont[area1]+propont[area2] != 0, :]
            #split train test for X data and y labels
            #split data function is seeded so all will split the same wa
            Xtrain, Xtest, ytrain, ytest = train_test_split(Xcurr, ylabels, test_size=0.5,\
                                                            random_state=42, shuffle=True,\
                                                            stratify=ylabels)

            #z-score current X data
            zXtrain = zscore(Xtrain)
            zXtest = zscore(Xtest)
            
            #NOTE: train test folds filpped for this run for fold 2
            currauroc_train, currauroc_test = applyLASSO(zXtest, zXtrain, ytest, ytrain)
            allbyall_train.iloc[i,j] = currauroc_train
            allbyall_test.iloc[i,j] = currauroc_test
            #curr_row[0,j] = currauroc
            
        #if i == 1:
        #break
     
    #return temp
    return allbyall_train, allbyall_test

########################################################################################
#Main

def main():
    #pre-processing
    STspotsmeta, STspots, STpropont, ontology = read_data()
    STpropont = filterproponto(STpropont)
    STspots = STspots.astype('float64') #convert int to float for z-scoring
    #get leaf areas
    leaves = getleaves(STpropont, ontology)
    leafSTpropont = STpropont.loc[STspotsmeta.id.isin(leaves),leaves] #subset prop onto for leaf areas
    leafSTspots = STspots.loc[STspotsmeta.id.isin(leaves),:] #subset data for samples from leaves
    
    #predictability matrix using LASSO
    allbyall_train, allbyall_test = getallbyall(leafSTspots, leafSTpropont)
    allbyall_test.to_csv(OUT_PATH_TEST, sep=',', header=True, index=False)
    allbyall_train.to_csv(OUT_PATH_TRAIN, sep=',', header=True, index=False)
    
main()

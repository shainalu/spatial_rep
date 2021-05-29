"""
Script that will determine best hyperparameters for LASSO using cross validation
Run on newly filtered ABA, copied from allenadultmouseISH/allbyall1f.py

Shaina Lu
Gillis & Zador Labs
March 2021
"""

from __future__ import division
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV #stratify train/test split
from sklearn.metrics import roc_auc_score
import random

#globals to reset as needed
ALLEN_FILT_PATH = "/home/slu/spatial/data/ABAISH_filt_v6_avgdup.h5"
ONTOLOGY_PATH = "/data/slu/allen_adult_mouse_ISH/ontologyABA.csv"
OUT_PATH_1 = "ABA_bestscore.csv"
OUT_PATH_2 = "ABA_bestparams.csv"
OUT_PATH_3 = "ABA_meantestscore.csv"
    
######################################################################################
#preprocessing functions

def read_data():
    """read in all datasets needed using pandas"""
    metabrain = pd.read_hdf(ALLEN_FILT_PATH, key='metabrain', mode='r')
    voxbrain = pd.read_hdf(ALLEN_FILT_PATH, key='avgvoxbrain', mode='r')
    propontvox = pd.read_hdf(ALLEN_FILT_PATH, key='propontology', mode='r')
    #geneIDName = pd.read_hdf(ALLEN_FILT_PATH, key='geneIDName', mode='r')	

    ontology = pd.read_csv(ONTOLOGY_PATH)
    ontology = ontology.drop([ontology.columns[5], ontology.columns[6]], axis=1)
    ontology = ontology.fillna(-1)  #make root's parent -1

    return metabrain, voxbrain, propontvox, ontology

def zscore(voxbrain):
    """zscore voxbrain or subsets of voxbrain (rows: voxels, cols: genes)"""
    #z-score on whole data set before splitting into test and train
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(voxbrain)
    z_voxbrain = scaler.transform(voxbrain)
    
    #store z-scored voxbrain as pandas dataframe
    z_voxbrain = pd.DataFrame(z_voxbrain)
    z_voxbrain.columns = voxbrain.columns
    z_voxbrain.index = voxbrain.index
    
    return z_voxbrain

def filterproponto(sampleonto):
    """pre-processing for propogated ontology"""
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
    
    try:
        auroc = ((sumranks/(neglabels*poslabels)) - ((poslabels+1)/(2*neglabels)))
    except:
        print("divide by 0 in auroc")
        auroc = -1
    
    return auroc
    
######################################################################################
def getallbyall(data, propont):
    """only running this on a subset of brain areas with over X samples"""
    #initialize zeros dataframe to store entries
    bestscore = pd.DataFrame(index=list(propont), columns=list(propont))
    bestscore = bestscore.fillna(0)
    bestparams = pd.DataFrame(index=list(propont), columns=list(propont))
    bestparams = bestparams.fillna(0)
    meantestscore = {}

    areas = list(propont)
    #for each column, brain area
    for i in range(propont.shape[1]):
        print("col %d" %i)
        #for each row in each column
        for j in range(i+1,propont.shape[1]): #upper triangular!
            area1 = areas[i]
            area2 = areas[j]
            #get binary label vectors
            ylabels = propont.loc[propont[area1]+propont[area2] != 0, area1]
            #subset train and test sets for only samples in the two areas
            Xcurr = data.loc[propont[area1]+propont[area2] != 0, :]

            #z-score moved to before split for CV
            zXcurr = zscore(Xcurr)

            #split train test for X data and y labels
            #Xtrain, Xtest, ytrain, ytest = train_test_split(Xcurr, ylabels, test_size=0.2,\
            #                                                random_state=42, shuffle=True,\
            #                                                stratify=ylabels)
            #further stratified splits for CV
            ssplits = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            ssplits.split(Xcurr,ylabels)

            #to actually get the folds
            #for train_index, test_index in ssplits.split(Xcurr, ylabels):
            #    #print("TRAIN:", train_index, "TEST:", test_index)
            #    X_train, X_test = Xcurr.iloc[train_index], Xcurr.iloc[test_index]
            #    y_train, y_test = ylabels.iloc[train_index], ylabels.iloc[test_index]

            model = Lasso(max_iter=10000)
            alpha = [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]

            grid = dict(alpha=alpha)
            grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, \
                                       cv=ssplits, scoring='roc_auc', error_score=-1)
            grid_result = grid_search.fit(zXcurr,ylabels)


            #z-score current X data
            #zXtrain = zscore(Xtrain)
            #zXtest = zscore(Xtest)

            bestscore.iloc[i,j] = grid_result.best_score_
            bestparams.iloc[i,j] = grid_result.best_params_['alpha']
            key = "%s,%s" %(str(i), str(j))
            meantestscore[key] = grid_result.cv_results_['mean_test_score']
            break

        #if i == 1:
        break
    return bestscore, bestparams, meantestscore
    
def main():
    #pre-processing
    metabrain, voxbrain, propontvox, ontology = read_data()
    propontvox = filterproponto(propontvox)
    #get leaf areas only
    leaves = getleaves(propontvox,ontology)
    leafpropontvox = propontvox.loc[metabrain.ids.isin(leaves),leaves] #subset propontvox for leaf areas
    leafvoxbrain = voxbrain.loc[metabrain.ids.isin(leaves),:] #subset voxbrain for voxels from leaves
    
    #subset for sufficient samples size for hyperparameter selection
    leafsizes = leafpropontvox.sum()
    #plt.hist(leafsizes, bins=100)
    #plt.xlabel("leaf brain area sizes ABA")
    #plt.ylabel("number of brain areas")
    #plt.savefig("032321.pdf", bbox_inches='tight')
    #filter propogated ontology for brain areas with X number of samples min
    leafsizes_sub = leafsizes[leafsizes>100]  #139 areas
    leafpropontvox_sub = leafpropontvox.loc[:,leafsizes_sub.index]
    #remove areas (rows) from propogated ontology that don't have any samples
    rowsum = leafpropontvox_sub.sum(axis=1)
    leafpropontvox_sub = leafpropontvox_sub.loc[rowsum > 0, :]
    #index samples for rows to match porp ont
    leafvoxbrain_sub = leafvoxbrain.loc[rowsum > 0, :]
    print(len(leafsizes_sub))
    print(leafvoxbrain_sub.shape)
    print(leafpropontvox_sub.shape)
    
    #predictability matrix using LASSO
    bestscore, bestparams, meantestscore = getallbyall(leafvoxbrain_sub, leafpropontvox_sub)
    bestscore.to_csv(OUT_PATH_1, sep=',', header=True, index=False)
    bestparams.to_csv(OUT_PATH_2, sep=',', header=True, index=False)
    np.save(OUT_PATH_3, meantestscore)
 
main()

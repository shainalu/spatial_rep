"""
cross dataset re-do w/ new ST data
major changes: 
 - gene name referencing depends on gene symbol w/o converting to entrez ID's first
 - read functions changed
 - filter proponto is lightly changed
 
Shaina Lu
May 2020
Zador + Gillis Labs
"""

import matplotlib.pyplot as plt
import h5py
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split #stratify train/test split
import random

#file paths
ALLEN_FILT_PATH = "/home/slu/spatial/data/ABAISH_filt_v6_avgdup.h5"
ONTOLOGY_PATH = "/data/slu/allen_adult_mouse_ISH/ontologyABA.csv"
ST_CANTIN_FILT_PATH = "/home/slu/spatial/data/cantin_ST_filt_v2.h5"

#outfiles
MOD_TEST_OUT = "STtoABA_STtest_f1_0p1newsplit_031921.csv"
MOD_TRAIN_OUT = "STtoABA_STtrain_f1_0p1newsplit_031921.csv"
CROSS_ALL_OUT = "STtoABA_ABAall_f1_0p1newsplit_031921.csv"

################################################################################################
#read data and pre-processing functions
def read_ABAdata():
    """read in all ABA datasets needed using pandas"""
    metabrain = pd.read_hdf(ALLEN_FILT_PATH, key='metabrain', mode='r')
    voxbrain = pd.read_hdf(ALLEN_FILT_PATH, key='avgvoxbrain', mode='r')
    propontvox = pd.read_hdf(ALLEN_FILT_PATH, key='propontology', mode='r')
    #geneIDName = pd.read_hdf(ALLEN_FILT_PATH, key='geneIDName', mode='r')	

    return metabrain, voxbrain, propontvox

def read_STdata():
    """read in all ST datasets needed using pandas"""
    STspotsmeta = pd.read_hdf(ST_CANTIN_FILT_PATH, key='STspotsmeta', mode='r')
    STspots = pd.read_hdf(ST_CANTIN_FILT_PATH, key='STspots', mode='r')
    STpropont = pd.read_hdf(ST_CANTIN_FILT_PATH, key='STpropont', mode='r')
    
    return STspotsmeta, STspots, STpropont

def read_ontology():
    ontology = pd.read_csv(ONTOLOGY_PATH)
    ontology = ontology.drop([ontology.columns[5], ontology.columns[6]], axis=1)
    ontology = ontology.fillna(-1)  #make root's parent -1

    return ontology

def filterproponto(sampleonto):
    """pre-processing for propogated ontology"""
    #remove brain areas that don't have any samples
    sampleonto_sums = sampleonto.apply(lambda col: col.sum(), axis=0)
    sampleonto = sampleonto.loc[:,sampleonto_sums > 5] #greater than 5 becuase less is not enough for train/test split to have non-zero areas
    
    return sampleonto

def getleaves(propontvox, ontology):
    """helper function to get only leaf brain areas"""
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
    
    print("number of leaf areas: %d" %len(leaves))
    return leaves

def findoverlapareas(STonto, propontvox, ontology):
    """find leaf brain areas overlapping between the two datasets"""
    leafST = getleaves(STonto, ontology)
    leafABA = getleaves(propontvox, ontology)

    leafboth = [] 
    for i in range(len(leafABA)):
        if leafABA[i] in leafST:
            leafboth.append(leafABA[i])
    
    STonto = STonto.loc[:,leafboth]
    propontvox = propontvox.loc[:,leafboth]
    
    return STonto, propontvox    

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

def analytical_auroc(featurevector, binarylabels):
    """analytical calculation of auroc
       inputs: feature (mean rank of expression level), binary label (ctxnotctx)
       returns: auroc
    """
    #sort ctxnotctx binary labels by mean rank, aescending
    s = sorted(zip(featurevector, binarylabels))
    feature_sort, binarylabels_sort = map(list, zip(*s))

    #get the sum of the ranks in feature vector corresponding to 1's in binary vector
    sumranks = 0
    for i in range(len(binarylabels_sort)):
        if binarylabels_sort[i] == 1:
            sumranks = sumranks + feature_sort[i]
    
    poslabels = binarylabels.sum()
    neglabels = (len(binarylabels) - poslabels)
    
    auroc = ((sumranks/(neglabels*poslabels)) - ((poslabels+1)/(2*neglabels)))
    
    return auroc

def getoverlapgenes(STspots, ABAvox):
    ABAgenes = list(ABAvox)
    STgenes = list(STspots)
    
    #get overlapping genes
    overlap = []
    for i in range(len(ABAgenes)):
        if ABAgenes[i] in STgenes:
            overlap.append(ABAgenes[i])
    
    print("number of overlapping genes: %d" %len(overlap))
    
    #index datasets to keep only genes that are overlapping
    STspots = STspots.loc[:,overlap]
    ABAvox = ABAvox.loc[:,overlap]
    
    return STspots, ABAvox

################################################################################################
#lasso functions
def applyLASSO(Xtrain, Xtest, Xcross, ytrain, ytest, ycross):
    """apply LASSO regression"""
    #lasso_reg = sklearn.linear_model.Lasso(alpha=alphaval)
    lasso_reg = Lasso(alpha=0.1,max_iter=10000) #alpha=alphaval) #,max_iter=10000)
    #lasso_reg = LinearRegression()
    #lasso_reg = LogisticRegression(penalty='l1', solver='saga', max_iter=10000, multi_class='ovr')
    lasso_reg.fit(Xtrain, ytrain)
    
    #train
    predictions_train = lasso_reg.predict(Xtrain)
    auroc_train = analytical_auroc(sp.stats.mstats.rankdata(predictions_train), ytrain)
    #auroc_train = metrics.roc_auc_score(y_true = ytrain, y_score = predictions_train)
    #test
    predictions_test = lasso_reg.predict(Xtest)
    auroc_test = analytical_auroc(sp.stats.mstats.rankdata(predictions_test), ytest)
    #auroc_test = metrics.roc_auc_score(y_true = ytest, y_score = predictions_test)
    
    #cross
    predictions_cross = lasso_reg.predict(Xcross)
    auroc_cross = analytical_auroc(sp.stats.mstats.rankdata(predictions_cross), ycross)
    
    return auroc_train, auroc_test, auroc_cross

def getallbyall(mod_data, mod_propont, cross_data, cross_propont):
    #initialize zeros dataframe to store entries
    allbyall_test = pd.DataFrame(index=list(mod_propont), columns=list(mod_propont))
    allbyall_train = pd.DataFrame(index=list(mod_propont), columns=list(mod_propont))
    allbyall_cross = pd.DataFrame(index=list(mod_propont), columns=list(mod_propont))

    areas = list(mod_propont)
    #for each column, brain area
    for i in range(mod_propont.shape[1]):
    #for i in range(5,6,1):
        print("col %d" %i)
        #for each row in each column
        for j in range(i+1,mod_propont.shape[1]): #upper triangular!
            area1 = areas[i]
            area2 = areas[j]
            #get binary label vectors
            ylabels = mod_propont.loc[mod_propont[area1]+mod_propont[area2] != 0, area1]
            ycross = cross_propont.loc[cross_propont[area1]+cross_propont[area2] != 0, area1]
            #ylabels = pd.Series(np.random.permutation(ylabels1),index=ylabels1.index) #try permuting
            #subset train and test sets for only samples in the two areas
            Xcurr = mod_data.loc[mod_propont[area1]+mod_propont[area2] != 0, :]
            Xcrosscurr = cross_data.loc[cross_propont[area1]+cross_propont[area2] != 0, :]
            #split train test for X data and y labels
            #split data function is seeded so all will split the same way
            Xtrain, Xtest, ytrain, ytest = train_test_split(Xcurr, ylabels, test_size=0.5,\
                                                            random_state=9, shuffle=True,\
                                                            stratify=ylabels)
            #z-score train and test folds
            zXtrain = zscore(Xtrain)
            zXtest = zscore(Xtest)
            zXcross = zscore(Xcrosscurr)

            currauroc_train, currauroc_test, currauroc_cross = applyLASSO(zXtrain, zXtest, zXcross, ytrain, ytest, ycross)
            allbyall_train.iloc[i,j] = currauroc_train
            allbyall_test.iloc[i,j] = currauroc_test
            allbyall_cross.iloc[i,j] = currauroc_cross

        #if i == 1:
        #break

    return allbyall_train, allbyall_test, allbyall_cross

################################################################################################
#main
def main():
    #read in data
    ontology = read_ontology()
    ABAmeta, ABAvox, ABApropont = read_ABAdata()
    STmeta, STspots, STpropont = read_STdata()
    
    #filter brain areas for those that have at least x samples
    STpropont = filterproponto(STpropont)
    ABApropont = filterproponto(ABApropont)
    #filter brain areas for overlapping leaf areas
    STpropont, ABApropont = findoverlapareas(STpropont, ABApropont, ontology)
    
    #keep only genes that are overlapping between the two datasets
    STspots, ABAvox = getoverlapgenes(STspots, ABAvox)
    
    #predictability matrix using LASSO
    allbyall_train, allbyall_test, allbyall_cross = getallbyall(STspots, STpropont, ABAvox, ABApropont)
    
    allbyall_test.to_csv(MOD_TEST_OUT, sep=',', header=True, index=False)
    allbyall_train.to_csv(MOD_TRAIN_OUT, sep=',', header=True, index=False)
    allbyall_cross.to_csv(CROSS_ALL_OUT, sep=',', header=True, index=False)
    
main()
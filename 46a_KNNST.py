"""
try a dedicated multiclass classifier: KNN
split out classification into full area matrix to calculate AUROC
modified from 40_multiclass.ipynb

Shaina Lu
Zador & Gillis Labs
May 2021
"""

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
from sklearn.neighbors import KNeighborsClassifier
import random

#file paths
#ALLEN_FILT_PATH = "/home/slu/spatial/data/ABAISH_filt_v6_avgdup.h5"
ONTOLOGY_PATH = "/data/slu/allen_adult_mouse_ISH/ontologyABA.csv"
ST_CANTIN_FILT_PATH = "/home/slu/spatial/data/cantin_ST_filt_v2.h5"

#####################################################################################
#pre-processing functions
def read_ABAdata():
    """read in all ABA datasets needed using pandas"""
    metabrain = pd.read_hdf(ALLEN_FILT_PATH, key='metabrain', mode='r')
    voxbrain = pd.read_hdf(ALLEN_FILT_PATH, key='avgvoxbrain', mode='r')
    propontvox = pd.read_hdf(ALLEN_FILT_PATH, key='propontology', mode='r')
    #geneIDName = pd.read_hdf(ALLEN_FILT_PATH, key='geneIDName', mode='r')	

    return metabrain, voxbrain, propontvox

#ST
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

#####################################################################################
#KNN
def collapsey(propont):
    """collapse propont so I can use it to stratify train/test split"""
    columns = propont.columns
    ycollapse = pd.Series(index = propont.index)
    for i in range(len(columns)):
        currrows = np.where(propont[columns[i]] == 1.0)[0]
        ycollapse.iloc[currrows] = columns[i]
    
    return ycollapse

def predvectortomatrix(predvector, mod_propont):
    """break predictions vector out into matrix where ncols = nareas"""
    predmatrix = pd.DataFrame(0, index=(range(len(predvector))), columns = mod_propont.columns)
    for i in range(len(predmatrix.columns)):
        area = predmatrix.columns[i]
        for j in range(len(predvector)):
            if predvector[j] == area:
                predmatrix.iloc[j,i] = 1
                
    return predmatrix

def matrixtoauroc(predmatrix, ytrue):
    """calculate AUROC for each brain area"""
    aurocs = []
    
    for col in predmatrix.columns:
        aurocs.append(analytical_auroc(sp.stats.mstats.rankdata(predmatrix.loc[:,col]),ytrue.loc[:,col]))
        
    return aurocs

def runknn(ycollapse, mod_data, mod_propont):
    #to get single y vector split
    Xtrain, Xtest, ytrain, ytest = train_test_split(mod_data, ycollapse, test_size=0.5,\
                                                    random_state=42, shuffle=True,\
                                                    stratify=ycollapse)
    
    #to get propont split on the same indicies
    null1, null2, ytrain2, ytest2 = train_test_split(mod_data, mod_propont, test_size=0.5,\
                                                    random_state=42, shuffle=True,\
                                                    stratify=ycollapse)
    
    #z-score
    zXtrain = zscore(Xtrain)
    zXtest = zscore(Xtest)
    
    #KNN
    model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', n_jobs=-1)
    model.fit(zXtrain, ytrain)
    
    testpreds = model.predict(zXtest)
    trainpreds = model.predict(zXtrain)
    testpredmatrix = predvectortomatrix(testpreds, mod_propont)
    trainpredmatrix = predvectortomatrix(trainpreds, mod_propont)
    testaurocs = matrixtoauroc(testpredmatrix, ytest2)
    trainaurocs = matrixtoauroc(trainpredmatrix, ytrain2)
    
    print("train mean auroc = %.3f" %np.mean(trainaurocs))
    print("test mean auroc = %.3f" %np.mean(testaurocs))
    
    return testpreds, trainpreds, testaurocs, trainaurocs


#####################################################################################
def main():
    ontology = read_ontology()
    #pre-processing
    STspotsmeta, STspots, STpropont = read_STdata()
    STpropont = filterproponto(STpropont)
    STspots = STspots.astype('float64') #convert int to float for z-scoring
    #get leaf areas
    leaves = getleaves(STpropont, ontology)
    STpropont = STpropont.loc[STspotsmeta.id.isin(leaves),leaves] 
    STspots = STspots.loc[STspotsmeta.id.isin(leaves),:] 
    
    #KNN
    ycollapse = collapsey(STpropont)
    testpreds, trainpreds, testaurocs, trainaurocs = runknn(ycollapse, STspots, STpropont)
    
    np.save("STknnAUROC_train_051121.npy", trainaurocs)
    np.save("STknnAUROC_test_051121.npy", testaurocs)
    
main()
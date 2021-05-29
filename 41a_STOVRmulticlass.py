"""
Use a one v. rest approach to create a multiclass classifier from LASSO

Shaina Lu
Zador & Gillis Labs
April 2021
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
import random

#file paths
#ALLEN_FILT_PATH = "/home/slu/spatial/data/ABAISH_filt_v6_avgdup.h5"
ONTOLOGY_PATH = "/data/slu/allen_adult_mouse_ISH/ontologyABA.csv"
ST_CANTIN_FILT_PATH = "/home/slu/spatial/data/cantin_ST_filt_v2.h5"

######################################################################################
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


######################################################################################
def applyLASSO(Xtrain, Xtest, ytrain, ytest):
    """apply LASSO regression"""
    #lasso_reg = Lasso(alpha=0.01, max_iter=10000)
    model = LinearRegression()
    model.fit(Xtrain, ytrain)
    
    #train
    predictions_train = model.predict(Xtrain)
    #auroc_train = analytical_auroc(sp.stats.mstats.rankdata(predictions_train), ytrain)
    #test
    predictions_test = model.predict(Xtest)
    #auroc_test = analytical_auroc(sp.stats.mstats.rankdata(predictions_test), ytest)
    
    return predictions_train, predictions_test

def allbyall(mod_data,mod_propont):
    trainpreds = pd.DataFrame(columns=list(mod_propont))
    testpreds = pd.DataFrame(columns=list(mod_propont))

    areas = list(mod_propont)
    #for each column, brain area
    for i in range(mod_propont.shape[1]):
        print("col %d" %i)
        area1 = areas[i]

        #get binary label vectors
        ylabels = mod_propont.loc[:, area1]

        #split train test for X data and y labels
        #split data function is seeded so all will split the same way
        Xtrain, Xtest, ytrain, ytest = train_test_split(mod_data, ylabels, test_size=0.5,\
                                                        random_state=42, shuffle=True,\
                                                        stratify=ylabels)
        #z-score train and test folds
        zXtrain = zscore(Xtrain)
        zXtest = zscore(Xtest)

        #fit LASSO on train set for 1 v all
        currpred_train,currpred_test = applyLASSO(zXtrain, zXtest, ytrain, ytest)
        #key = "%s,%s" %(str(i))
        trainpreds[area1] = currpred_train
        testpreds[area1] = currpred_test
        #break
        
        if i%10 == 0:
            trainpreds.to_csv("STtrainpreds_040821.csv", sep=',',header=True,index=False)
            testpreds.to_csv("STtestpreds_040821.csv", sep=',',header=True,index=False)
    
    return trainpreds,testpreds

def getaurocs(mod_data,mod_propont,trainpreds,testpreds):
    print("STARTING AUROCS")
    #get aurocs
    trainauroc = {}
    testauroc = {}

    areas = list(mod_propont)
    #for each column, brain area
    for i in range(mod_propont.shape[1]):
        if i %10 == 0:
            print("col %d" %i)
        area1 = areas[i]

        #get binary label vectors
        ylabels = mod_propont.loc[:, area1]

        #split train test for X data and y labels
        #split data function is seeded so all will split the same way
        Xtrain, Xtest, ytrain, ytest = train_test_split(mod_data, ylabels, test_size=0.5,\
                                                        random_state=42, shuffle=True,\
                                                        stratify=ylabels)

        #get auroc of new prediction vector
        trainauroc[area1] = analytical_auroc(sp.stats.mstats.rankdata(trainranks[area1]), ytrain)
        testauroc[area1] = analytical_auroc(sp.stats.mstats.rankdata(testranks[area1]), ytest)
        
        
    return trainauroc,testauroc
######################################################################################
def main():
    #read in data
    ontology = read_ontology()
    STspotsmeta, STspots, STpropont = read_STdata()
    #pre-processing
    STpropont = filterproponto(STpropont)
    STspots = STspots.astype('float64') #convert int to float for z-scoring
    #get leaf areas
    leaves = getleaves(STpropont, ontology)
    STpropont = STpropont.loc[STspotsmeta.id.isin(leaves),leaves] #subset prop onto for leaf areas
    STspots = STspots.loc[STspotsmeta.id.isin(leaves),:] #subset data for samples from leaves
    
    trainpreds,testpreds = allbyall(STspots, STpropont)
    trainpreds.to_csv("STtrainpreds_040821.csv", sep=',',header=True,index=False)
    testpreds.to_csv("STtestpreds_040821.csv", sep=',',header=True,index=False)
    
    #rank rows
    trainranks = sp.stats.mstats.rankdata(trainpreds.to_numpy(),axis=1)
    testranks = sp.stats.mstats.rankdata(testpreds.to_numpy(),axis=1)

    #convert back to pandas
    trainranks = pd.DataFrame(trainranks, index=trainpreds.index, columns=trainpreds.columns)
    testranks = pd.DataFrame(testranks, index=testpreds.index, columns=testpreds.columns)
    
    trainauroc,testauroc = getaurocs(STspots, STpropont, trainpreds, testpreds)
    np.save("STtrainauroc_040821.npy", trainauroc)
    np.save("STtestauroc_040821.npy", testauroc)

main()
    
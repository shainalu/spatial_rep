"""
new script for cross dataset separating out the planes of ABA plus Cantin ST
copied from: 14_planecrossdataset.ipynb
modified from: allenadultmouseISH/allbyall_planecrossdataset.py, allenadultmouseISH/crossplane_sixteen.ipynb and spatial/10_crossdataset.py


Shaina Lu
Zador + Gillis Labs
May 2020
"""

import matplotlib.pyplot as plt
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
ALLEN_FILT_PATH = "/home/slu/spatial/data/ABAISH_filt_v6.h5" #non-averaged version
ONTOLOGY_PATH = "/data/slu/allen_adult_mouse_ISH/ontologyABA.csv"
ST_CANTIN_FILT_PATH = "/home/slu/spatial/data/cantin_ST_filt_v2.h5"
ALLEN_PLANES_PATH = '/home/slu/spatial/data/ABAplanes_v2.h5'

#outfiles
MOD_TEST_OUT = "crossplane100_SAGtest_0p1_032721.csv"
MOD_TRAIN_OUT = "crossplane100_SAGtrain_0p1_032721.csv"
CROSS1_ALL_OUT = "crossplane100_SAGtoCOR_0p1_032721.csv"
CROSS2_ALL_OUT = "crossplane100_SAGtoST_0p1_032721.csv"

################################################################################################
#read in data and pre-processing functions
def read_ABAdata():
    """read in all ABA datasets needed using pandas"""
    metabrain = pd.read_hdf(ALLEN_FILT_PATH, key='metabrain', mode='r')
    voxbrain = pd.read_hdf(ALLEN_FILT_PATH, key='voxbrain', mode='r')
    propontvox = pd.read_hdf(ALLEN_FILT_PATH, key='propontology', mode='r')
    geneIDName = pd.read_hdf(ALLEN_FILT_PATH, key='geneIDName', mode='r')	

    return metabrain, voxbrain, propontvox, geneIDName

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

def read_ABAplanes():
    sagvoxbrain = pd.read_hdf(ALLEN_PLANES_PATH, key='sagvoxbrain', mode='r')
    corvoxbrain = pd.read_hdf(ALLEN_PLANES_PATH, key='corvoxbrain', mode='r')
    
    return sagvoxbrain, corvoxbrain

def filterproponto(sampleonto, cutoff):
    """pre-processing for propogated ontology"""
    #remove brain areas that don't have any samples
    sampleonto_sums = sampleonto.apply(lambda col: col.sum(), axis=0)
    sampleonto = sampleonto.loc[:,sampleonto_sums > cutoff]
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

def getoverlapgenes(corvoxbrain,sagvoxbrain,STspots):
    """find genes that are present in all 3 datasets using gene symbol directly"""
    #this function is new from before, but same idea as sptial/10_crossdataset.py
    corgenes = list(corvoxbrain)
    saggenes = list(sagvoxbrain)
    STgenes = list(STspots)

    #getoverlapping genes
    overlap = []
    for i in range(len(STgenes)):
        if STgenes[i] in corgenes:
            if STgenes[i] in saggenes:
                overlap.append(STgenes[i])

    print("number of overlapping genes: %d" %len(overlap))

    #subset datasets to only keep genes that are overlapping
    corvoxbrain = corvoxbrain.loc[:,overlap]
    sagvoxbrain = sagvoxbrain.loc[:,overlap]
    STspots = STspots.loc[:,overlap]

    return corvoxbrain, sagvoxbrain, STspots

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

################################################################################################
#LASSO functions
def applyLASSO(Xtrain, Xtest, Xcross1, Xcross2, ytrain, ytest, ycross1, ycross2):
    """apply LASSO regression"""
    lasso_reg = Lasso(alpha=0.1,max_iter=10000) #alpha=alphaval) #,max_iter=10000)
    #lasso_reg = LinearRegression()
    lasso_reg.fit(Xtrain, ytrain)
    
    #train
    predictions_train = lasso_reg.predict(Xtrain)
    auroc_train = analytical_auroc(sp.stats.mstats.rankdata(predictions_train), ytrain)
    #auroc_train = metrics.roc_auc_score(y_true = ytrain, y_score = predictions_train)
    #test
    predictions_test = lasso_reg.predict(Xtest)
    auroc_test = analytical_auroc(sp.stats.mstats.rankdata(predictions_test), ytest)
    #auroc_test = metrics.roc_auc_score(y_true = ytest, y_score = predictions_test)
    
    #cross 1
    predictions_cross1 = lasso_reg.predict(Xcross1)
    auroc_cross1 = analytical_auroc(sp.stats.mstats.rankdata(predictions_cross1), ycross1)
    
    #cross 2
    predictions_cross2 = lasso_reg.predict(Xcross2)
    auroc_cross2 = analytical_auroc(sp.stats.mstats.rankdata(predictions_cross2), ycross2)
    
    return auroc_train, auroc_test, auroc_cross1, auroc_cross2

def getallbyall(mod_data, mod_propont, cross1_data, cross1_propont, cross2_data, cross2_propont):
    #initialize zeros dataframe to store entries
    allbyall_selftest = pd.DataFrame(index=list(mod_propont), columns=list(mod_propont))
    allbyall_selftrain = pd.DataFrame(index=list(mod_propont), columns=list(mod_propont))
    allbyall_cross1 = pd.DataFrame(index=list(mod_propont), columns=list(mod_propont))
    allbyall_cross2 = pd.DataFrame(index=list(mod_propont), columns=list(mod_propont))
    
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
            ycross1 = cross1_propont.loc[cross1_propont[area1]+cross1_propont[area2] != 0, area1]
            ycross2 = cross2_propont.loc[cross2_propont[area1]+cross2_propont[area2] !=0, area1]
            #ylabels = pd.Series(np.random.permutation(ylabels1),index=ylabels1.index) #try permuting
            #subset train and test sets for only samples in the two areas
            Xcurr = mod_data.loc[mod_propont[area1]+mod_propont[area2] != 0, :]
            Xcross1curr = cross1_data.loc[cross1_propont[area1]+cross1_propont[area2] != 0, :]
            Xcross2curr = cross2_data.loc[cross2_propont[area1]+cross2_propont[area2] != 0, :]
            #split train test for X data and y labels
            #split data function is seeded so all will split the same way
            Xtrain, Xtest, ytrain, ytest = train_test_split(Xcurr, ylabels, test_size=0.5,\
                                                            random_state=42, shuffle=True,\
                                                            stratify=ylabels)
            #z-score train and test folds
            zXtrain = zscore(Xtrain)
            zXtest = zscore(Xtest)
            zXcross1 = zscore(Xcross1curr)
            zXcross2 = zscore(Xcross2curr)
            
            currauroc_train, currauroc_test, currauroc_cross1, currauroc_cross2 = applyLASSO(zXtrain, zXtest, zXcross1, zXcross2, ytrain, ytest, ycross1, ycross2)
            allbyall_selftrain.iloc[i,j] = currauroc_train
            allbyall_selftest.iloc[i,j] = currauroc_test
            allbyall_cross1.iloc[i,j] = currauroc_cross1
            allbyall_cross2.iloc[i,j] = currauroc_cross2
            #curr_row[0,j] = currauroc
            #break
            
        #if i == 1:
        #break
    
    #return temp
    return allbyall_selftrain, allbyall_selftest, allbyall_cross1, allbyall_cross2

################################################################################################
#main
def main():
    #read in data
    ontology = read_ontology()
    ABAmeta, ABAvox, ABApropont, geneIDName = read_ABAdata()
    STmeta, STspots, STpropont = read_STdata()
    sagvoxbrain, corvoxbrain = read_ABAplanes()
    
    #filter brain areas for those that have at least x samples
    STpropont = filterproponto(STpropont, 100)
    ABApropont = filterproponto(ABApropont, 100)
    #filter brain areas for overlapping leaf areas
    STpropont, ABApropont = findoverlapareas(STpropont, ABApropont, ontology)
    
    #keep only genes that are overlapping between the datasets
    corvoxbrain, sagvoxbrain, STspots = getoverlapgenes(corvoxbrain, sagvoxbrain, STspots)
    
    #remove rows that don't have any samples
    STrowsum = STpropont.sum(axis=1)
    STpropont = STpropont.loc[STrowsum > 0, :]
    STspots = STspots.loc[STrowsum > 0, :]

    ABArowsum = ABApropont.sum(axis=1)
    ABApropont = ABApropont.loc[ABArowsum > 0, :]
    corvoxbrain = corvoxbrain.loc[ABArowsum > 0, :]
    sagvoxbrain = sagvoxbrain.loc[ABArowsum > 0, :]
    print("overlapping areas with X samples: %d" %ABApropont.shape[1])
    
    #predictability matrix using LASSO
    allbyall_train, allbyall_test, allbyall_cross1, allbyall_cross2 = getallbyall(sagvoxbrain, ABApropont, corvoxbrain, ABApropont, STspots, STpropont)
    
    allbyall_test.to_csv(MOD_TEST_OUT, sep=',', header=True, index=False)
    allbyall_train.to_csv(MOD_TRAIN_OUT, sep=',', header=True, index=False)
    allbyall_cross1.to_csv(CROSS1_ALL_OUT, sep=',', header=True, index=False)
    allbyall_cross2.to_csv(CROSS2_ALL_OUT, sep=',', header=True, index=False)
    
main()
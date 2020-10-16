"""
re-writing CFS cross dataset for re-do
script version of 12_CFScrossdataset.ipynb
modified from: allenadultmouseISH/allbyallCFS.py, allenadultmouseISH/debugCFSallbyall_twentytwo.ipynb, and spatial/10_crossdataset.py
major changes: MWU vectorized, getting and testing feat sets using pd.apply

Shaina Lu
Zador + Gillis Labs
May 2020
"""

import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split #stratify train/test split
import random
import matplotlib.pyplot as plt
import logging

#file paths
ALLEN_FILT_PATH = "/home/slu/spatial/data/ABAISH_filt_v6_avgdup.h5"
ONTOLOGY_PATH = "/data/slu/allen_adult_mouse_ISH/ontologyABA.csv"
ST_CANTIN_FILT_PATH = "/home/slu/spatial/data/cantin_ST_filt_v2.h5"

#outfiles
FEATSETS_OUT = "ABAtoST_featsetsABAtrain_f1_5_CFS_052420.csv"
MOD_TEST_OUT = "ABAtoST_ABAtest_f1_5_CFS_052420.csv" 
MOD_TRAIN_OUT = "ABAtoST_ABAtrain_f1_5_CFS_052420.csv"
CROSS_ALL_OUT = "ABAtoST_STall_f1_5_CFS_052420.csv"

################################################################################################
#read and pre-processing functions from cross dataset
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
#CFS functions
def calcDE(Xtrain, ytrain):
    #Ha: areaofinterest > not areaofinterest; i.e. alternative = greater
    Xtrain_ranked = Xtrain.apply(lambda col: sp.stats.mstats.rankdata(col), axis=0)
    n1 = ytrain.sum() #instances of brain area marked as 1
    n2 = len(ytrain) - n1
    U = Xtrain_ranked.loc[ytrain==1, :].sum() - ((n1*(n1+1))/2)

    T = Xtrain_ranked.apply(lambda col: sp.stats.tiecorrect(col), axis=0)
    sd = np.sqrt(T * n1 * n2 * (n1+n2+1) / 12.0)
    meanrank = n1*n2/2.0 + 0.5
    z = (U - meanrank) / sd

    pvals_greater = pd.Series(stats.norm.sf(z), index=list(Xtrain), name='pvals_greater')
    
    #Ha: areaofinterest < notareaofinterest; i.e. alternative = less
    Xtrain_ranked = Xtrain.apply(lambda col: sp.stats.mstats.rankdata(col), axis=0)
    n2 = ytrain.sum() #instances of brain area marked as 1
    n1 = len(ytrain) - n1
    U = Xtrain_ranked.loc[ytrain==0, :].sum() - ((n1*(n1+1))/2)

    T = Xtrain_ranked.apply(lambda col: sp.stats.tiecorrect(col), axis=0)
    sd = np.sqrt(T * n1 * n2 * (n1+n2+1) / 12.0)
    meanrank = n1*n2/2.0 + 0.5
    z = (U - meanrank) / sd

    pvals_less = pd.Series(stats.norm.sf(z), index=list(Xtrain), name='pvals_less')
    
    allpvals = pd.concat([pvals_greater, pvals_less], axis=1)
    return allpvals

def getDEgenes(allpvals, numtotal):
    #melt
    allpvals['gene'] = allpvals.index
    allpvals_melt = allpvals.melt(id_vars='gene')
    #sort by p-value
    allpvals_melt = allpvals_melt.sort_values(by='value', ascending=True)
    #get top X number of DE genes
    topDEgenes = allpvals_melt.iloc[0:numtotal, :]
    
    return topDEgenes

def get_featset(featurecorrs, ranksdf, ylabels, seedgene):
    """picking feature set based on fwd sellection, random seed, and lowest possible corr,
       stop when average auroc prediction is no longer improving
       inputs: featurecorrs - correlation matrix of features being considered; seedgene - gene to start CFS
       returns: feature set"""
    #start with passed in randomly picked gene
    featset = [seedgene]
    #get start performance
    curr_auroc = analytical_auroc(sp.stats.mstats.rankdata(ranksdf.loc[:,featset].mean(axis=1)), ylabels)
    improving = True
    while improving:
        #look at all other possible features and take lowest correlated to seed, others in feat set
        means = featurecorrs.loc[:,featset].mean(axis=1)  #get average corr across choosen features
        featset.append(means.idxmin())                  #gets row name of min mean corrs, picks first of ties
        #check featset performance
        new_auroc = analytical_auroc(sp.stats.mstats.rankdata(ranksdf.loc[:,featset].mean(axis=1)),ylabels)
        if new_auroc <= curr_auroc:  #if not improved, stop
            featset.pop(len(featset)-1)
            final_auroc = curr_auroc
            improving = False
        else:
            curr_auroc = new_auroc
            
    return featset

def applyCFS(zXtrain, zXtest, zXcross, ytrain, ytest, ycross):
    #calculate DE for all genes across the two brain areas
    allpvals = calcDE(zXtrain, ytrain)
    #get top X DE genes
    topDEgenes = getDEgenes(allpvals, 500)

    #ranks DE genes
    #train
    rankedXtrain = zXtrain.loc[:, topDEgenes.gene]
    rankedXtrain.loc[:,(topDEgenes.variable=='pvals_less').values] = \
                     -1 * rankedXtrain.loc[:,(topDEgenes.variable=='pvals_less').values]
    rankedXtrain = rankedXtrain.apply(lambda col: sp.stats.mstats.rankdata(col), axis=0)
    #test
    rankedXtest = zXtest.loc[:, topDEgenes.gene]
    rankedXtest.loc[:,(topDEgenes.variable=='pvals_less').values] = \
                    -1 * rankedXtest.loc[:,(topDEgenes.variable=='pvals_less').values]
    rankedXtest = rankedXtest.apply(lambda col: sp.stats.mstats.rankdata(col), axis=0)
    #cross
    rankedXcross = zXcross.loc[:, topDEgenes.gene]
    rankedXcross.loc[:,(topDEgenes.variable=='pvals_less').values] = \
                    -1 * rankedXcross.loc[:,(topDEgenes.variable=='pvals_less').values]
    rankedXcross = rankedXcross.apply(lambda col: sp.stats.mstats.rankdata(col), axis=0)

    #correlation matrix (spearman, b/c already ranked)
    traincorrs = np.corrcoef(rankedXtrain.values.T)
    traincorrs = pd.DataFrame(traincorrs, index=topDEgenes.gene.values, columns=topDEgenes.gene.values)

    #get 100 feature sets using CFS
    random.seed(0)
    random.seed(42)
    startingpts = pd.Series(random.sample(list(traincorrs),100))
    featsets = startingpts.apply(lambda x: get_featset(traincorrs, rankedXtrain, ytrain, x))
    trainaurocs = featsets.apply(lambda x: analytical_auroc(sp.stats.mstats.rankdata(rankedXtrain.loc[:,x].mean(axis=1)), ytrain))
    testaurocs = featsets.apply(lambda x: analytical_auroc(sp.stats.mstats.rankdata(rankedXtest.loc[:,x].mean(axis=1)), ytest))
    crossaurocs = featsets.apply(lambda x: analytical_auroc(sp.stats.mstats.rankdata(rankedXcross.loc[:,x].mean(axis=1)), ycross))

    #return all 100 feature sets and aurocs
    return featsets, trainaurocs, testaurocs, crossaurocs

def getallbyall(mod_data, mod_propont, cross_data, cross_propont):
    #initialize zeros dataframe to store entries
    allbyall_featsets = pd.DataFrame(index=list(mod_propont), columns=list(mod_propont))
    allbyall_test = pd.DataFrame(index=list(mod_propont), columns=list(mod_propont))
    allbyall_train = pd.DataFrame(index=list(mod_propont), columns=list(mod_propont))
    allbyall_cross = pd.DataFrame(index=list(mod_propont), columns=list(mod_propont))

    areas = list(mod_propont)
    #for each column, brain area
    for i in range(361,mod_propont.shape[1]):
        print("starting col %d" %i)
        #for each row in each column
        for j in range(i+1,mod_propont.shape[1]): #upper triangular!
            #print("brain area j: %d" %j)
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
                                                            random_state=42, shuffle=True,\
                                                            stratify=ylabels)
            #z-score train and test folds
            zXtrain = zscore(Xtrain)
            zXtest = zscore(Xtest)
            zXcross = zscore(Xcrosscurr)

            featsets, currauroc_train, currauroc_test, currauroc_cross = applyCFS(zXtrain, zXtest, zXcross, ytrain, ytest, ycross)
            allbyall_featsets.iloc[i,j] = featsets.values
            allbyall_train.iloc[i,j] = currauroc_train.values
            allbyall_test.iloc[i,j] = currauroc_test.values
            allbyall_cross.iloc[i,j] = currauroc_cross.values


        #periodically save
        if i%10 == 0:
            print("saving")
            allbyall_featsets.to_csv(FEATSETS_OUT, sep=',', header=True, index=False)
            allbyall_train.to_csv(MOD_TRAIN_OUT, sep=',', header=True, index=False)
            allbyall_test.to_csv(MOD_TEST_OUT, sep=',', header=True, index=False)
            allbyall_cross.to_csv(CROSS_ALL_OUT, sep=',', header=True, index=False)

        #if i == 360:
        #break

    return allbyall_featsets, allbyall_train, allbyall_test, allbyall_cross

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

    #predictability matrix using CFS
    allbyall_featsets, allbyall_train, allbyall_test, allbyall_cross = getallbyall(ABAvox, ABApropont, STspots, STpropont)

    #write AUROC matrices to outfiles
    allbyall_featsets.to_csv(FEATSETS_OUT, sep=',', header=True, index=False)
    allbyall_train.to_csv(MOD_TRAIN_OUT, sep=',', header=True, index=False)
    allbyall_test.to_csv(MOD_TEST_OUT, sep=',', header=True, index=False)
    allbyall_cross.to_csv(CROSS_ALL_OUT, sep=',', header=True, index=False)

main()


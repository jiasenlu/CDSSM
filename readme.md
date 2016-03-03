## Note: Currently, we only have torch c version batch input sparse linear and sparse covolution. 

##implementation of (c)DSSM in torch

* For more detail of the paper, see (http://research.microsoft.com/en-us/projects/dssm/)

* contributer: Jiasen Lu

##dependencies:
* torch
* tds (DSSM dense)

##Data Preprocessing
###Related Functions:
* Batch.lua
* WordHash.lua
* ComputelogPD.lua
* Preprocess.lua


##Tranining
1: generate data from dataset. The data format follows the C# implementation. Each query and document in the same line, and the seperator is 'Tab'.
2: generate vocabulary for question and answers. Using WordHash.Pair2Voc(). 
you should get the result like this:
'''
Creating Voc file form ...	
srcVoc contains vocabulary: 5584	
tgtVoc contains vocabulary: 10876	
'''
3: Create Pair2Seq Feature and save to txt. Using WordHash.Pair2SeqFea()

4: Convert the seq Feature to Binay file, we give the batchsize here. (this can't be change after you train the model. for orginial data, the batch size is 1024. Using WordHash.SeqFea2Bin(), See more info under the function.

###Related functions
* (Data Provider): BatchSample.lua, SequenceInputStream.lua, PairInputStream.lua
* (Model): DSSM_Train, DSSM_MMI_Criterion.lua
* (Training): th train.lua

##Predicting
1: generate feature file, refer PreProcess.lua for details

* Preprocessing:

* (Predict): th predict.lua

##To-do List

* testing the cu implementation of sparse Linear.
* implement the cu implementation of sparse convolution.



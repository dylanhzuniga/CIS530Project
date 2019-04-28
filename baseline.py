# ==============  SemEval-2015 Task 1  ==============
#  Paraphrase and Semantic Similarity in Twitter
# ===================================================
# Author: Wei Xu (UPenn xwe@cis.upenn.edu)
# ===================================================
# This code was adapted for the CIS530 project 
# baseline. We reimplemented many of the model 
# components, given that the code was out-of-date
# with python dependecies (being deprecated).


from __future__ import division

import sys
import random
import nltk
from nltk.classify import MaxentClassifier
from _pickle import load
from _pickle import dump
from features import readInData

# Evaluation by Precision/Recall/F-measure
# cut-off at probability 0.5, estimated by the model
def OneEvaluation():
    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0
    
    # read in training/test data with labels and create features
    trainfull, _  = readInData(trainfilename)    
    testfull, _  = readInData(testfilename)
    
    train = [(x[0], x[1]) for x in trainfull]
    test  = [(x[0], x[1]) for x in testfull]
    
    print("Read in" , len(train) , "valid training data ... ")
    print("Read in" , len(test) , "valid test data ...  ")
    print()
    if len(test) <=0 or len(train) <=0 :
        sys.exit()

    # train the model
    classifier = nltk.classify.maxent.train_maxent_classifier_with_gis(train)
    
    # uncomment the following lines if you want to save the trained model into a file
    # modelfile = './baseline_logisticregression.model'
    # outmodel = open(modelfile, 'wb')
    # dump(classifier, outmodel)
    # outmodel.close()
    
    # uncomment the following lines if you want to load a trained model from a file
    
    # inmodel = open(modelfile, 'rb') 
    # classifier = load(inmodel)
    # inmodel.close()
    
    for i, t in enumerate(test):
        sent1 = testfull[i][2]
        sent2 = testfull[i][3]
        guess = classifier.classify(t[0])

        label = t[1]
        if guess == True and label == False:
            fp += 1.0
        elif guess == False and label == True:
            fn += 1.0
        elif guess == True and label == True:
            tp += 1.0
        elif guess == False and label == False:
            tn += 1.0  			

        if guess == True:
             print("GOLD-" + str(label) + "\t" + "SYS-" + str(guess) + "\t" + sent1 + "\t" + sent2)

    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F = 2 * P * R / (P + R)
  
    print()
    print("PRECISION: %s, RECALL: %s, F1: %s" % (P,R,F))
 
    print("ACCURACY: ", nltk.classify.accuracy(classifier, test))

    print("# true pos:", tp)
    print("# false pos:", fp)
    print("# false neg:", fn)
    print("# true neg:", tn)



# Evaluation by precision/recall curve
def PREvaluation():
    
    # read in training/test data with labels and create features
    trainfull, traintrends  = readInData(trainfilename)    
    testfull, testtrends  = readInData(testfilename)
    
    
    train = [(x[0], x[1]) for x in trainfull]
    test  = [(x[0], x[1]) for x in testfull]
    
    print("Read in" , len(train) , "valid training data ... ")
    print("Read in" , len(test) , "valid test data ...  ")
    print()
    if len(test) <=0 or len(train) <=0 :
        sys.exit()

    # train the model
    # classifier = nltk.classify.maxent.train_maxent_classifier_with_megam(train, gaussian_prior_sigma=10, bernoulli=True)
    classifier = nltk.classify.maxent.train_maxent_classifier_with_gis(train)

    # comment the following lines to skip saving the above trained model into a file
    
    modelfile = './baseline_logisticregression.model'
    outmodel = open(modelfile, 'wb')
    dump(classifier, outmodel)
    outmodel.close()
    
    # comment the following lines to skip loading a previously trained model from a file
    
    inmodel = open(modelfile, 'rb') 
    classifier = load(inmodel)
    inmodel.close()
            
    probs = []
    totalpos = 0
        
    for i, t in enumerate(test):
    	prob = classifier.prob_classify(t[0]).prob(True)
    	probs.append(prob)
    	goldlabel = t[1]
    	if goldlabel == True:
            totalpos += 1
    	
    # rank system outputs according to the probabilities predicted
    sortedindex = sorted(range(len(probs)), key = probs.__getitem__)   
    sortedindex.reverse() 
    
    truepos = 0
    falsepos = 0
    
    print("\t\tPREC\tRECALL\tF1\t|||\tMaxEnt\tSENT1\tSENT2")
    
    i = 0
    for sortedi in sortedindex:
        i += 1
        strhit = "HIT"
        sent1 = testfull[sortedi][2]
        sent2 = testfull[sortedi][3]
        if test[sortedi][1] == True:
            truepos += 1
        else:
            falsepos += 1
            strhit = "ERR"
            
        precision = truepos / (truepos + falsepos)
        recall = truepos / totalpos
        f1 = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        
        
        print(str(i) + "\t" + strhit + "\t" + "{0:.3f}".format(precision) + '\t' + "{0:.3f}".format(recall) + "\t" + "{0:.3f}".format(f1),)
        print("\t|||\t" + "{0:.3f}".format(probs[sortedi]) + "\t" + sent1 + "\t" + sent2)


# Load the trained model and output the predictions
def OutputPredictions(modelfile, outfile):
    
    # read in test data and create features
    testfull, testtrends  = readInData(testfilename)
    
    test  = [(x[0], x[1]) for x in testfull]
    
    print("Read in" , len(test) , "valid test data ...  ")
    print()
    if len(test) <=0:
        sys.exit()

	# read in pre-trained model    
    inmodel = open(modelfile, 'rb') 
    classifier = load(inmodel)
    inmodel.close()
           
    # output the results into a file
    outf = open(outfile,'w', encoding='utf8') 
          
    for i, t in enumerate(test):
        prob = classifier.prob_classify(t[0]).prob(True)
        if prob >= 0.5:
             outf.write("true\t" + "{0:.4f}".format(prob) + "\n")
        else:
             outf.write("false\t" + "{0:.4f}".format(prob) + "\n")
             
    outf.close()
             
if __name__ == "__main__":
    trainfilename = "./data/train.data"
    testfilename  = "./data/test.data"
    
    # Training and Testing by precision/recall curve
    #PREvaluation()
    
    # Training and Testing by Precision/Recall/F-measure
    OneEvaluation()
    
    # write results into a file in the SemEval output format
    outputfilename = "./baseline.output"
    modelfilename = './baseline_logisticregression.model'
    OutputPredictions(modelfilename, outputfilename)
   
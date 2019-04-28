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
from features import readInData

# Evaluation by Precision/Recall/F-measure
# cut-off at probability 0.5, estimated by the model
def OneEvaluation(trainfilename, testfilename, outputfilename):
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

    print()
    print('Outputting the results')
    # output the results into a file
    with open(outputfilename,'w', encoding='utf8') as outf: 
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
    outputfilename = "./outputs/baseline.output"
    
    OneEvaluation(trainfilename, testfilename, outputfilename)

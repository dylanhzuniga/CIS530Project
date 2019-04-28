import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from features import readInDataFeatureVector
import sys

def setup_models():
  clfs = []
  # clfs.append(LinearRegression())
  clfs.append(LogisticRegression())
  clfs.append(LinearSVC(C=100, loss='hinge'))
  clfs.append(SVC(kernel='rbf'))
  clfs.append(SVC(kernel='poly'))
  clfs.append(SVC(kernel='sigmoid'))
  clfs.append(SVC(kernel=cosine_similarity, C=100))
  clfs.append(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                algorithm="SAMME",
                                n_estimators=200))
  clfs.append(MLPClassifier(solver='lbfgs'))
  clfs.append(MLPClassifier(solver='sgd', learning_rate='constant'))
  clfs.append(MLPClassifier(solver='sgd', learning_rate='adaptive'))
  clfs.append(MLPClassifier(solver='sgd', learning_rate='invscaling'))
  clfs.append(MLPClassifier(solver='adam'))
  clfs.append(MLPClassifier(solver='lbfgs'))
  return clfs
  
def train(clfs, X, y):
  for clf in clfs:
    clf.fit(X, y)
  return clfs

def predict(clfs, X):
  labels = []
  for clf in clfs:
    labels.append(clf.predict(X))
  labels = np.matrix(labels)
  print(labels)
  probs = np.asarray(np.mean(labels, axis=0))
  return np.asarray(probs >= 0.5), probs

def print_scores(label, y, pred_y):
    pred_y = pred_y.reshape(y.shape)
    P = precision_score(y, pred_y)
    R = recall_score(y, pred_y)
    F = f1_score(y, pred_y)
    A = accuracy_score(y, pred_y)
    print(label)
    print("PRECISION: %s, RECALL: %s, F1: %s, ACCURACY: %s" % (P,R,F,A))

def extension_one(trainfilename, devfilename, testfilename, outputfilename):
    # read in training/dev/test data with labels and create features
    train_X, train_y  = readInDataFeatureVector(trainfilename)
    dev_X, dev_y  = readInDataFeatureVector(devfilename)       
    test_X, test_y  = readInDataFeatureVector(testfilename)
    
    print("Read in" , len(train_X) , "valid training data ... ")
    print("Read in" , len(dev_X) , "valid dev data ... ")
    print("Read in" , len(test_X) , "valid test data ...  ")
    print()
    if len(test_X) <=0 or len(dev_X)<=0 or len(train_X) <=0 :
        sys.exit()

    # train the model
    print('Training the model ... ')
    print()
    clfs = setup_models() 
    clfs = train(clfs, train_X, train_y)

    # predict
    print('Making predictions ... ')
    pred_train_y, _ = predict(clfs, train_X)
    pred_dev_y, _ = predict(clfs, dev_X)
    pred_test_y , prob_test = predict(clfs, test_X) 

    print_scores('TRAIN', train_y, pred_train_y)
    print_scores('DEV', dev_y, pred_dev_y)
    # print_scores('TEST', test_y, pred_test_y)
    print()

    # output the results into a file
    print('Outputting the file')
    with open(outputfilename,'w', encoding='utf8') as f:
      prob_test = prob_test.tolist()[0]
      for i in range(len(prob_test)):
        prob = prob_test[i]
        # prin/t(prob)
        if prob >= 0.5:
          f.write("true\t" + "{0:.4f}".format(prob) + "\n")
        else:
          f.write("false\t" + "{0:.4f}".format(prob) + "\n")     
      f.close()

if __name__ == "__main__":
    trainfilename = "./data/train.data"
    devfilename = "./data/dev.data"
    testfilename  = "./data/test.data"
    outputfilename = "./outputs/extensionone.output"
    # write results into a file in the SemEval output format
    extension_one(trainfilename, devfilename, testfilename, outputfilename)


import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import porter
from collections import Counter

# sub-functions for find overlapping n-grams
def intersect_modified (list1, list2) :
    cnt1 = Counter()
    cnt2 = Counter()
    for tk1 in list1:
        cnt1[tk1] += 1
    for tk2 in list2:
        cnt2[tk2] += 1    
    inter = cnt1 & cnt2
    union = cnt1 | cnt2
    largeinter = Counter()
    for (element, count) in inter.items():
        largeinter[element] = union[element]
    return list(largeinter.elements())

def intersect (list1, list2) :
    cnt1 = Counter()
    cnt2 = Counter()
    for tk1 in list1:
        cnt1[tk1] += 1
    for tk2 in list2:
        cnt2[tk2] += 1    
    inter = cnt1 & cnt2
    return list(inter.elements())

# create n-gram features and stemmed n-gram features
def paraphrase_Das_features(source, target, trend):
    source_words = word_tokenize(source)
    target_words = word_tokenize(target)
	
    features = {}
    
    ###### Word Features ########
	
    s1grams = [w.lower() for w in source_words]
    t1grams = [w.lower() for w in target_words]
    s2grams = []
    t2grams = []
    s3grams = []
    t3grams = []
        
    for i in range(0, len(s1grams)-1) :
        if i < len(s1grams) - 1:
            s2gram = s1grams[i] + " " + s1grams[i+1]
            s2grams.append(s2gram)
        if i < len(s1grams)-2:
            s3gram = s1grams[i] + " " + s1grams[i+1] + " " + s1grams[i+2]
            s3grams.append(s3gram)
            
    for i in range(0, len(t1grams)-1) :
        if i < len(t1grams) - 1:
            t2gram = t1grams[i] + " " + t1grams[i+1]
            t2grams.append(t2gram)
        if i < len(t1grams)-2:
            t3gram = t1grams[i] + " " + t1grams[i+1] + " " + t1grams[i+2]
            t3grams.append(t3gram)

    f1gram = 0        
    precision1gram = len(set(intersect(s1grams, t1grams))) / len(set(s1grams))
    recall1gram    = len(set(intersect(s1grams, t1grams))) / len(set(t1grams))
    if (precision1gram + recall1gram) > 0:
        f1gram = 2 * precision1gram * recall1gram / (precision1gram + recall1gram)
    precision2gram = len(set(intersect(s2grams, t2grams))) / len(set(s2grams))
    recall2gram    = len(set(intersect(s2grams, t2grams))) / len(set(t2grams))
    f2gram = 0
    if (precision2gram + recall2gram) > 0:
        f2gram = 2 * precision1gram * recall2gram / (precision2gram + recall2gram)
    precision3gram = len(set(intersect(s3grams, t3grams))) / len(set(s3grams))
    recall3gram    = len(set(intersect(s3grams, t3grams))) / len(set(t3grams))
    f3gram = 0
    if (precision3gram + recall3gram) > 0:
        f3gram = 2 * precision3gram * recall3gram /(precision3gram + recall3gram)

    features["precision1gram"] = precision1gram
    features["recall1gram"] = recall1gram
    features["f1gram"] = f1gram
    features["precision2gram"] = precision2gram
    features["recall2gram"] = recall2gram
    features["f2gram"] = f2gram
    features["precision3gram"] = precision3gram
    features["recall3gram"] = recall3gram
    features["f3gram"] = f3gram
    
    ###### Stemmed Word Features ########
    
    porterstemmer = porter.PorterStemmer()
    s1stems = [porterstemmer.stem(w.lower()) for w in source_words]
    t1stems = [porterstemmer.stem(w.lower()) for w in target_words]
    s2stems = []
    t2stems = []
    s3stems = []
    t3stems = []
        
    for i in range(0, len(s1stems)-1) :
        if i < len(s1stems) - 1:
            s2stem = s1stems[i] + " " + s1stems[i+1]
            s2stems.append(s2stem)
        if i < len(s1stems)-2:
            s3stem = s1stems[i] + " " + s1stems[i+1] + " " + s1stems[i+2]
            s3stems.append(s3stem)
            
    for i in range(0, len(t1stems)-1) :
        if i < len(t1stems) - 1:
            t2stem = t1stems[i] + " " + t1stems[i+1]
            t2stems.append(t2stem)
        if i < len(t1stems)-2:
            t3stem = t1stems[i] + " " + t1stems[i+1] + " " + t1stems[i+2]
            t3stems.append(t3stem)
                
    precision1stem = len(set(intersect(s1stems, t1stems))) / len(set(s1stems))
    recall1stem    = len(set(intersect(s1stems, t1stems))) / len(set(t1stems))
    f1stem = 0
    if (precision1stem + recall1stem) > 0:
        f1stem = 2 * precision1stem * recall1stem / (precision1stem + recall1stem)
    precision2stem = len(set(intersect(s2stems, t2stems))) / len(set(s2stems))
    recall2stem    = len(set(intersect(s2stems, t2stems))) / len(set(t2stems))
    f2stem = 0
    if (precision2stem + recall2stem) > 0:
        f2stem = 2 * precision2stem * recall2stem / (precision2stem + recall2stem)
    precision3stem = len(set(intersect(s3stems, t3stems))) / len(set(s3stems))
    recall3stem    = len(set(intersect(s3stems, t3stems))) / len(set(t3stems))
    f3stem = 0
    if (precision3stem + recall3stem) > 0:
        f3stem = 2 * precision3stem * recall3stem / (precision3stem + recall3stem)
	
    features["precision1stem"] = precision1stem
    features["recall1stem"] = recall1stem
    features["f1stem"] = f1stem
    features["precision2stem"] = precision2stem
    features["recall2stem"] = recall2stem
    features["f2stem"] = f2stem
    features["precision3stem"] = precision3stem
    features["recall3stem"] = recall3stem
    features["f3stem"] = f3stem

    return features

# read from train/test data files and create features dict
def readInData(filename):

    data = []
    trends = set([])
    
    (trendid, trendname, origsent, candsent, judge, origsenttag, candsenttag) = (None, None, None, None, None, None, None)
    
    for line in open(filename):
        line = line.strip()
        #read in training or dev data with labels
        if len(line.split('\t')) == 7:
            (trendid, trendname, origsent, candsent, judge, origsenttag, candsenttag) = line.split('\t')
        #read in test data without labels
        elif len(line.split('\t')) == 6:
            (trendid, trendname, origsent, candsent, origsenttag, candsenttag) = line.split('\t')
        else:
            continue
        
        #if origsent == candsent:
        #    continue
        
        trends.add(trendid)
        features = paraphrase_Das_features(origsent, candsent, trendname)
        
        if judge == None:
            data.append((features, judge, origsent, candsent, trendid))
            continue

        # ignoring the training/test data that has middle label 
        if judge[0] == '(':  # labelled by Amazon Mechanical Turk in format like "(2,3)"
            nYes = eval(judge)[0]            
            if nYes >= 3:
                amt_label = True
                data.append((features, amt_label, origsent, candsent, trendid))
            elif nYes <= 1:
                amt_label = False
                data.append((features, amt_label, origsent, candsent, trendid))   
        elif judge[0].isdigit():   # labelled by expert in format like "2"
            nYes = int(judge[0])
            if nYes >= 4:
                expert_label = True
                data.append((features, expert_label, origsent, candsent, trendid))
            elif nYes <= 2:
                expert_label = False
                data.append((features, expert_label, origsent, candsent, trendid))     
            else:
            	expert_label = None
            	data.append((features, expert_label, origsent, candsent, trendid))        
                
    return data, trends

features_names = ['precision1gram', 'recall1gram', 'f1gram', 
'precision2gram', 'recall2gram', 'f2gram', 
'precision3gram', 'recall3gram', 'f3gram',
'precision1stem', 'recall1stem', 'f1stem', 
'precision2stem', 'recall2stem', 'f2stem', 
'precision3stem', 'recall3stem', 'f3stem']

# reads data and return feature vector
def readInDataFeatureVector(filename):
  data, _ = readInData(filename)
  X = []
  for d, _, _, _, _ in data:
    instance = []
    for feat in features_names:
      # print(d[feat])
      instance.append(d[feat])
    X.append(instance)
  return np.matrix(X), np.array([x[1] for x in data])
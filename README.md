# Simple Baselines #

## Random Baseline ##
Based off of code written by Wei Xu, a student working under Prof. Callison-Burch, we've generated a 
simple baseline which operates at random. For every line of data in the input file, it produces a random 
number between 0 and 1. If that number is below .5 it is labeled as false and true otherwise. Below is 
the output of the eval metric for the baseline. The numbers are F1, Precision and Recall, respectively. 

0.304	0.215	0.520

*Note: to run, change the lines in simple_random_baseline.py to desired file's path.*

## Majority Baseline ##
We implemented a simple majority baseline. We were unable to calculate the normal metrics using the 
evaluation script. The accuracy for this baseline is 0.682. 

*Note: to run, change the lines in simple_majority_baseline.py to desired file's path.*

# Published Baseline #
For our published baseline, we decided to implement the logistic regression model described in Das and Smith (2009). The paper can be found here: https://www.aclweb.org/anthology/P09-1053.

We adapted the model and code from the Wei Xu scripts, updating the model using a gis classifier instead of megam (which is out of date and incompatible with many computers). 

To run the script, make sure that the data is in the correct data folder in the same directory as the script. Then simply run:

	python baseline.py

The scores for the baseline from dev data is:  F1 - 0.489, Precision - 0.692, Recall - 0.378. The scores for the baseline from test data is:  F1 - 0.546, Precision - 0.643, Recall - 0.474. 

#	Extension One #
For our first extension, we engineered new features and ran feature ablation. The table of the various features, and their metric scores can be found in the presentation (page 9).

# Extension Two #
We implemented an ensemble of various classifiers. Each classifier gives a binary ouput, and we take the mean of
all the outputs. If the mean was above a certain threshold, we output a positive label. Otherwise, we hace a negative.

For the development of this extension, we used the 18 published baseline features. The scores for this model from dev data is:  F1 - 0.505, Precision - 0.765, Recall - 0.377. The scores for this model from test data is:  F1 - 0.603, Precision - 0.708, Recall - 0.526.

After completing extension one, we decided that the best features to use is the stem ones. We used this for this model. The scores for this model from dev data is:  F1 - 0.516, Precision - 0.757, Recall - 0.392. The scores for this model from test data is:  F1 - 0.617, Precision - 0.706, Recall - 0.549. As its evident, we improved F1 and recall using these features.

# Extension Three #
We implemented a simple feed forward neural network as our third extension. The architecture is summarized below
in the table.

*Architecture of Neural Network*

| Layer       			| Hyperparameters                								 |
| ----------------- | ---------------------------------------------- |
| Fully Connected 1 | Out channels = 3000. ReLu activation functions |
| Fully Connected 2 | Out channels = 2.                              |

For the development of this extension, we used the 18 published baseline features. The scores for this model from dev data is:  F1 - 0.519, Precision - 0.755, Recall - 0.396. The scores for this model from test data is:  F1 - 0.615, Precision - 0.701, Recall - 0.549.

After completing extension one, we decided that the best features to use is the stem ones. We used this for this model. The scores for this model from dev data is:  F1 - 0.521, Precision - 0.748, Recall - 0.400. The scores for this model from test data is:  F1 - 0.623, Precision - 0.692, Recall - 0.566. These new features improved the F1 and recall scores.
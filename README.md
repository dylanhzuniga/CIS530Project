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

The scores for the baseline from test data is:  F1 - 0.546, Precision - 0.643, Recall - 0.474

The scores for the baseline from dev data is:  F1 - 0.489, Precision - 0.692, Recall - 0.378

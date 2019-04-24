For our published baseline, we decided to implement the logistic regression model described in Das and Smith (2009). The paper can be found here: https://www.aclweb.org/anthology/P09-1053.

We adapted the model and code from the Wei Xu scripts, updating the model using a gis classifier instead of megam (which is out of date and incompatible with many computers). 

To run the script, make sure that the data is in the correct data folder in the same directory as the script. Then simply run:

	python baseline.py

The scores for the baseline from test data is:  F1 - 0.546, Precision - 0.643, Recall - 0.474

The scores for the baseline from dev data is:  F1 - 0.489, Precision - 0.692, Recall - 0.378

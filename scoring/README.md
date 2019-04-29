# Evaluation # 

There are two scripts for the evaluation:

    pit2015_checkformat.py (checks the format of the system output file)
    pit2015_eval_single.py (evaluation metrics)

To run script to check the format of the system output file:

    python pit2015_checkformat.py sampletestdata samplesystemoutput
    
To run the evaluation metrics script:
    
    python pit2015_eval_single.py ../data/test.label ../outputs/baseline.output

Each line in the system output file should match the corresponding lines of the test data. 
Each line should have 2 columns separated by a tab, like this:
    | Binary Label (true/false) | Degreed Score (between 0 and 1, in the 4 decimal format) |

The degreed score is optional. If the system only gives binary labels, 
"0.0000" should be put in all second columns.  

The output looks like:
    size_of_test_data|name_of_the_run||F1|Precision|Recall||Pearson|maxF1|mPrecision|mRecall

F1, Precision, and Recall are based on binary outputs, and Pearson, maxF1, mPrecision, and mRecall
are based on degreed outputs. Pearson refers to the Pearson Correlation coefficient 
(see https://en.wikipedia.org/wiki/Pearson_correlation_coefficient). Higher f1, precision, recall is better. Higher absolute value of correlation is better than lower. 

## Precedent ## 
In the paper "SemEval-2015Task1: Paraphrase and Semantic Similarity in Twitter(PIT)" by Wei Xu, Chris Callison-Burch, and William B. Dolan, they state that following the literature on paraphrase identification, they use F1 score to evaluate system performance. The paper "Extracting Lexically Divergent Paraphrases from Twitter" (again Wei Xu) follows the same protocol. 

## Additional Comments ##
In the paper "DATA-DRIVEN APPROACHES FOR PARAPHRASING ACROSS LANGUAGE VARIATIONS" by Wei Xu, multiple evaluation metrics are tested. These metrics to test paraphrase similarity include Cosine similarity, logistic regression, and language model. All three of these metrics were found to correlate better with human judgments than other existing metrics. As an extension to the project, we could consider evaluating our models as well using these metrics. But our main method of evaluation will be f1 score, as that is the standard used in majority of past papers. 

  

  
    
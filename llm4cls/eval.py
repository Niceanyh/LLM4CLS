import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from llm4cls.utils import majority_voting, outputs2Labels

def compute_mcnemar_table(method1_labels, method2_labels, true_labels):
    # Initialize the four counts
    a = b = c = d = 0
    
    # Calculate the counts based on the labels
    for m1, m2, true in zip(method1_labels, method2_labels, true_labels):
        if m1 == true and m2 == true:
            a += 1
        elif m1 != true and m2 == true:
            b += 1
        elif m1 == true and m2 != true:
            c += 1
        else:
            d += 1
    
    # Return the contingency table as a list
    contingency_table = [[a, b], [c, d]]
    return contingency_table

def perform_mcnemar_test(method1_predictions, method2_predictions,true_labels,text_to_label):
    """"
    Perform McNemar's test on two methods
    
    method1_predictions: list of predictions from method 1
    method2_predictions: list of predictions from method 2
    text_to_label: function that converts text to label
    """
    
    method1_labels = majority_voting(outputs2Labels(method1_predictions, text_to_label))
    method2_labels = majority_voting(outputs2Labels(method2_predictions, text_to_label))
    
    mcnemar_table = compute_mcnemar_table(method1_labels, method2_labels,true_labels)
    result = mcnemar(mcnemar_table, exact=True, correction=True)
    
    statistic = result.statistic
    pvalue = result.pvalue
    
    return statistic, pvalue

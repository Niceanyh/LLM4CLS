import re
from collections import Counter
from sklearn.metrics import classification_report,balanced_accuracy_score

def zero_shot_prompt_builder(task_description,query,tailor_size=None):
    if tailor_size is None:
        
        if isinstance(task_description, list):
            return [single_task_description + " <Input>: " + query["text"] + " <Answer>:" for single_task_description in task_description]
        else:
            return task_description + " <Input>: "+ query["text"]+ " <Answer>:"
    else:
        words = query["text"].split()
        truncated_words = words[:tailor_size]
        truncated_string = ' '.join(truncated_words)
        if isinstance(task_description, list):
            return [single_task_description + " <Input>: " + truncated_string for single_task_description in task_description]
        else:
            return task_description + " <Input>: " + truncated_string + " <Answer>:"


def few_shot_prompt_builder(task_description,query,samples,tailor_size=None):
    """Description

    Args:
        task_description (_String_): task description as a qestion
        query (string): query x
        sample_dataset (_Dataset_): dataset to sample from
        tailor_size (Int, optional): tailor size for query. Defaults to None.

    Returns:
        String: Prompt 
    """
    if tailor_size is None:
        
        if isinstance(task_description, list):
            return [single_task_description + demos +"<Input>: " + query["text"] for single_task_description in task_description]
        else:
            return task_description + demos +"<Input>:  " + query["text"]+ " <Answer>:"
    else:
        words = query["text"].split()
        truncated_words = words[:tailor_size]
        truncated_string = ' '.join(truncated_words)
        
        if isinstance(task_description, list):
            return [single_task_description + "<Input>: " + truncated_string for single_task_description in task_description]
        else:
            return task_description + "<Input>:  " + truncated_string + " <Answer>:"


def outputs2Labels(generated_texts,text_to_label):
    """
    Convert text to labels
    """
    if isinstance(generated_texts, list) and all(isinstance(item, list) for item in generated_texts):
        # Apply method 1 to each element in the 2D list
        labels = [[text_to_label(text) for text in inner_list] for inner_list in generated_texts]
    else:
        # Apply method 2 to each element in the list
        labels = [text_to_label(text) for text in generated_texts]
    
    return labels


def majority_voting(labels):
    """
    Perform majority voting on a list of labels
    """
    if isinstance(labels[0], (int, str, float)):
        return labels
    
    num_rows = len(labels)
    num_cols = len(labels[0])
    
    if num_rows % 2 == 0:
        raise ValueError("Number of prompt must not be even for majority voting")
    
    result = []
    
    for col in range(num_cols):
        column_labels = [labels[row][col] for row in range(num_rows)]
        counts = Counter(column_labels)
        most_common_label = counts.most_common(1)[0][0]
        result.append(most_common_label)
    
    return result

def compute_similarity(query_dataset, sample_dataset):
    """
    Compute the similarity between query dataset and sample dataset
    """
    sim_matrix = []
    for query in query_dataset:
        sim_row = []
        for sample in sample_dataset:
            sim_row.append(query["embedding"].cosine(sample["embedding"]))
        sim_matrix.append(sim_row)
    return sim_matrix

def eval(true_y,pre_y):
    task_completeness = 1- ((pre_y.count(-1))/len(pre_y))
    pre_y[:] = [1 if x == -1 else x for x in pre_y]
    print(classification_report(true_y, pre_y,labels=[0,1],target_names=["Not Personal","Personal"]))
    print("BAC: ",round(balanced_accuracy_score(true_y, pre_y), 3))
    print("Task Completeness: ",round(task_completeness,3))
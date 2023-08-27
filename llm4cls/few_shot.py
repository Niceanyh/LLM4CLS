from . import util 
from tqdm.auto import tqdm
from datasets import Dataset
import numpy as np

def inference(dataset,sample_dataset,model,tokenizer,task_description,label2text,device,k,sample_method="random",max_new_tokens=20,temperature=0.1, tailor_size=None,majority_vote=False):
    """
    Zero-shot inference
    dataset: Dataset({features: ['label', 'text', 'embedding']}
    sample_dataset: Dataset({features: ['label', 'text', 'embedding']}
    model: Huggingface model
    tokenizer: Huggingface tokenizer
    task_description: A string or a list of strings
    device: torch.device
    k: int number of samples to sample from dataset
    sample_method: str ["random", "knn"]
    temperature: float
    tailor_size: int
    majority_vote: bool
    """
    if majority_vote:
        if isinstance(task_description, str):
            raise ValueError("task_description needs to be a list when majority_vote is True.")
    else:
        if not isinstance(task_description, str):
            raise ValueError("task_description needs to be a string when majority_vote is False.")
        
    
    if majority_vote:
        # Perform majority vote inference
        all_generated_texts = []
        num_samples =k * len(task_description)

        for query in tqdm(dataset):
            generated_texts_for_query = []
            samples = sampler(sample_method, sample_dataset, query, num_samples, shuffle=True)
            
            indices = [[i*k+j for j in range(k)] for i in range(len(task_description))]
            samples_subset = [samples.shuffle().select(indice) for indice in indices]
            for i in range(len(task_description)):
                encoded_inputs = tokenizer.encode(util.few_shot_prompt_builder(
                    task_description[i], query, samples_subset[i],label2text, tailor_size), return_tensors="pt").to(device)
                outputs = model.generate(encoded_inputs, do_sample=True,max_new_tokens=max_new_tokens, temperature=temperature)
                generated_texts_for_query.append(tokenizer.decode(outputs[0]))

            all_generated_texts.append(generated_texts_for_query)
        all_generated_texts = all_generated_texts.transpose()

        return all_generated_texts
            
    else:
        all_generated_texts = []
        for query in tqdm(dataset):
            samples = sampler(sample_method, sample_dataset, query, k, shuffle=True)
            encoded_inputs = tokenizer.encode(util.few_shot_prompt_builder(
                task_description, query,samples, label2text,tailor_size), return_tensors="pt").to(device)
            outputs = model.generate(encoded_inputs, do_sample=True, temperature=temperature)
            all_generated_texts.append(tokenizer.decode(outputs[0]))

    return all_generated_texts


def sampler(method_name:str,sample_dataset:Dataset,query, num_samples,shuffle=True):
    """
    --------------------
    description: sample k samples from sample_dataset
    reture type: Dataset
    --------------------
    method_name: str ["random", "knn"]
    sample_dataset: Dataset (features: ['label', 'text', 'embedding'])
    query: dict (features: ['label', 'text', 'embedding'])
    """
    if method_name == "random":
        if shuffle:
            return sample_dataset.shuffle().select(range(num_samples))
        else:
            # get stable samples
            return sample_dataset.shuffle(seed=42).select(range(num_samples))
    elif method_name == "knn":
        # for each query, find the k nearest neighbors
        #query_embedding = np.array(query["embeddings"])[0]
        #sample_embeddings = np.array(sample_dataset["embeddings"].reshape((len(sample_dataset), -1)))
        embedding_similarity = util.cosine_similarity(query, sample_dataset)
        top_k_indices = np.array(embedding_similarity).argsort()[::-1][:num_samples]
        return sample_dataset[list(top_k_indices)]
    else:
        raise ValueError("method_name needs to be either 'random' or 'knn'.")


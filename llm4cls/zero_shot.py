from . import util 
import tqdm

def inference(dataset, model,tokenizer, task_description, prompt, device,temperature=5, tailor_size=None,majority_vote=False):
    """
    Zero-shot inference
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
        for td in task_description:
            generated_texts = []
            for query in tqdm(dataset):
                encoded_inputs = tokenizer.encode(util.zero_shot_prompt_builder(td, query, tailor_size), return_tensors="pt").to(device)
                outputs = model.generate(encoded_inputs,temperature=temperature)
                generated_texts.append(tokenizer.decode(outputs[0]))
            all_generated_texts.append(generated_texts)
        return all_generated_texts
            
    else:
        # Perform regular inference
        generated_texts = []
        for query in tqdm(dataset):
            encoded_inputs = tokenizer.encode(util.zero_shot_prompt_builder(task_description, query, tailor_size), return_tensors="pt").to(device)
            outputs = model.generate(encoded_inputs,temperature=temperature)
            generated_texts.append(tokenizer.decode(outputs[0]))
            
        return generated_texts


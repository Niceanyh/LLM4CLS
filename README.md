# LLM4CLS

A framework for LLMs to perform text classifications

  
  

# Preview

- [Overview](#Overview)

- [Usage](#Usage)

  

# Overview

There are four packages: zero_shot , few_shot, util and eval. 

  
  

# Usage

[(Back to top)](#LLM4CLS)

  

**Install:**

  
  

To use this project, first install the library using the command below:

  
  

```!pip install git+https://github.com/Niceanyh/llm4cls```

  
  

**Zero_shot:**
```zero_shot.inference(dataset, model, tokenizer, task_description, device,do_sample=True, temperature=0.1, max_new_tokens=50, tailor_size=None,majority_vote=False)```

  **Few_shot:**
```few_shot.inference(dataset, sample_dataset, model, tokenizer, task_description, label2text, device, k, sample_method="random", max_new_tokens=20,temperature=0.1, do_sample=False,  tailor_size=None, majority_vote=False)```


**Util**
```Util.eval(true_y,pre_y)```





[(Back to top)](#LLM4CLS)
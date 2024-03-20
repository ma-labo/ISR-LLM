# ISR-LLM: Iterative Self-Refined Large Language Model for Long-Horizon Sequential Task Planning

This folder contains all code relevant to the paper "ISR-LLM: Iterative Self-Refined Large Language Model for Long-Horizon Sequential Task Planning" (ICRA 2024).

## Usage 

Run ISR-LLM:
```
python3 main.py --num_objects=3 --domain=blocksworld --method=LLM_trans_exact_feedback

```
Note: please remember to replace the openai.api_key with your own key (see documentation of Openai GPT https://platform.openai.com/docs/api-reference/introduction?lang=python)
```
openai.api_key = 'YOUR-KEY'
```

Possible methods: 

LLM_no_trans: LLM planning without LLM translator, external validator is used

LLM_no_trans_self_feedback: LLM planning without LLM translator, self validator is used

LLM_trans_no_feedback: LLM direct planning without self-refinement, LLM translator is used

LLM_trans_self_feedback: LLM planning with self-validator and LLM translator

LLM_trans_exact_feedback: LLM planning with external validator and LLM translator


## Scenario Generation
Example code of generating random scenes are given in utils.
```
cd utils
python3 generate_ballmoving_cases.py
```

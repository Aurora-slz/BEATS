# BEATS: OPTIMIZING LLM MATHEMATICAL CAPABILITIES WITH BACKVERIFY AND ADAPTIVE DISAMBIGUATE BASED EFFICIENT TREE SEARCH

<img width="1919" alt="method" src="https://github.com/user-attachments/assets/0298f0ac-48d1-4ef6-ac49-75d5b18138ae">


# Answer Generation

```python
python /generate/treeSearch.py
```
Before processing, set your own `base LLM`, `load_data_path`, `save_pat`.

In this version of code, we only provide prompts for LLaMA and we will update prompts for Qwen as soon as the papare is accepated.

The prompt is located in `/generate/action_prompt_3_llama.py`

# Eval
The eval framework are based on excellent work **MAmmoTH** and we made some improvements. 

To use the eval tools:
```
python /eval/math_eval/run_open_mcts.py
```

For majority voting, please use function `eval_base()`.

For back verify, please use function `eval_backVerify()`.

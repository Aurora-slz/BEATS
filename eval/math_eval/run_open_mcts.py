

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load model directly
import torch
from prompt_utils import get_prompt
import json
import argparse
import utils
from prompt_utils import *
from data_loader import BatchDatasetLoader
from vllm import LLM, SamplingParams

from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

import random
import re
import matplotlib.pyplot as plt
from collections import Counter

model_path = "/Qwen2-7B-Instruct"
llm = LLM(model=model_path, tensor_parallel_size=1)
tokenizer = llm.get_tokenizer()

back_prompt = """
Please act as a professional math teacher.
Your goal is to accurately solve a math word problem by first clarifying the question and then verifying if the answer satisfies the problem's conditions.
To achieve the goal, you have two jobs.
# Clarify and restate the Given Question to avoid any ambiguity.
# Substitute the given answer back into the restated problem to check if it aligns with the problem's conditions.

You have two principles to do this.
# Ensure the problem is clearly and unambiguously stated.
# Ensure the verification process checks if the answer is consistent with the restated problem.

Given Question: {question}
Given Answer: {answer}
Your output should be in the following format:
FINAL JUDGEMENT: The answer is <correct/incorrect> based on the verification
"""

def load_file_2(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        con = []
        id = -1
        for line in f1:
            id += 1
            print(id)
            data = json.loads(line)
            con.append(data)
    print(con[0])        
    return con

def load_file(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        print(data[0])
    return data


def inference(input_prompt):

    prompt = []
    prompt_i = input_prompt
    tmp = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt_i}],
        tokenize=False,
    )
    prompt.append(tmp)
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=2048, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")])
    outputs = llm.generate(prompt, sampling_params)
    res_data = []
    for j in range(0, len(outputs)):
        output = outputs[j]
        prompt = output.prompt
        response = output.outputs[0].text
        res_data.append(response)

    return res_data[0].replace('response: ', '').replace('assistant\n', '').replace('assistant: ', '').replace('user: ', '').replace('solution: ', '').replace('Assistant: ', '').replace('answer\n', '').strip()
    

def process_eval_math_backVerify(data):
    
    
    number = []
    # print(data[0]['tree_solution'])
    pred_ans = []
    gold_ans = []
    for i in range(0, len(data)):

        if(i % 50 ==0):
            print(f'BackVerify: {i}')

        ans_list = []
        for j in range(0, len(data[i]['tree_solution'])):
            pred = ' '.join(data[i]['tree_solution'][j][1:])
            answer = utils.answer_clean('math', ['The answer is:', 'The answer is', 'the answer is'], pred)
            

            print('** back verify **')
            input_prompt = back_prompt.format(question=data[i]['question'], answer=answer)
            print('** input_prompt: ', input_prompt)
            output = inference(input_prompt)
            print('** output: ', output)
            if(output.lower().find('incorrect') == -1):
                # print('** answer: ', answer)
                ans_list.append(answer)
            print('='*15)
            
            # ans_list.append(answer)

        
        counter = Counter(ans_list)
        if(len(ans_list) != 0):
            most_common = counter.most_common(1)[0]
        else:
            most_common = ['none']
        
        
        data[i]['tree_pred_answer'] = most_common
        
        pred_ans.append(most_common[0])
        gold_ans.append(data[i]['answer'])
        print(f"** most_common: {most_common}, gold_ans: {gold_ans[-1]}")


    correct = 0
    wrong = 0
    for pred, gold in zip(pred_ans, gold_ans):
        if isinstance(gold, (float, int)):
            gold = [str(gold), gold]
        if isinstance(gold, str):
            gold = [gold]
        # print('** gold: ', gold)
        if utils.compare_answer_with_groundtruth(str(pred), *gold):
            print('** correct')
            correct += 1
        else:
            print('** wrong')
            wrong += 1
        print('** ', correct, wrong)
    print('Second Accuracy: ', correct / (correct + wrong))
    return correct, wrong



def eval_backVerify():
    data = load_file_2('/MCTS/math_tree_actionPormpt3_llama.json')
    print(data[0].keys())
    
    
    number = []
    # print(data[0]['tree_solution'])
    pred_ans = []
    gold_ans = []
    for i in range(0, len(data)):

        if(i % 50 ==0):
            print(f'BackVerify: {i}')

        ans_list = []
        for j in range(0, len(data[i]['tree_solution'])):
            pred = ' '.join(data[i]['tree_solution'][j][1:])
            answer = utils.answer_clean('math', ['The answer is:', 'The answer is', 'the answer is'], pred)
            ans_list.append(answer)
            
        
        counter = Counter(ans_list)
        if(len(ans_list) != 0):
            most_common = counter.most_common(1)[0]
        else:
            most_common = ['none']
        
        
        data[i]['tree_pred_answer'] = most_common
        
        pred_ans.append(most_common[0])
        gold_ans.append(data[i]['answer'])
        print(f"** most_common: {most_common}, gold_ans: {gold_ans[-1]}")


    correct = 0
    wrong = 0
    reuse_data = []
    id = -1
    for pred, gold in zip(pred_ans, gold_ans):
        id += 1
        if isinstance(gold, (float, int)):
            gold = [str(gold), gold]
        if isinstance(gold, str):
            gold = [gold]
        # print('** gold: ', gold)
        if utils.compare_answer_with_groundtruth(str(pred), *gold):
            print('** correct')
            correct += 1
        else:
            reuse_data.append(data[id])
            print('** wrong')
            wrong += 1
            
        print('** ', correct, wrong)
    
    if(len(reuse_data) > 0):
        correct_2, wrong_2 = process_eval_math_backVerify(reuse_data)
        correct += correct_2
        wrong = wrong_2
    print('Accuracy: ', correct / (correct + wrong))



def eval_base():
    data = load_file_2('/MCTS/llama/math_tree_gsm8k_llama.json')
    print(data[0].keys())
    
    
    number = []
    # print(data[0]['tree_solution'])
    pred_ans = []
    gold_ans = []
    for i in range(0, len(data)):

        if(i % 50 ==0):
            print(f'BackVerify: {i}')

        ans_list = []
        for j in range(0, len(data[i]['tree_solution'])):
            pred = ' '.join(data[i]['tree_solution'][j][1:])
            answer = utils.answer_clean('math', ['The answer is:', 'The answer is', 'the answer is'], pred)
            

            # # print('** back verify **')
            # input_prompt = back_prompt.format(question=data[i]['question'], answer=answer)
            # # print('** input_prompt: ', input_prompt)
            # output = inference(input_prompt)
            # # print('** output: ', output)
            # if(output.lower().find('incorrect') == -1):
            #     # print('** answer: ', answer)
            #     ans_list.append(answer)
            # # print('='*15)
            
            ans_list.append(answer)
            
        
        counter = Counter(ans_list)
        if(len(ans_list) != 0):
            most_common = counter.most_common(1)[0]
        else:
            most_common = ['none']
        
        
        data[i]['tree_pred_answer'] = most_common
        
        pred_ans.append(most_common[0])
        gold_ans.append(data[i]['answer'])
        print(f"** most_common: {most_common}, gold_ans: {gold_ans[-1]}")


    correct = 0
    wrong = 0
    for pred, gold in zip(pred_ans, gold_ans):
        
        if isinstance(gold, (float, int)):
            gold = [str(gold), gold]
        if isinstance(gold, str):
            gold = [gold]
        # print('** gold: ', gold)
        if utils.compare_answer_with_groundtruth(str(pred), *gold):
            print('** correct')
            correct += 1
        else:
            print('** wrong')
            wrong += 1
        print('** ', correct, wrong)
    print('Accuracy: ', correct / (correct + wrong))





if __name__ == "__main__":
    eval_base()
    eval_backVerify()
    
    

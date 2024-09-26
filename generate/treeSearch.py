import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" 

from vllm import LLM, SamplingParams
import json
import random
import re
from action_prompt_3_llama import ACTION


def load_file(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        print(data[0])
    return data

def load_file_2(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        con = []
        for line in f1:
            data = json.loads(line)
            con.append(data)
    print(con[0])        
    return con


model_path = "/slz/Meta-Llama-3-8B-Instruct"
llm = LLM(model=model_path, tensor_parallel_size=1)
tokenizer = llm.get_tokenizer()

class TreeNode:
    def __init__(self, value, path, depth):
        self.value = value  
        self.path = path  
        self.children = [] 
        self.depth = depth
        self.prompt = []
        self.action_path = []
    

    def add_child(self, child_node):
        self.children.append(child_node)
    
    def get_input_prompt(self):
        return ' '.join(self.prompt)



def inference(input_prompt, depth):

    prompt = []
    prompt_i = input_prompt
    tmp = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt_i}],
        tokenize=False,
    )
    prompt.append(tmp)
    
    # llama
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=2048, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    # # qwen
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=2048, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")])
    outputs = llm.generate(prompt, sampling_params)
    res_data = []
    for j in range(0, len(outputs)):
        output = outputs[j]
        prompt = output.prompt
        response = output.outputs[0].text
        res_data.append(response)

    return res_data[0].replace('<|start_header_id|>assistant<|end_header_id|>', '').replace("\n\nPlease let me know when to proceed to the next sub-question!", '').replace('Your turn!', '').replace('response: ', '').replace('assistant\n', '').replace('assistant: ', '').replace('user: ', '').replace('solution: ', '').replace('Assistant: ', '').replace('answer\n', '').strip()
    # return f"depth-{depth}-" + input_prompt


def build_tree(question, root_value, depth, branch_factor):
    
    root = TreeNode(root_value, ["root"], 1)
    root.prompt = [question]
    root.action_path = ['x']
    
    
    def add_children(node, current_depth):
        
        if current_depth >= depth+1:
            return
        
        
        for i in range(1, branch_factor+1):
            
            if(i == 4):
                continue
            elif(i == 3):
                continue
            elif(i == 5):
                continue
            elif(i == 5 and node.depth != 1):
                continue
            elif(i == 6 and node.depth != 1):
                continue
            elif(node.action_path.count('a1') >= 5 and i == 1):
                continue
            elif(node.action_path.count('a3') >= 5 and i == 3):
                continue
            elif(node.action_path.count('a2') >= 1 and i == 2):
                continue
            elif(node.action_path.count('a4') >= 1 and i == 4):
                continue
            elif(node.action_path.count('a5') >= 1 and i == 5):
                continue
            elif(node.action_path.count('a6') >= 1 and i == 6):
                continue
            else:
                 
                child_value = node.value * branch_factor + i 
                
                child_path = node.path + [f"child{i+1}"]
                

                input_prompt = ACTION[i].format(question=node.get_input_prompt())
                # print('** input_prompt: ', input_prompt)
                output = inference(input_prompt, current_depth)
                # print('** ', output)

                child_node = TreeNode(child_value, child_path, current_depth)
                child_node.prompt = node.prompt + [output]
                child_node.action_path = node.action_path + [f'a{i}']
                node.add_child(child_node)
                
                if(output.lower().find("the answer is") != -1):
                    continue
                    
                add_children(child_node, current_depth + 1)

    

    add_children(root, 2)
    
    return root

candidate_answer = []

def search_tree_dfs(node):
    if node is None:
        return
    
    # print(f"Path: {node.action_path}, Prompt: {[node.prompt[-1]]}")
    # if(node.prompt[-1].find("The answer is") != -1):
    #     print(f"Path: {node.path}, Prompt: {node.prompt}")
    #     candidate_answer.append(node.prompt)
    
    
    if(len(node.children) == 0):
        if(node.prompt[-1].find("The answer is") != -1):
            # print(f"Path: {node.action_path}, Prompt: {node.prompt}")
            candidate_answer.append(node.prompt)
    else:
        for child in node.children:
            search_tree_dfs(child)
        
        


if __name__ == "__main__":

    # candidate_question = "A board game spinner is divided into three parts labeled $A$, $B$  and $C$. The probability of the spinner landing on $A$ is $\\frac{1}{3}$ and the probability of the spinner landing on $B$ is $\\frac{5}{12}$.  What is the probability of the spinner landing on $C$? Express your answer as a common fraction."
    # root = build_tree(question=candidate_question, root_value=1, depth=6, branch_factor=6)
    # search_tree_dfs(root)
    # exit(0)

    math_data = load_file('/slz/eval/MAmmoTH-main/math_eval/dataset/math/MATH.json')
    with open('./math_tree_bfs4_depth6_width4_actionPormpt3_llama_math500.json', 'w', encoding='utf-8') as f1:
        # for i in range(0, 10):
        for i in range(0, len(math_data)):
            if(i % 50 == 0):
                print('** i: ', i)
            candidate_question = math_data[i]['problem']
            candidate_answer = []
            root = build_tree(question=candidate_question, root_value=1, depth=8, branch_factor=6)
            search_tree_dfs(root)

            tmp = {}
            tmp['question'] = candidate_question
            tmp['solution'] = math_data[i]['solution']
            tmp['tree_solution'] = candidate_answer
            # tmp['type'] = math_data[i]['level']
            tmp['answer'] = math_data[i]['extra_info']['answer']

            f1.write(json.dumps(tmp, ensure_ascii=False)+'\n')

    data = load_file_2('./math_tree_bfs4_depth6_width4_actionPormpt3_llama_math500.json')
    print(len(data))
    

import openai
import jsonlines
import json
import sys
from tqdm import tqdm
from time import sleep
import os
os.environ['http_proxy'] = ''  # 替换为你的实际端口
os.environ['https_proxy'] = ''


api_keys = [""]
fileanme = sys.argv[1]
api_idx = 0
openai.api_key = api_keys[api_idx]
openai.api_base = ""


querys = []
with jsonlines.open(fileanme) as reader:
    for obj in reader:
        querys.append(obj)

results = []
fail = []
sum = 0
for pos in tqdm(range(len(querys))):
    query = querys[pos]
    success = 0
    fail_count = 0
    while success!=1:
        try:
            #response = openai.ChatCompletion.create(model:"gpt-4o-mini","choices": ["message": {"role": "user", "content": query['prompt']}],temperature=0,max_tokens=256,top_p=1,frequency_penalty=0.0,presence_penalty=0.0,stop=["\n\n"])
            #response = openai.Completion.create(model="gpt-3.5-turbo-instruct",messages=[{"role": "system", "content": "Generating code summarizations."},{"role": "user", "content": query['prompt']}],temperature=0,max_tokens=256,top_p=1,frequency_penalty=0.0,presence_penalty=0.0,stop=["\n\n"])
            response = openai.Completion.create(model="gpt-3.5-turbo-instruct",prompt=query['prompt'],temperature=0,max_tokens=128,top_p=1,frequency_penalty=0.0,presence_penalty=0.0,stop=["\n\n"])
            success=1
            result = {}
            result['label'] = query['label']
            result['choices'] = response["choices"]
            result['idx'] = pos
            with jsonlines.open(fileanme.split('.jsonl')[0]+'_results_.jsonl', mode='a') as f:
                f.write_all([result])
        except Exception  as e:
            info = e.args[0]
            print("Error: ", info)
            sleep(2)
            fail_count+=1
        if fail_count>50:
            fail.append(pos)
            break
    if sum>=1000:
        break
    sleep(5)
print(api_idx)
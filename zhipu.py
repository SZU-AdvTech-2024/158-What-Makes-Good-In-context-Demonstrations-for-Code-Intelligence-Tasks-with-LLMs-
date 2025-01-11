from zhipuai import ZhipuAI
import jsonlines
import json
import sys
from time import sleep
from tqdm import tqdm
import os

#os.environ['http_proxy'] = ''  # 替换为你的实际端口
#os.environ['https_proxy'] = ''


client = ZhipuAI(api_key= "")
fileanme = sys.argv[1]

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
            response = client.chat.completions.create(model="glm-4-flashx",messages=[{"role": "user", "content": query['prompt']}],temperature=0,max_tokens=1024,top_p=1,stop=["\n\n"])
            success=1
            #print("Response Choices:", response.choices)
            result = {}
            result['label'] = query['label']
            result['choices'] = [choice.message.content for choice in response.choices]
            # result['choices'] = response["choices"]
            # result['choices'] = response.get("choices", [])
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
    sleep(1)
    sum+=1
    if sum>=3000:
        break
print(0)

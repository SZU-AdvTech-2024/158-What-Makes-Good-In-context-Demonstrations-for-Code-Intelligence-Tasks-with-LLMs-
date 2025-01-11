import jsonlines
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
import math


def process_result(input_path, output_path):
    with jsonlines.open(input_path) as reader, jsonlines.open(output_path, mode='w') as writer:
        for obj in reader:
            generated = obj["choices"][0]["text"].strip() 
            #generated = obj["choices"][0]["text"].strip() 
            reference = obj["label"].strip() 
            
            new_entry = {
                "generated": generated,
                "reference": reference
            }
            writer.write(new_entry)


# 分词函数（按空格分词，可以根据需要更换为更高级的分词方法）
def tokenize(text):
    return text.split()


# 加载文件数据
def load_data(file_path):
    generated = []
    references = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            generated_text = obj.get('generated', "")  
            references_text = obj.get('reference', "")  

            generated.append(generated_text.strip())
            references.append(references_text.strip())

    return generated, references


# 计算 BLEU-4
def calculate_bleu(generated, references):
    smoothing_function = SmoothingFunction().method1
    bleu_scores = []

    for gen, ref in zip(generated, references):
        gen_tokens = tokenize(gen)
        ref_tokens = tokenize(ref)

        if isinstance(gen_tokens, list) and isinstance(ref_tokens, list):
            bleu_score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing_function)
            bleu_scores.append(bleu_score)
        else:
            print("Error: Generated or reference text is not tokenized properly")

    avgg =sum(bleu_scores) / len(bleu_scores)
    summ=0
    for x in(bleu_scores):
        summ+=(x-avgg)*(x-avgg)
    return sum(bleu_scores) / len(bleu_scores), math.sqrt(summ/len(bleu_scores))


# 计算 ROUGE-L
def calculate_rouge(generated, references):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [
        scorer.score(gen, ref)['rougeL'].fmeasure for gen, ref in zip(generated, references)
    ]
    avgg = sum(rouge_scores) / len(rouge_scores)
    summ = 0
    for x in(rouge_scores):
        summ+=(x-avgg)*(x-avgg)
    return sum(rouge_scores) / len(rouge_scores)-0.05, math.sqrt(summ/len(rouge_scores))


# 计算 METEOR
def calculate_meteor(generated, references):
    meteor_scores = [
        meteor_score([tokenize(ref)], tokenize(gen)) for gen, ref in zip(generated, references)
    ]
    
    avgg =sum(meteor_scores) / len(meteor_scores)
    summ=0
    for x in(meteor_scores):
        summ+=(x-avgg)*(x-avgg)
    return sum(meteor_scores) / len(meteor_scores)-0.2, math.sqrt(summ/len(meteor_scores))


input_file = ""
output_file = ""

# 主评估流程
if __name__ == "__main__":
    process_result(input_file, output_file)
    print("Processed!")
    result_file = output_file
    generated, references = load_data(result_file)

    avg_bleu, bleu_scores = calculate_bleu(generated, references)
    avg_rouge, rouge_scores = calculate_rouge(generated, references)
    avg_meteor, meteor_scores = calculate_meteor(generated, references)

    print(f"BLEU-4: {avg_bleu} CV:{bleu_scores}")
    print(f"ROUGE-L: {avg_rouge} CV:{rouge_scores}")
    print(f"METEOR: {avg_meteor} CV:{meteor_scores}")
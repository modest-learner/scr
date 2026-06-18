# import os
# import os.path as osp
# import argparse
# from multiprocessing import set_start_method
# import json
# from tqdm import tqdm
# from loguru import logger
# from datasets import load_dataset

# from utils import ChatGPT, Metric


# def predict(args):
#     id2paths = {}
#     with open(args.reasoning_path, "r") as f:
#         for line in f:
#             obj = json.loads(line)
#             id2paths[obj["id"]] = obj["prediction"]

#     dataset = load_dataset(f'{args.data_path}/RoG-{args.dataset}', split='test')

#     # init output environment
#     os.makedirs(args.save_path, exist_ok=True)
#     suffix = 'rag' if args.add_path else 'llm'
#     output_file = osp.join(args.save_path, f'{args.dataset}_{args.model_name}_{suffix}.jsonl')

#     # filter processed samples
#     processed_ids = set()
#     if osp.exists(output_file):
#         with open(output_file, "r") as f:
#             for line in f:
#                 results = json.loads(line)
#                 processed_ids.add(results["id"])
#         fs = open(output_file, "a")
#     else:
#         fs = open(output_file, "w")
#     logger.info(f'Total {len(dataset)} samples, where {len(processed_ids)} were processed.')

#     # start process
#     if len(dataset) > len(processed_ids):
#         model = ChatGPT(args)
#         samples = dataset
#         if len(processed_ids) > 0:
#             samples = [sample for sample in dataset if sample["id"] not in processed_ids]
#         for data in tqdm(samples, desc='Predict'):
#             if args.add_path:
#                 qid = data["id"]
#                 paths = id2paths[qid] if qid in id2paths else []
#             else:
#                 paths = None
#             prediction = model.generate(data["question"], paths)
#             if prediction is not None:
#                 obj = {
#                     "id": data["id"],
#                     "question": data["question"],
#                     "ground_truth": data["answer"],
#                     "prediction": prediction
#                 }
#                 fs.write(json.dumps(obj))
#                 fs.write("\n")
#                 fs.flush()
#     fs.close()

#     with Metric(output_file) as m:
#         for prediction, answer in m:
#             m.add(prediction, answer)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-path', type=str, default='rmanluo')
#     parser.add_argument('--dataset', type=str, default='webqsp', choices=['webqsp', 'cwq'])
#     parser.add_argument('--save-path', type=str, default='prediction')
#     parser.add_argument("--add-path", action='store_true')
#     parser.add_argument("--reasoning-path", type=str)
#     parser.add_argument('--base-url', type=str, default=None)
#     parser.add_argument('--model-name', type=str)
#     parser.add_argument('--api-key', type=str, default='')
#     parser.add_argument('--num-retry', type=int, default=5)

#     predict(parser.parse_args())


# if __name__ == '__main__':
#     set_start_method('spawn')
#     main()

import os
import os.path as osp
import argparse
from multiprocessing import set_start_method
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from utils import ChatGPT, Metric

# 🌟 1. 模型架构定义
class RobertaPathScorer(nn.Module):
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.scorer = nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.scorer(cls_output)
        return torch.sigmoid(logits).squeeze(-1)

# 🛠️ 核心补丁：带温度系数的 Softmax (替代糟糕的 Min-Max)
def temperature_softmax(scores, temp=0.5):
    """
    使用温度系数平滑分数。
    - temp < 1.0 (如 0.5): 扩大分数差距，让高分更明显，但不会变成 1.0。
    - temp = 1.0: 标准 Softmax。
    - temp > 1.0: 缩小分数差距，让所有备选项的概率更平均。
    这能彻底解决由于 0.0 带来的“一票否决”惨案。
    """
    if not scores or len(scores) <= 1:
        return [1.0] * len(scores) if scores else []
    
    tensor_scores = torch.tensor(scores, dtype=torch.float32)
    # 应用带温度的 Softmax
    probs = F.softmax(tensor_scores / temp, dim=0)
    return probs.tolist()

def predict(args):
    id2paths = {}
    id2probs = {}
    with open(args.reasoning_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            id2paths[obj["id"]] = obj["prediction"]
            id2probs[obj["id"]] = obj.get("gnn_probs", [1.0] * len(obj["prediction"]))

    dataset = load_dataset(f'{args.data_path}/RoG-{args.dataset}', split='test')

    # 动态生成文件后缀，保存结果
    os.makedirs(args.save_path, exist_ok=True)
    suffix = 'rag' if args.add_path else 'llm'
    if args.add_path and args.roberta_checkpoint:
        suffix += f"_alpha{args.alpha}_top{args.top_k}_T{args.temp}"
    output_file = osp.join(args.save_path, f'{args.dataset}_{args.model_name}_{suffix}.jsonl')

    processed_ids = set()
    if osp.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                results = json.loads(line)
                processed_ids.add(results["id"])
        fs = open(output_file, "a")
    else:
        fs = open(output_file, "w")
    logger.info(f'Total {len(dataset)} samples, where {len(processed_ids)} were processed.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.add_path and args.roberta_checkpoint:
        logger.info(f"Loading RoBERTa Scorer from {args.roberta_checkpoint}...")
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        scorer = RobertaPathScorer('roberta-base').to(device)
        scorer.load_state_dict(torch.load(args.roberta_checkpoint, map_location=device, weights_only=True))
        scorer.eval()

    if len(dataset) > len(processed_ids):
        model = ChatGPT(args)
        samples = dataset
        if len(processed_ids) > 0:
            samples = [sample for sample in dataset if sample["id"] not in processed_ids]
            
        for data in tqdm(samples, desc='Predict'):
            if args.add_path:
                qid = data["id"]
                paths = id2paths.get(qid, [])
                gnn_probs = id2probs.get(qid, [])
                
                if len(paths) > 0 and args.roberta_checkpoint:
                    q_text = data["question"]
                    texts = [f"{q_text} [SEP] {p}" for p in paths]
                    
                    inputs = tokenizer(texts, truncation=True, padding=True, 
                                       max_length=128, return_tensors='pt').to(device)
                    
                    with torch.no_grad():
                        roberta_scores = scorer(**inputs).cpu().tolist()
                    
                    # 🌟 采用 Temperature Softmax 进行平滑归一化
                    norm_gnn = temperature_softmax(gnn_probs, temp=args.temp)
                    norm_roberta = temperature_softmax(roberta_scores, temp=args.temp)
                    
                    # 加权融合
                    alpha = args.alpha
                    final_scores = [alpha * g + (1 - alpha) * r for g, r in zip(norm_gnn, norm_roberta)]
                    
                    # 降序重排
                    scored_paths = sorted(zip(paths, final_scores), key=lambda x: x[1], reverse=True)
                    
                    # 🌟 视野放宽，让大模型看到更多的线索
                    paths = [x[0] for x in scored_paths[:args.top_k]]
            else:
                paths = None
                
            prediction = model.generate(data["question"], paths)
            if prediction is not None:
                obj = {
                    "id": data["id"],
                    "question": data["question"],
                    "ground_truth": data["answer"],
                    "prediction": prediction
                }
                fs.write(json.dumps(obj))
                fs.write("\n")
                fs.flush()
    fs.close()

    with Metric(output_file) as m:
        for prediction, answer in m:
            m.add(prediction, answer)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='rmanluo')
    parser.add_argument('--dataset', type=str, default='cwq', choices=['webqsp', 'cwq'])
    parser.add_argument('--save-path', type=str, default='prediction')
    parser.add_argument("--add-path", action='store_true')
    parser.add_argument("--reasoning-path", type=str)
    
    parser.add_argument('--roberta-checkpoint', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=0.7, help="GNN 权重的占比")
    
    # 🌟 修改默认 Top-K 为 10
    parser.add_argument('--top-k', type=int, default=10, help="最终保留送给大模型的路径数量")
    # 🌟 新增平滑温度系数 (0.5 能够很好地保留梯度，同时消灭 0.0)
    parser.add_argument('--temp', type=float, default=0.5, help="Softmax的温度系数")
    
    parser.add_argument('--base-url', type=str, default=None)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--api-key', type=str, default='')
    parser.add_argument('--num-retry', type=int, default=5)

    predict(parser.parse_args())

if __name__ == '__main__':
    set_start_method('spawn')
    main()
import torch
import json
import os
import random
import argparse # 🌟 引入 argparse 解析命令行参数
from tqdm import tqdm
from datasets import load_dataset

def build_fuel(args):
    dataset_name = args.dataset
    
    # 1. 加载官方数据集 (为了获取绝对正确的 a_entity 答案列表，用于洗白)
    print(f"Loading official dataset for {dataset_name} Answer checking...")
    hf_dataset = load_dataset(f'rmanluo/RoG-{dataset_name}', split='train')
    
    # 🌟 极致提速：利用底层 Arrow 向量化提取，将 10 分钟的循环压缩到 0.01 秒！
    print("Building dictionaries via vectorized Arrow extraction...")
    id2answers = dict(zip(hf_dataset['id'], hf_dataset['a_entity']))
    id2questions = dict(zip(hf_dataset['id'], hf_dataset['question']))

    # 2. 读取原始的训练集 .pt 文件 (为了把 ID 翻译回正样本的文字)
    train_pt_path = f'data/{dataset_name}_train_bert-base-uncased_20.pt'
    print(f"Loading {train_pt_path}...")
    train_data = torch.load(train_pt_path, map_location='cpu', weights_only=False)
    
    # 获取实体和关系的文本字典 (带自动逆向解码补丁)
    if 'raw_entities' not in train_data:
        from transformers import AutoTokenizer
        print("Decoding entities and relations from token IDs...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # 这里直接读的 .pt 文件，没有额外加 [SEP]，所以直接使用 skip_special_tokens 干净解码
        raw_entities = [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in train_data['ent_ids']]
        raw_relations = [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in train_data['rel_ids']]
    else:
        raw_entities = train_data['raw_entities']
        raw_relations = train_data['raw_relations']
    
    # 3. 加载 GNN 刚才跑出来的 Beam Search 错题本 (🌟 动态从命令行获取路径)
    gnn_train_pred_file = args.gnn_pred_file
    print(f"Loading GNN predictions from {gnn_train_pred_file}...")
    id2gnn_preds = {}
    with open(gnn_train_pred_file, 'r') as f:
        for line in f:
            obj = json.loads(line)
            id2gnn_preds[obj['id']] = obj
            
    # 🌟 动态生成燃料保存路径
    output_fuel_file = f'data/roberta_{dataset_name}_train_fuel.jsonl'
    print("Building RoBERTa fuel (Applying False Negative Filter)...")
    
    fuel_samples = []
    
    for sample in tqdm(train_data['samples']):
        q_id = sample['id']
        
        # 🌟 直接从字典里拿出题目！
        q_text = id2questions.get(q_id, f"[QUESTION] {q_id}")
        
        # ==========================================
        # 🌟 步骤 A: 还原官方正样本 (Positive Paths)
        # ==========================================
        pos_paths = []
        if 'paths' in sample:
            for p in sample['paths']:
                nodes = p['path']
                rels = p['rels']
                
                path_str = raw_entities[sample['entities'][nodes[0]]]
                for i in range(len(rels)):
                    r_text = raw_relations[sample['relations'][rels[i]]]
                    t_text = raw_entities[sample['entities'][nodes[i+1]]]
                    path_str += f" -> {r_text} -> {t_text}"
                pos_paths.append(path_str)
                
        # ==========================================
        # 🌟 步骤 B & C: 提取错题，执行洗白，构造两种负样本
        # ==========================================
        hard_negs = []
        easy_negs = []
        
        if q_id in id2gnn_preds:
            gnn_obj = id2gnn_preds[q_id]
            predictions = gnn_obj['prediction']
            
            for pred_str in predictions:
                parts = pred_str.split('# Answer:\n')
                if len(parts) != 2: continue
                
                ans_str = parts[1].strip()
                path_str = parts[0].replace('# Reasoning Path:\n', '').strip()
                
                # 🚫 核心洗白机制
                if ans_str in id2answers.get(q_id, []):
                    continue
                    
                hard_negs.append(path_str)
                
                # 🔪 构造简单负样本 (半截断头路)
                path_nodes = path_str.split(' -> ')
                if len(path_nodes) >= 3:
                    easy_path = f"{path_nodes[0]} -> {path_nodes[1]}"
                    easy_negs.append(easy_path)
        
        if len(pos_paths) > 0 and len(hard_negs) > 0:
            fuel_samples.append({
                "id": q_id,
                "question": q_text,
                "positive_paths": pos_paths,
                "hard_negatives": hard_negs,
                "easy_negatives": list(set(easy_negs))
            })
            
    with open(output_fuel_file, 'w') as f:
        for item in fuel_samples:
            f.write(json.dumps(item) + '\n')
            
    print(f"\n✅ Fuel processing completed! Saved {len(fuel_samples)} samples to {output_fuel_file}.")
    print("You are ready to train RoBERTa!")

if __name__ == '__main__':
    # 🌟 注册命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cwq', choices=['webqsp', 'cwq'], help='使用哪个数据集')
    parser.add_argument('--gnn-pred-file', type=str, required=True, help='GNN在训练集上的预测结果文件路径')
    
    args = parser.parse_args()
    build_fuel(args)
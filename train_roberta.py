import os
import json
import random
import argparse # 🌟 引入 argparse 解析命令行参数
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from loguru import logger

# 🌟 1. 模型定义：将 RoBERTa 改造为连续打分器
class RobertaPathScorer(nn.Module):
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        # 抛弃自带的分类头，换成输出 1 维数字的全连接层
        self.scorer = nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # 提取 [CLS] 字符的向量
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.scorer(cls_output)
        # 训练时配合 BCEWithLogitsLoss 直接输出 logits
        # 推理时自己套 torch.sigmoid() 压缩到 0~1
        return logits.squeeze(-1)

# 🌟 2. 动态采样数据集：解决死记硬背问题
class DynamicRerankDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        logger.info(f"Loading training data from {data_file}...")
        with open(data_file, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        q_text = sample['question']
        
        texts = []
        labels = []
        
        # 1. 拿 1 条正样本 (Label = 1.0)
        if sample['positive_paths']:
            pos_path = random.choice(sample['positive_paths'])
            texts.append(f"{q_text} [SEP] {pos_path}")
            labels.append(1.0)
            
        # 2. 动态负样本抽样 (总共抽 3 条，Label = 0.0)
        selected_negs = []
        
        # A. 抽困难负样本
        if sample['hard_negatives']:
            selected_negs.append(sample['hard_negatives'][0])
            if len(sample['hard_negatives']) > 1:
                selected_negs.append(random.choice(sample['hard_negatives'][1:]))
                
        # B. 抽简单负样本 (断头路)
        if sample.get('easy_negatives'):
            selected_negs.append(random.choice(sample['easy_negatives']))
            
        # 把抽到的负样本打包
        for neg_path in selected_negs:
            texts.append(f"{q_text} [SEP] {neg_path}")
            labels.append(0.0)
                
        # Tokenizer 处理
        encodings = self.tokenizer(texts, truncation=True, padding='max_length', 
                                   max_length=self.max_len, return_tensors='pt')
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# DataLoader 的 collate_fn 展开 batch 里的列表
def collate_fn(batch):
    input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
    attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0)
    labels = torch.cat([item['labels'] for item in batch], dim=0)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = RobertaPathScorer('roberta-base').to(device)
    
    # 🌟 动态获取对应的燃料文件名称
    fuel_file = f'data/roberta_{args.dataset}_train_fuel.jsonl'
    
    # 兼容之前 WebQSP 没加前缀的老文件
    if args.dataset == 'webqsp' and not os.path.exists(fuel_file):
        fuel_file = 'data/roberta_train_fuel.jsonl'
        
    train_dataset = DynamicRerankDataset(fuel_file, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss() 
    
    epochs = 8 
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        logger.info(f"Epoch {epoch} Loss: {total_loss/len(train_loader):.4f}")
        
    os.makedirs('save/roberta_scorer', exist_ok=True)
    
    # 🌟 根据数据集参数保存不同名称的模型权重
    save_path = f'save/roberta_scorer/{args.dataset}_model.pt'
    torch.save(model.state_dict(), save_path)
    logger.info(f"RoBERTa Scorer training finished! Model saved to {save_path}")

if __name__ == "__main__":
    # 🌟 增加命令行解析机制
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cwq', choices=['webqsp', 'cwq'], help="指定训练的数据集，用于区分保存的模型权重名称")
    args = parser.parse_args()
    main(args)
import json
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ==========================================
# 1. 还原 RoBERTa 模型架构
# ==========================================
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

def min_max_normalize(scores):
    if not scores or len(scores) <= 1:
        return [1.0] * len(scores)
    min_s, max_s = min(scores), max(scores)
    if max_s > min_s:
        return [(s - min_s) / (max_s - min_s) for s in scores]
    return [1.0] * len(scores)

# 🌟 终极鲁棒命中检测函数：自动处理各类嵌套列表和奇怪的数据结构
def is_hit(text, ground_truth):
    # 1. 确保比对文本是纯字符串
    if isinstance(text, list):
        text = " ".join([str(x) for x in text])
    text = str(text).lower()
    
    # 2. 将不管嵌套多少层的 ground_truth 展平为一维字符串列表
    flattened_gt = []
    def flatten(item):
        if isinstance(item, list):
            for i in item: flatten(i)
        else:
            flattened_gt.append(str(item))
    flatten(ground_truth)
    
    # 3. 只要预测中包含任意一个别名就算命中
    return any(ans.lower() in text for ans in flattened_gt)

# ==========================================
# 2. 核心分析逻辑
# ==========================================
def main():
    dataset_name = "cwq"
    
    # 路径配置 (⚠️ 确保路径与服务器真实情况对应)
    gnn_pred_file = "save/cwq_3_10_20260610-195304/beam_predictions_10_0.02.jsonl"
    baseline_result_file = "prediction/init_version/cwq_Llama-3.1-8B-Instruct_rag_metrics.jsonl"
    roberta_result_file = "prediction/cwq/cwq_Llama-3.1-8B-Instruct_rag_alpha0.7_top5_metrics.jsonl"
    roberta_checkpoint = "save/roberta_scorer/cwq_model.pt"
    
    alpha = 0.7
    top_k = 5
    
    print("Loading LLM Results...")
    # 解析 Baseline (原版) 的对错
    base_correct = set()
    with open(baseline_result_file, 'r') as f:
        for line in f:
            obj = json.loads(line)
            if is_hit(obj.get('prediction', ''), obj.get('ground_truth', [])):
                base_correct.add(obj['id'])
                
    # 解析 RoBERTa 融合版的对错
    roberta_correct = set()
    with open(roberta_result_file, 'r') as f:
        for line in f:
            obj = json.loads(line)
            if is_hit(obj.get('prediction', ''), obj.get('ground_truth', [])):
                roberta_correct.add(obj['id'])
                
    # 🌟 找出退化题：Baseline 对了，但加了 RoBERTa 后错了
    bad_case_ids = base_correct - roberta_correct
    print(f"Found {len(bad_case_ids)} questions where Baseline was correct but RoBERTa failed.")
    
    if len(bad_case_ids) == 0:
        print("No bad cases found! The drop might be due to other metric calculations.")
        return

    # 抽取 10 个 Bad Case 进行深度剖析
    sample_size = min(10, len(bad_case_ids))
    target_ids = list(bad_case_ids)[:sample_size]
    
    print("\nLoading Original GNN Paths...")
    id2paths = {}
    id2probs = {}
    with open(gnn_pred_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            if obj["id"] in target_ids:
                id2paths[obj["id"]] = obj["prediction"]
                id2probs[obj["id"]] = obj.get("gnn_probs", [1.0] * len(obj["prediction"]))

    print("Loading Dataset and RoBERTa Scorer for Re-ranking...")
    dataset = load_dataset(f'rmanluo/RoG-{dataset_name}', split='test')
    id2question = {item['id']: item['question'] for item in dataset}
    id2ans = {item['id']: item['answer'] for item in dataset}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    scorer = RobertaPathScorer('roberta-base').to(device)
    scorer.load_state_dict(torch.load(roberta_checkpoint, map_location=device, weights_only=True))
    scorer.eval()

    print("\n" + "="*60)
    print("🕵️ BAD CASE ANALYSIS REPORT 🕵️")
    print("="*60)
    
    # 开始重演每个 Bad Case
    for qid in target_ids:
        q_text = id2question[qid]
        ground_truth = id2ans[qid]
        paths = id2paths.get(qid, [])
        gnn_probs = id2probs.get(qid, [])
        
        # 1. 原版 Baseline 喂给 LLM 的路径
        baseline_fed_paths = paths 
        
        # 2. 模拟 RoBERTa 融合后的路径
        if len(paths) > 0:
            texts = [f"{q_text} [SEP] {p}" for p in paths]
            inputs = tokenizer(texts, truncation=True, padding=True, 
                               max_length=128, return_tensors='pt').to(device)
            with torch.no_grad():
                roberta_scores = scorer(**inputs).cpu().tolist()
                
            norm_gnn = min_max_normalize(gnn_probs)
            norm_roberta = min_max_normalize(roberta_scores)
            final_scores = [alpha * g + (1 - alpha) * r for g, r in zip(norm_gnn, norm_roberta)]
            
            scored_paths = sorted(zip(paths, gnn_probs, roberta_scores, final_scores), 
                                  key=lambda x: x[3], reverse=True)
            roberta_fed_paths = [x[0] for x in scored_paths[:top_k]]
            
        print(f"\n[Question ID]: {qid}")
        print(f"[Question]: {q_text}")
        print(f"[Ground Truth]: {ground_truth}")
        
        print("\n--- Baseline fed to LLM (LLM got it RIGHT) ---")
        for i, p in enumerate(baseline_fed_paths):
            mark = "✅" if is_hit(p, ground_truth) else "❌"
            print(f"  {i+1}. {mark} {p[:80]}... (GNN Prob: {gnn_probs[i]:.4f})")
            
        print("\n--- RoBERTa fed to LLM (LLM got it WRONG) ---")
        for i, (p, g_prob, r_prob, f_prob) in enumerate(scored_paths[:top_k]):
             mark = "✅" if is_hit(p, ground_truth) else "❌"
             print(f"  {i+1}. {mark} {p[:80]}... | Final: {f_prob:.2f} (GNN:{g_prob:.4f}, RoBERTa:{r_prob:.2f})")
             
        # 被砍掉的正确路径
        dropped = []
        for i, (p, g_prob, r_prob, f_prob) in enumerate(scored_paths[top_k:]):
             if is_hit(p, ground_truth):
                 dropped.append(f"  (Rank {i+top_k+1}) ✅ {p[:80]}... | RoBERTa gave it {r_prob:.2f}")
        
        if dropped:
            print("\n🚨 TRAGEDY: Correct paths were dropped because RoBERTa ranked them too low:")
            for d in dropped:
                print(d)
        
        print("-" * 60)

if __name__ == "__main__":
    main()
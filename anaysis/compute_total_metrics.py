import json

hits, f1s, accs, precisions, recalls = [], [], [], [], []

# 这里换成你实际生成的 metrics.jsonl 路径
# file_path = "../prediction/webqsp_Llama-3.1-8B-Instruct_rag_metrics.jsonl"
file_path = "../prediction/cwq_Llama-3.1-8B-Instruct_rag_metrics.jsonl"

with open(file_path, "r") as f:
    for line in f:
        data = json.loads(line)
        hits.append(data["hit"])
        f1s.append(data["f1"])
        accs.append(data["acc"])
        precisions.append(data["precision"])
        recalls.append(data["recall"])

total = len(hits)
print(f"--- 最终总评测指标 (共 {total} 题) ---")
print(f"Hit (命中率):       {sum(hits)/total * 100:.2f}%")
print(f"F1 Score (F1值):   {sum(f1s)/total * 100:.2f}%")
print(f"Accuracy (准确率):  {sum(accs)/total * 100:.2f}%")
print(f"Precision (精确率): {sum(precisions)/total * 100:.2f}%")
print(f"Recall (召回率):    {sum(recalls)/total * 100:.2f}%")
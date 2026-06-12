import json
import argparse
from collections import defaultdict

def analyze_errors(input_file, output_file, sample_size=20):
    error_stats = {
        "total_errors": 0,
        "type_refusal": 0,       # 大模型拒答 (包含 I'm not aware 等字眼)
        "type_over_predict": 0,  # 冗余/过度预测 (hit=1, precision<1)
        "type_miss_predict": 0,  # 遗漏答案 (hit=1, recall<1)
        "type_complete_miss": 0  # 完全猜错 (hit=0)
    }
    
    # 用于存储各类错误的具体样本
    bad_cases = defaultdict(list)
    
    refusal_keywords = ["not aware", "i don't know", "cannot answer", "no information", "context does not"]

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # 如果 F1 == 1.0，说明完全正确，跳过
            if data.get("f1", 0) == 1.0 and data.get("hit", 0) == 1:
                continue
                
            error_stats["total_errors"] += 1
            
            predictions = data.get("prediction", [])
            pred_str = " ".join(predictions).lower()
            hit = data.get("hit", 0)
            precision = data.get("precision", 0)
            recall = data.get("recall", 0)
            
            error_type = ""
            
            # 1. 检查是否为大模型拒答
            if any(kw in pred_str for kw in refusal_keywords):
                error_type = "Refusal (大模型拒答)"
                error_stats["type_refusal"] += 1
            # 2. 完全猜错
            elif hit == 0:
                error_type = "Complete Miss (完全猜错或粒度不匹配)"
                error_stats["type_complete_miss"] += 1
            # 3. 冗余预测 (给出了正确答案，但带了多余的错误答案)
            elif hit == 1 and precision < 1.0:
                error_type = "Over-prediction (冗余预测)"
                error_stats["type_over_predict"] += 1
            # 4. 遗漏答案 (给出了部分正确答案，但没给全)
            elif hit == 1 and recall < 1.0:
                error_type = "Under-prediction (遗漏答案)"
                error_stats["type_miss_predict"] += 1
            else:
                error_type = "Other Error (其他)"
            
            # 收集样本用于输出
            if len(bad_cases[error_type]) < sample_size:
                bad_cases[error_type].append(data)

    # 打印统计结果
    print("\n" + "="*50)
    print(f"📊 错误分析统计报告: {input_file}")
    print("="*50)
    print(f"发现总错误样本数: {error_stats['total_errors']}")
    print("-" * 50)
    print(f"1. 冗余预测 (Over-prediction):   {error_stats['type_over_predict']} 例 (模型废话太多)")
    print(f"2. 完全猜错 (Complete Miss):     {error_stats['type_complete_miss']} 例 (包含粒度不匹配)")
    print(f"3. 遗漏答案 (Under-prediction):  {error_stats['type_miss_predict']} 例 (答案没找全)")
    print(f"4. 大模型拒答 (Refusal):         {error_stats['type_refusal']} 例 (触发安全/拒答机制)")
    print("="*50)

    # 将具体的 Bad Case 写入文件
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write(f"错误样例分析报告 (来源: {input_file})\n")
        out_f.write("="*60 + "\n\n")
        
        for err_type, cases in bad_cases.items():
            out_f.write(f"### 错误类型: {err_type}\n")
            out_f.write("-" * 40 + "\n")
            for idx, case in enumerate(cases, 1):
                out_f.write(f"【样本 {idx}】 ID: {case['id']}\n")
                out_f.write(f"   真实答案 (Ground Truth): {case['ground_truth']}\n")
                out_f.write(f"   模型预测 (Prediction)  : {case['prediction']}\n")
                out_f.write(f"   指标: Hit={case['hit']}, F1={case['f1']:.2f}, Pre={case['precision']:.2f}, Rec={case['recall']:.2f}\n")
                out_f.write("\n")
            out_f.write("\n")
            
    print(f"\n✅ 详细的 Bad Case 样本已保存至: {output_file}")
    print("建议打开该文件，结合 GNN 的图谱推理路径进一步分析原因。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KGQA 错误样例分析工具")
    parser.add_argument("input", help="输入的 jsonl 预测结果文件")
    parser.add_argument("--out", default="bad_cases_analysis.txt", help="输出的文本报告文件名")
    parser.add_argument("--samples", type=int, default=10, help="每种错误类型抽取的样本数量")
    
    args = parser.parse_args()
    analyze_errors(args.input, args.out, args.samples)
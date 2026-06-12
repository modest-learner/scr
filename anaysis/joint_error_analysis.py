import json
import argparse
from collections import defaultdict

def analyze_joint_errors(gnn_file, llm_file, output_file, sample_size=10):
    # 1. 读取 GNN 检索出的路径 (构建 Context 字典)
    print(f"正在加载 GNN 路径文件: {gnn_file}")
    gnn_contexts = {}
    try:
        with open(gnn_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                # 将该 ID 的所有路径拼接成一个小写字符串，方便做子串包含判断
                gnn_contexts[data['id']] = " ".join(data.get("prediction", [])).lower()
    except Exception as e:
        print(f"读取 GNN 文件失败: {e}")
        return

    stats = {
        "Total": 0,
        "Type1_Retrieval_Error": 0,
        "Type2_Reasoning_Error": 0,
        "Type3_Parametric_Guess": 0,
        "Type4_Faithful_Success": 0
    }
    
    bad_cases = defaultdict(list)

    # 2. 联合 LLM 的预测结果进行交叉比对
    print(f"正在交叉比对 LLM 预测文件: {llm_file}")
    try:
        with open(llm_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                q_id = data['id']
                
                if q_id not in gnn_contexts:
                    continue
                
                stats["Total"] += 1
                gnn_context_str = gnn_contexts[q_id]
                ground_truths = [gt.lower() for gt in data.get('ground_truth', [])]
                llm_hit = data.get('hit', 0)

                # 核心判断：真实答案是否在 GNN 的推理路径中被召回？
                gt_in_path = any(gt in gnn_context_str for gt in ground_truths)

                # 分类逻辑
                if not gt_in_path and llm_hit == 0:
                    case_type = "Type 1: 检索失败 (GNN未召回, 导致LLM答错)"
                    stats["Type1_Retrieval_Error"] += 1
                elif gt_in_path and llm_hit == 0:
                    case_type = "Type 2: 推理失败 (GNN已召回, 但LLM提取错误/被干扰)"
                    stats["Type2_Reasoning_Error"] += 1
                elif not gt_in_path and llm_hit == 1:
                    case_type = "Type 3: 参数记忆作弊 (GNN未召回, LLM靠内部知识蒙对)"
                    stats["Type3_Parametric_Guess"] += 1
                else:
                    case_type = "Type 4: 忠实成功 (GNN正确召回, LLM正确提取)"
                    stats["Type4_Faithful_Success"] += 1

                # 收集 Bad Cases (跳过 Type 4)
                if "Type 4" not in case_type and len(bad_cases[case_type]) < sample_size:
                    bad_cases[case_type].append({
                        "id": q_id,
                        "ground_truth": data.get('ground_truth'),
                        "llm_prediction": data.get('prediction'),
                        # 只截取前300个字符用于展示，避免文件过大
                        "gnn_context": gnn_context_str[:300] + " ... [截断]" 
                    })
    except Exception as e:
        print(f"读取 LLM 文件失败: {e}")
        return

    # 3. 打印统计报告
    print("\n" + "="*60)
    print("🧠 GNN-LLM 联合错误分析报告 (Joint Error Analysis)")
    print("="*60)
    print(f"有效样本总数: {stats['Total']}")
    print("-" * 60)
    print(f"🔴 Type 1 (检索失败 - GNN优化重点): {stats['Type1_Retrieval_Error']} 例 ({(stats['Type1_Retrieval_Error']/stats['Total'])*100:.2f}%)")
    print(f"🟡 Type 2 (推理失败 - Prompt/LLM问题): {stats['Type2_Reasoning_Error']} 例 ({(stats['Type2_Reasoning_Error']/stats['Total'])*100:.2f}%)")
    print(f"🟣 Type 3 (参数作弊 - 脱离图谱答对): {stats['Type3_Parametric_Guess']} 例 ({(stats['Type3_Parametric_Guess']/stats['Total'])*100:.2f}%)")
    print(f"🟢 Type 4 (忠实成功 - 完美工作流):   {stats['Type4_Faithful_Success']} 例 ({(stats['Type4_Faithful_Success']/stats['Total'])*100:.2f}%)")
    print("="*60)

    # 4. 写入详细的文本报告
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write("GNN-LLM 联合错误分类详细样例\n")
        out_f.write("="*60 + "\n\n")
        for err_type, cases in bad_cases.items():
            out_f.write(f"### {err_type}\n")
            out_f.write("-" * 50 + "\n")
            for idx, case in enumerate(cases, 1):
                out_f.write(f"【样例 {idx}】 ID: {case['id']}\n")
                out_f.write(f"真实答案 (GT)     : {case['ground_truth']}\n")
                out_f.write(f"LLM 实际预测 (Pred): {case['llm_prediction']}\n")
                out_f.write(f"GNN 提供的路径片段 : {case['gnn_context']}\n\n")
            out_f.write("\n")
            
    print(f"✅ 联合分析样本已保存至: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn-file", required=True, help="GNN生成的 jsonl 路径文件 (例如 beam_predictions...)")
    parser.add_argument("--llm-file", required=True, help="LLM预测的 jsonl 结果文件 (例如 _rag_metrics.jsonl)")
    parser.add_argument("--out", default="joint_error_analysis.txt", help="输出的分析报告")
    args = parser.parse_args()
    
    analyze_joint_errors(args.gnn_file, args.llm_file, args.out)
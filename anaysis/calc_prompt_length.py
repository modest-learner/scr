import json
import argparse
from transformers import AutoTokenizer

def calculate_token_lengths(file_path, model_path_or_id):
    print(f"正在加载 Tokenizer: {model_path_or_id} ...")
    try:
        # 加载 Llama 3.1 的 Tokenizer
        # 默认不使用 fast tokenizer，避免某些环境下的 C++ 编译警告，但如果本地支持，加上 use_fast=True 会更快
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
    except Exception as e:
        print(f"Tokenizer 加载失败，请检查模型路径或网络: {e}")
        return

    token_lengths = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                # 提取 prediction 列表
                predictions = data.get("prediction", [])
                
                # 模拟拼接到 Prompt 中的完整上下文（用换行符连接）
                combined_text = "\n".join(predictions)
                
                # 使用 Tokenizer 进行分词
                # add_special_tokens=False 确保我们只计算纯文本的 Token 数，不包含 <s> 等特殊控制符
                tokens = tokenizer.encode(combined_text, add_special_tokens=False)
                token_lengths.append(len(tokens))

    except FileNotFoundError:
        print(f"Error: 找不到文件 '{file_path}'")
        return

    if not token_lengths:
        print("文件中没有提取到有效数据。")
        return

    # 计算统计指标
    num_records = len(token_lengths)
    max_token = max(token_lengths)
    min_token = min(token_lengths)
    avg_token = sum(token_lengths) / num_records

    # 打印结果
    print(f"\n--- Llama 3.1 Token 统计结果 ({file_path}) ---")
    print(f"总计处理样本数: {num_records}")
    print(f"最长 (Max Tokens): {max_token}")
    print(f"最短 (Min Tokens): {min_token}")
    print(f"平均 (Avg Tokens): {avg_token:.2f}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Llama 3.1 Tokenizer 计算 JSONL 文件中 prediction 的 Token 长度分布")
    parser.add_argument("filepath", help="要计算的 .jsonl 文件路径")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Hugging Face 模型 ID 或本地模型文件夹路径")
    args = parser.parse_args()
    
    calculate_token_lengths(args.filepath, args.model)
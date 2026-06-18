#!/bin/bash

# 定义结果汇总文件
RESULTS_FILE="webqsp_alpha_search_results.txt"
echo "=== Alpha Parameter Search Results ===" >> $RESULTS_FILE
echo "Dataset: webqsp | Model: Llama-3.1-8B-Instruct | Top-K: 10" >> $RESULTS_FILE
echo "------------------------------------------------------------" >> $RESULTS_FILE

# 定义要测试的 alpha 数组
ALPHAS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8)

for alpha in "${ALPHAS[@]}"
do
    echo "=========================================="
    echo "🚀 Testing Alpha = $alpha ..."
    echo "=========================================="

    # 🌟 强制清理上一轮跑出来的老文件，确保 LLaMA 重新生成
    # rm -f prediction/webqsp_Llama-3.1-8B-Instruct_rag_alpha${alpha}_top10.jsonl

    # 执行 Python 脚本，将所有的输出同时打印到屏幕并存入临时文件 temp.log
    CUDA_VISIBLE_DEVICES=7 python predict_answer.py \
        --dataset webqsp \
        --add-path \
        --reasoning-path save/webqsp_3_10_20260610-162231/beam_predictions_10_0.02.jsonl \
        --base-url http://localhost:11435/v1/ \
        --model-name Llama-3.1-8B-Instruct \
        --api-key vllm \
        --roberta-checkpoint save/roberta_scorer/webqsp_model.pt \
        --alpha $alpha \
        --top-k 10 2>&1 | tee temp.log

    # 用 grep 提取日志中最后包含 Evaluated 的那一行指标数据
    METRICS=$(grep "Evaluated.*items" temp.log | tail -n 1 | awk -F'- ' '{print $2}')
    
    # 写入结果汇总文件
    if [ -z "$METRICS" ]; then
        echo "[Alpha: $alpha] ❌ Failed or Metrics not found." >> $RESULTS_FILE
    else
        echo "[Alpha: $alpha] -> $METRICS" >> $RESULTS_FILE
    fi

    echo "✅ Alpha $alpha finished!"
    echo ""
done

# 清理临时文件
rm temp.log

echo "🎉 All done! Here is the summary:"
echo "------------------------------------------------------------"
cat $RESULTS_FILE

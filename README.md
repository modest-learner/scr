# Stepwise Contrastive Reasoning (SCR)

### Step 1: Prepare environment

We recommend using **Python 3.10.12** and **CUDA 12.8** for optimal compatibility and performance.

Run the following commands to install the required python dependencies:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets torch_geometric loguru openai
(optional) pip install flash-attn --no-build-isolation
```

### Step 2: Preprocess datasets
SCR supports two knowledge graph question answering (KGQA) datasets: `WebQSP` and `CWQ`. Both were introduced in **Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning**.

Run the following command to preprocess both datasets, including their `train`, `validation`, and `test` splits:

```bash
python preprocess_datasets.py
```
The processed dataset will be saved to the `data` folder by default. 

To specify a different location, use the `--save-dir {folder_path}` argument.

The datasets and *bert-base-uncased* will be automatically downloaded from Huggingface when running the code.

If you have already downloaded the dataset locally, use `--data-path {folder_path}` to specify its location. 

To use a locally saved pre-trained language model, set `--plm` to its directory.

### Step 3: Train SCR

We use PyTorch's Distributed Data Parallel (DDP) to support multi-GPU execution on a local machine.

If your PyTorch version supports `torchrun`, run:
```bash
torchrun --standalone --nproc_per_node gpu train_gnn.py --dataset {dataset}
```
Otherwise, run:
```bash
python -m torch.distributed.launch --use-env train_gnn.py --dataset {dataset}
```

If an Out-of-Memory (OOM) error occurs during training, reduce memory usage by setting `--max-batch-len` with a smaller value.

We recommend setting `max-batch-len = g × 750`, where `g` is the available GPU memory in gigabytes (GB).

By default, the checkpoint of the trained model is saved to: `save/{dataset}_{gnn_layers}_{num_beam}_{time}/model.pt`

### Step 4: Generate Reasoning Paths
```bash
torchrun --standalone --nproc_per_node gpu train_gnn.py --dataset {dataset} --test --checkpoint {checkpoint_path}
```
By default, the reasoning results are saved to: `save/{dataset}_{gnn_layers}_{num_beam}_{time}/beam_predictions_{max_hop}_{prob_threshold}.jsonl`

### Step 5: Evaluation

```bash
python predict_answer.py \
    --dataset {dataset} \
    --add-path \
    --reasoning-path {reasoning_results_path} \
    --model-name {llm_name} \
    --api-key {api_key}
```
To evaluate using the LLM only, you can omit the `--add-path` option.

If you want to use Ollama on your local machine, run:
```bash
python predict_answer.py \
    --dataset {dataset} \
    --add-path \
    --reasoning-path {reasoning_results_path} \
    --base-url http://localhost:11434/v1/ \
    --model-name {llm_model_name} \
    --api-key ollama
```
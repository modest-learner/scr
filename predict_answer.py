import os
import os.path as osp
import argparse
from multiprocessing import set_start_method
import json
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset

from utils import ChatGPT, Metric


def predict(args):
    id2paths = {}
    with open(args.reasoning_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            id2paths[obj["id"]] = obj["prediction"]

    dataset = load_dataset(f'{args.data_path}/RoG-{args.dataset}', split='test')

    # init output environment
    os.makedirs(args.save_path, exist_ok=True)
    suffix = 'rag' if args.add_path else 'llm'
    output_file = osp.join(args.save_path, f'{args.dataset}_{args.model_name}_{suffix}.jsonl')

    # filter processed samples
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

    # start process
    if len(dataset) > len(processed_ids):
        model = ChatGPT(args)
        samples = dataset
        if len(processed_ids) > 0:
            samples = [sample for sample in dataset if sample["id"] not in processed_ids]
        for data in tqdm(samples, desc='Predict'):
            if args.add_path:
                qid = data["id"]
                paths = id2paths[qid] if qid in id2paths else []
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
    parser.add_argument('--dataset', type=str, default='webqsp', choices=['webqsp', 'cwq'])
    parser.add_argument('--save-path', type=str, default='prediction')
    parser.add_argument("--add-path", action='store_true')
    parser.add_argument("--reasoning-path", type=str)
    parser.add_argument('--base-url', type=str, default=None)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--api-key', type=str, default='')
    parser.add_argument('--num-retry', type=int, default=5)

    predict(parser.parse_args())


if __name__ == '__main__':
    set_start_method('spawn')
    main()

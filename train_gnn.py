import argparse
import os
import torch
from transformers import set_seed
from torch.distributed.elastic.multiprocessing.errors import record

from src import Trainer

@record
def main(args):
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.processed_data_path, exist_ok=True)
    Trainer(args, local_rank).start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # inference
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test-on-train', action='store_true', help='强制在训练集上跑 Beam Search (用于挖掘困难负样本)') # 🌟 新增开关
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--max-hop', type=int, default=10)
    parser.add_argument('--prob-threshold', default=0.02, type=float)
    parser.add_argument('--num-beam', type=int, default=10)
    # data
    parser.add_argument('--dataset', type=str, default='webqsp', choices=['webqsp', 'cwq'])
    parser.add_argument('--processed-data-path', type=str, default='data')
    parser.add_argument('--max-seq-len', type=int, default=20)
    # model
    parser.add_argument('--plm', type=str, default='bert-base-uncased')
    parser.add_argument('--temper', default=0.05, type=float)
    parser.add_argument('--gnn-layers', type=int, default=3)
    parser.add_argument('--gnn-heads', type=int, default=2)
    # train
    parser.add_argument('--save-path', type=str, default='save')
    parser.add_argument('--train-epochs', type=int, default=20)
    parser.add_argument('--max-batch-len', type=int, default=36000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=10.0)

    main(parser.parse_args())

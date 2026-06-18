import os
import os.path as osp
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from datetime import datetime
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from loguru import logger
import json

from .data import get_loader
from .model import CrossModel


class Trainer:
    def __init__(self, args, local_rank):
        self.args = args
        self.rank = local_rank
        dist.init_process_group(backend='nccl')

        use_checkpoint = args.checkpoint is not None and osp.exists(args.checkpoint)
        # 🌟 统一判断是否处于推理模式（不论是在 test 还是 train 上跑）
        is_eval_mode = self.args.test or getattr(self.args, 'test_on_train', False)
        if not is_eval_mode:
            time = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.save_path = osp.join(args.save_path, f'{args.dataset}_{args.gnn_layers}_{args.num_beam}_{time}')
            os.makedirs(self.save_path, exist_ok=True)
            if dist.get_rank() == 0:
                logger.add(osp.join(self.save_path, 'log.txt'))
                logger.info(json.dumps(args.__dict__))
            self.train_loader = get_loader(args, 'train')
            self.test_loader = get_loader(args, 'validation')
        else:
            # 🌟 如果是 test_on_train 开关打开了，就去加载 'train' 的文件，否则加载 'test'
            split_to_eval = 'train' if getattr(self.args, 'test_on_train', False) else 'test'
            self.test_loader = get_loader(args, split_to_eval)
            if use_checkpoint:
                self.save_path = osp.dirname(args.checkpoint)
            
            # 🌟 给输出文件加个前缀，区分是测试集还是训练集跑出来的
            self.output_prefix = 'gnn_train' if getattr(self.args, 'test_on_train', False) else 'beam'
        self.model = CrossModel(args).to(self.rank)
        if use_checkpoint:
            self.model.load_state_dict(torch.load(args.checkpoint, weights_only=False))
        self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)

    def start(self):
        # 🌟 确保 test_on_train 也能进入 _evaluate 推理阶段
        if self.args.test or getattr(self.args, 'test_on_train', False):
            score = self._evaluate()
            if score is not None:
                logger.info(f'hit = {score * 100:.2f}%')
        else:
            self._train()

    def _train(self):
        scaler = GradScaler()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(parameters, lr=self.args.lr, weight_decay=self.args.weight_decay)
        total_steps = len(self.train_loader) * self.args.train_epochs
        warmup_steps = int(total_steps * 0.15)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        best_score = 0
        for epoch in range(self.args.train_epochs):
            self.model.train()
            self.train_loader.sampler.set_epoch(epoch)

            for data in tqdm(self.train_loader, desc=f'Epoch {epoch}', disable=dist.get_rank() != 0):
                data = self.batch_to_device(data)
                with autocast('cuda'):
                    loss = self.model(data)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            score = self._evaluate(True, best_score)
            if score is not None:
                logger.info(f'Epoch {epoch}: hit = {score * 100:.2f}%')
                if score > best_score:
                    best_score = score
                    torch.save(self.model.module.state_dict(), osp.join(self.save_path, 'model.pt'))

        if dist.get_rank() == 0:
            logger.info(f'Training finish. Best score: {best_score:.4f}')

    def _evaluate(self, greedy=False, last_score=0):
        self.model.eval()
        results = []
        for data in tqdm(self.test_loader, desc='Evaluating', disable=dist.get_rank() != 0):
            data = self.batch_to_device(data)
            data['greedy'] = greedy
            with torch.no_grad():
                if greedy:
                    pred = self.model(data)
                    obj = {'pred': pred, 'label': data['end']}
                else:
                    pred, all_pred = self.model(data)
                    obj = {'id': data['id'], 'pred': pred, 'all': all_pred, 'label': data['end']}

            gather_list = [None for _ in range(dist.get_world_size())] if dist.get_rank() == 0 else None
            dist.gather_object(obj, gather_list, dst=0)
            if gather_list is not None:
                results.extend(gather_list)

        if len(results) == 0:
            return None

        hit, cnt = 0, 0
        for obj in results:
            if obj is not None:
                cnt += 1
                if len(set(obj['pred']).intersection(set(obj['label']))) > 0:
                    hit += 1

        acc = hit / cnt
        if not greedy:
            objects = self.test_loader.dataset.convert_to_raw_paths(results, self.args.prob_threshold, self.args.num_beam)
            # 🌟 使用我们刚才定义的 output_prefix
            with open(osp.join(self.save_path, f'{self.output_prefix}_predictions_{self.args.max_hop}_{self.args.prob_threshold}.jsonl'), 'w') as fs:
                for obj in objects:
                    fs.write(json.dumps(obj))
                    fs.write("\n")
                    fs.flush()
        elif acc > last_score:
            with open(osp.join(self.save_path, 'predictions.json'), 'w') as f:
                json.dump(results, f)
        return acc

    def batch_to_device(self, data):
        clone = {}
        for k, v in data.items():
            if torch.is_tensor(v):
                clone[k] = v.to(self.rank)
            elif k == 'paths':
                clone[k] = self.batch_to_device(v)
            else:
                clone[k] = v
        return clone


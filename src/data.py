import os
import os.path as osp
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import networkx as nx
from multiprocessing import Pool
from collections import defaultdict as ddict
from time import time
import torch.distributed as dist

from .model import apply_mean_pooling


ANS_TEMPLATE = """# Reasoning Path:
{reasoning_path}
# Answer:
{answer}"""


class IndexDict:
    def __init__(self):
        super().__init__()
        self.dictionary = {}
        self.stored_keys = []

    def add(self, key):
        if key in self.dictionary:
            return self.dictionary[key]
        index = len(self.stored_keys)
        self.dictionary[key] = index
        self.stored_keys.append(key)
        return index

    def update(self, keys):
        return [self.add(key) for key in keys]

    def __getitem__(self, key):
        if key not in self.dictionary:
            return None
        return self.dictionary[key]

    def __contains__(self, key):
        return key in self.dictionary


def parse_sample(sample):
    ent2id, rel2id = IndexDict(), IndexDict()
    heads, rels, tails = [], [], []
    graph = nx.DiGraph()
    for h, r, t in sample['graph']:
        head, rel, tail = ent2id.add(h), rel2id.add(r), ent2id.add(t)
        graph.add_edge(head, tail, r=rel)
        heads.append(head)
        rels.append(rel)
        tails.append(tail)
    shortest_paths = []
    for start in sample['q_entity']:
        if start not in ent2id:
            continue
        for end in sample['a_entity']:
            if end not in ent2id:
                continue
            try:
                for path in nx.all_shortest_paths(graph, ent2id[start], ent2id[end]):
                    shortest_paths.append(path)
            except nx.NetworkXNoPath:
                pass
    if len(shortest_paths) == 0:
        return None
    paths, src, dst = [], set(), set()
    for p in shortest_paths:
        src.add(p[0])
        dst.add(p[-1])
        paths.append({'path': p, 'rels': [graph[p[i]][p[i + 1]]['r'] for i in range(len(p) - 1)]})

    return {
        'id': sample['id'],
        'heads': heads,
        'rels': rels,
        'tails': tails,
        'paths': paths,
        'entities': ent2id.stored_keys,
        'relations': rel2id.stored_keys,
        'start': list(src),
        'end': list(dst),
        'question': sample['question']
    }


def encode(text_list, tokenizer, model, device, max_seq_len, batch_size, desc):
    embeddings = []
    for i in tqdm(range(0, len(text_list), batch_size), desc=f'Encoding {desc}'):
        token = tokenizer(text_list[i : i + batch_size], max_length=max_seq_len,
                          truncation=True, padding=True, return_tensors='pt')
        token = token.to(device)
        with torch.no_grad():
            output = model(**token)
        embeddings.append(apply_mean_pooling(output[0], token['attention_mask']))

    return torch.cat(embeddings, dim=0)


def process_dataset(data_path, dataset, split, plm, device, save_dir, max_seq_len, batch_size, num_processes):
    task = f'{dataset}[{split}]'
    raw_dataset = load_dataset(f'{data_path}/RoG-{dataset}', split=split)
    ent2id, rel2id = IndexDict(), IndexDict()
    processed_samples, questions = [], []
    with Pool(num_processes) as p:
        for sample in tqdm(p.imap_unordered(parse_sample, raw_dataset), total=len(raw_dataset), desc=f'Processing {task}'):
            if sample is not None:
                sample['entities'] = ent2id.update(sample['entities'])
                sample['relations'] = rel2id.update(sample['relations'])
                questions.append(sample.pop('question'))
                processed_samples.append(sample)

    entities = ent2id.stored_keys
    relations = [r.replace('.', ' ').replace('_', ' ').strip() for r in rel2id.stored_keys]
    tokenizer = AutoTokenizer.from_pretrained(plm)
    model = AutoModel.from_pretrained(plm).to(device)
    model.eval()

    ent_features = encode(entities, tokenizer, model, device, max_seq_len, batch_size, task)
    rel_features = encode(relations, tokenizer, model, device, max_seq_len, batch_size, task)

    ent_ids = tokenizer(entities, return_token_type_ids=False, return_attention_mask=False, add_special_tokens=False,
                        max_length=max_seq_len - 2, truncation=True)['input_ids']
    rel_ids = tokenizer(relations, return_token_type_ids=False, return_attention_mask=False, add_special_tokens=False,
                        max_length=max_seq_len - 2, truncation=True)['input_ids']
    q_ids = tokenizer(questions, return_token_type_ids=False, return_attention_mask=False,
                      add_special_tokens=False)['input_ids']

    for sample, ids in zip(processed_samples, q_ids):
        sample['question'] = ids

    data = {
        'samples': processed_samples,
        'ent': ent_features,
        'rel': rel_features,
        'ent_ids': ent_ids,
        'rel_ids': rel_ids,
        'cls': tokenizer.cls_token_id,
        'sep': tokenizer.sep_token_id,
        'pad': tokenizer.pad_token_id
    }
    if split == 'test':
        data['raw_entities'] = ent2id.stored_keys
        data['raw_relations'] = rel2id.stored_keys
    torch.save(data, osp.join(save_dir, f'{dataset}_{split}_{osp.basename(plm)}_{max_seq_len}.pt'))


def process_data(args):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.makedirs(args.save_dir, exist_ok=True)
    for dataset in args.datasets.split(','):
        for split in args.splits.split(','):
            process_dataset(args.data_path, dataset, split, args.plm, args.device, args.save_dir,
                            args.max_seq_len, args.batch_size, args.num_processes)


def load_data(data_dir, dataset, split, plm, max_seq_len, max_batch_len):
    data_path = osp.join(data_dir, f'{dataset}_{split}_{osp.basename(plm)}_{max_seq_len}.pt')
    task = f'{dataset}[{split}]'
    if is_first_proc():
        print(f'Loading dataset {task}...')
    t = time()
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    t = time() - t
    if is_first_proc():
        print(f'Loading took {t:.2f} seconds.')
    for ids in data['ent_ids']:
        ids.append(data['sep'])
    for ids in data['rel_ids']:
        ids.append(data['sep'])
    parsed_samples = []
    for sample in tqdm(data['samples'], desc=f'Resolving {task}', disable=not is_first_proc()):
        links, edges = ddict(set), {}
        for h, r, t in zip(sample['heads'], sample['rels'], sample['tails']):
            if h != t:
                edges[(h, t)] = r
                edges[(t, h)] = r
                links[h].add(t)
                links[t].add(h)
        sample['question'].insert(0, data['cls'])
        sample['question'].append(data['sep'])
        ent_ids = [data['ent_ids'][i] for i in sample['entities']]
        rel_ids = [data['rel_ids'][i] for i in sample['relations']]
        if split != 'train':
            sample.pop('paths')
            sample['links'] = {k: v for k, v in links.items()}
            sample['edges'] = edges
            sample['ent_ids'] = ent_ids
            sample['rel_ids'] = rel_ids
        else:
            sample.pop('start')
            sample.pop('end')
            q_ids = sample.pop('question')
            paths = []
            for obj in sample['paths']:
                path, rels = obj['path'], obj['rels']
                passed_nodes = set()
                ids = list(q_ids)
                for i, u in enumerate(path):
                    if i > 0:
                        ids += rel_ids[rels[i - 1]]
                    ids += ent_ids[u]
                    nodes = set(links[u] - passed_nodes)
                    if i + 1 < len(path):
                        v = path[i + 1]
                        neighbors = [v, u]
                        nodes.discard(v)
                        neighbors.extend(nodes)
                    else:
                        neighbors = [u]
                        neighbors.extend(nodes)
                    passed_nodes.add(u)
                    if len(neighbors) > 1:
                        paths.append((list(ids), neighbors))
                        # paths.append({'input_ids': torch.LongTensor([ids]), 'links': neighbors})
            if len(paths) > 0:
                paths.sort(key=lambda x: len(x[0]))
                # split ot batches
                all_paths, batch_paths = [], []
                for ids, neighbors in paths:
                    if len(batch_paths) * len(ids) > max_batch_len:
                        all_paths.append(batch_paths)
                        batch_paths = []
                    batch_paths.append((ids, neighbors))
                if len(all_paths) == 0:
                    sample['paths'] = build_batch_input(batch_paths)
                    parsed_samples.append(sample)
                else:
                    all_paths.append(batch_paths)
                    for item in all_paths:
                        clone = {k: v for k, v in sample.items()}
                        clone['paths'] = build_batch_input(item)
                        parsed_samples.append(clone)
    if split == 'train':
        data['samples'] = parsed_samples
    return data


def build_batch_input(batch_data):
    max_ids_len, max_link_len = 0, 0
    for ids, neighbors in batch_data:
        max_ids_len = max(max_ids_len, len(ids))
        max_link_len = max(max_link_len, len(neighbors))

    ids_list, mask_list, link_list, batch_mask = [], [], [], []
    for ids, neighbors in batch_data:
        length = len(ids)
        mask = [1] * length
        if length < max_ids_len:
            padding = [0] * (max_ids_len - length)
            ids_list.append(ids + padding)
            mask_list.append(mask + padding)
        else:
            ids_list.append(ids)
            mask_list.append(mask)

        padding_size = max_link_len - len(neighbors)
        link_list.append(neighbors + [0] * padding_size)
        batch_mask.append([False] * len(neighbors) + [True] * padding_size)

    return {
        'input_ids': torch.LongTensor(ids_list),
        'attention_mask': torch.LongTensor(mask_list),
        'links': torch.LongTensor(link_list).view(-1),
        'mask': torch.BoolTensor(batch_mask)
    }


def is_first_proc():
    return dist.is_initialized() and dist.get_rank() == 0


def get_loader(args, split):
    data_list = load_data(args.processed_data_path, args.dataset, split, args.plm, args.max_seq_len, args.max_batch_len)
    dataset = SampleDataset(data_list)
    sampler = DistributedSampler(dataset, shuffle=split == 'train')
    return DataLoader(dataset, collate_fn=lambda x: x[0], sampler=sampler, batch_size=1)


class SampleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['samples'])

    def __getitem__(self, index):
        data = self.data

        sample = data['samples'][index]
        heads, rels, tails = sample['heads'], sample['rels'], sample['tails']
        node_attr = data['ent'][sample['entities']]
        edge_index = torch.LongTensor([heads + tails, tails + heads])
        edge_attr = data['rel'][sample['relations']][rels + rels]

        if 'paths' in sample:
            return {
                'node_attr': node_attr,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'paths': sample['paths']
            }
        return {
            'id': sample['id'],
            'node_attr': node_attr,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'q_ids': sample['question'],
            'ent_ids': sample['ent_ids'],
            'rel_ids': sample['rel_ids'],
            'links': sample['links'],
            'edges': sample['edges'],
            'start': sample['start'],
            'end': sample['end']
        }

    def convert_to_raw_paths(self, results, prob_threshold, max_num):
        data = self.data

        raw_entities, raw_relations = data['raw_entities'], data['raw_relations']
        id2sample = {sample['id']: sample for sample in data['samples']}
        objects = []
        for item in results:
            key = item['id']
            sample = id2sample[key]
            prediction = []
            for i, sol in enumerate(item['all']):
                if len(sol['path']) == 0:
                    continue
                if i >= max_num or sol['prob'] < prob_threshold:
                    break
                ans = raw_entities[sample['entities'][sol['node']]]
                if ans.startswith('m.'):
                    continue
                if len(sol['path']) == 0:
                    prediction.append(ANS_TEMPLATE.format(reasoning_path=ans, answer=ans))
                else:
                    path_str = ''
                    for j, (h, r, t) in enumerate(sol['path']):
                        if j == 0:
                            path_str += raw_entities[sample['entities'][h]]
                        rel = raw_relations[sample['relations'][r]]
                        tail = raw_entities[sample['entities'][t]]
                        path_str += f" -> {rel} -> {tail}"
                    prediction.append(ANS_TEMPLATE.format(reasoning_path=path_str, answer=ans))
            objects.append({"id": key, "prediction": prediction})
        return objects

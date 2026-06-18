import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from transformers import AutoModel

class CrossModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(args.plm)
        self.gnn = GAT(self.encoder.config.hidden_size, args.gnn_layers, args.gnn_heads)
        self.criterion = nn.CrossEntropyLoss()
        self.log_t = torch.nn.Parameter(torch.tensor(1.0 / args.temper).log(), requires_grad=True)
        self.max_hop = args.max_hop
        self.num_beam = args.num_beam
        self.prob_threshold = args.prob_threshold

    def forward(self, data):
        graph_embeddings = self.gnn(data['node_attr'], data['edge_index'], data['edge_attr'])
        graph_embeddings = F.normalize(graph_embeddings, dim=-1)

        if self.training:
            path = data['paths']
            mask, links = path['mask'], path['links']
            text_embeddings = self.encoder(path['input_ids'], path['attention_mask'])
            text_embeddings = apply_mean_pooling(text_embeddings[0], path['attention_mask'])
            graph_embeddings = graph_embeddings[links].reshape(mask.shape[0], mask.shape[1], -1) # [bz, num_ent, dim]
            logits = torch.sum(text_embeddings.unsqueeze(dim=1) * graph_embeddings, dim=-1) # [bz, num_ent]
            logits *= self.log_t.exp()
            logits.masked_fill_(mask, -1e4)
            label = torch.zeros(logits.shape[0], dtype=torch.int64, device=logits.device)
            return self.criterion(logits, label)

        q_ids, e_ids, r_ids = data['q_ids'], data['ent_ids'], data['rel_ids']
        links, edges = data['links'], data['edges']
        pred = []
        if data['greedy']:
            for node in data['start']:
                passed_nodes = set()
                m = node
                ids = q_ids + e_ids[m]
                for i in range(self.max_hop):
                    nodes = [m]
                    nodes.extend(links[m] - passed_nodes)
                    if len(nodes) == 1:
                        break # 如果除了自己没有别的路可走了（走到死胡同），强制结束寻路
                    text_embeddings = self.encoder(torch.LongTensor([ids]).to(self.log_t.device))[0]
                    text_embeddings = F.normalize(text_embeddings.mean(dim=1), dim=-1)
                    logits = text_embeddings.mm(graph_embeddings[nodes].t())
                    a = torch.argsort(logits[0], descending=True)[0].item()
                    if a == 0:
                        break # 如果最高分是 0 号节点（即节点本身），说明模型认为已经找到答案了，主动停止
                    n = nodes[a]
                    r = edges[(m, n)]
                    ids += r_ids[r] + e_ids[n]
                    passed_nodes.add(m)
                    m = n
                pred.append(m)
            return pred
        # beam search
        solutions = [Solution(q_ids + e_ids[node], node) for node in data['start']]
        for _ in range(self.max_hop):
            size = len(solutions)
            for i in range(size):
                sol = solutions[i]
                nodes = [sol.node]
                nodes.extend(links[sol.node] - sol.passed_nodes)
                if len(nodes) == 1:
                    pred.append(Solution.clone(sol))
                    continue

                text_embeddings = self.encoder(torch.LongTensor([sol.ids]).to(self.log_t.device))[0]
                text_embeddings = F.normalize(text_embeddings.mean(dim=1), dim=-1)
                logits = text_embeddings.mm(graph_embeddings[nodes].t()) * self.log_t.exp()
                probs = logits.softmax(dim=-1).squeeze(dim=0)
                indices = torch.argsort(probs, descending=True)[:self.num_beam].tolist()

                passed_nodes = set(sol.passed_nodes)
                passed_nodes.add(sol.node)
                for idx in indices:
                    prob = sol.prob * probs[idx].item()
                    if prob < self.prob_threshold:
                        continue
                    if idx == 0:
                        pred.append(Solution.clone(sol, passed_nodes))
                    else:
                        n = nodes[idx]
                        r = edges[(sol.node, n)]
                        solutions.append(Solution.walk(sol, sol.ids + r_ids[r] + e_ids[n], r, n, passed_nodes, prob))

            solutions = solutions[size:]
            if not solutions:
                break
            solutions.sort(key=lambda x: x.prob, reverse=True)
            solutions = solutions[:self.num_beam]

        pred.sort(key=lambda x: x.prob, reverse=True)
        pred_nodes = set()
        for sol in pred:
            pred_nodes.add(sol.node)
            if len(pred_nodes) >= self.num_beam:
                break
        return list(pred_nodes), [sol.to_dict() for sol in pred]


class Solution:
    @classmethod
    def clone(cls, solution, passed_nodes=None):
        if passed_nodes is None:
            passed_nodes = solution.passed_nodes
        return cls(solution.ids, solution.node, list(solution.path), passed_nodes, solution.prob)

    @classmethod
    def walk(cls, solution, ids, next_edge, next_node, passed_nodes, prob):
        path = list(solution.path)
        path.append((solution.node, next_edge, next_node))
        return cls(ids, next_node, path, passed_nodes, prob)

    def __init__(self, ids, node, path=None, passed_nodes=None, prob=1.):
        self.ids = ids
        self.node = node
        self.path = path or []
        self.passed_nodes = passed_nodes or set()
        self.prob = prob

    def to_dict(self):
        return {'node': self.node, 'path': self.path, 'prob': self.prob}


class GAT(nn.Module):
    def __init__(self, hidden_size, num_layers=2, num_heads=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                dim = hidden_size
            else:
                dim = hidden_size * num_heads
            layer = GATConv(dim, hidden_size, num_heads, concat=i + 1 < num_layers,
                            edge_dim=hidden_size, residual=True, add_self_loops=False, bias=False)
            self.layers.append(layer)

    def forward(self, x, edge_index, edge_attr):
        for i, layer in enumerate(self.layers):
            if i > 0:
                x = F.elu(x)
            x = layer(x, edge_index, edge_attr)
        return x


def apply_mean_pooling(hidden_states, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states.dtype)
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    output_vector = sum_embeddings / sum_mask
    return F.normalize(output_vector, dim=1)
import json
import re
import string
import os.path as osp
from dataclasses import dataclass
from loguru import logger


def normalize(s: str) -> str:
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


class Metric:
    def __init__(self, input_file):
        self.f_in = open(input_file, "r")
        self.acc, self.hit = [], []
        self.metric = F1Score()
        self.obj = None
        self.line_no = 0
        self.iter = None
        dirname = osp.dirname(input_file)
        filename = osp.basename(input_file)
        fn, ext = osp.splitext(filename)
        self.f_detail = open(osp.join(dirname, f'{fn}_metrics{ext}'), "w")

    def add(self, prediction, answer):
        precision = self._calc_precision(prediction, answer)
        recall = self._calc_recall(prediction, answer)
        f1 = self._calc_f1(precision, recall)
        self.metric.f1.append(f1)
        self.metric.precision.append(precision)
        self.metric.recall.append(recall)
        self.obj['f1'] = f1
        self.obj['precision'] = precision
        self.obj['recall'] = recall

    def __iter__(self):
        if self.iter is not None:
            raise Exception(f'The iteration is already done.')
        self.iter = iter(self.f_in)
        return self

    def __next__(self):
        if self.obj is not None:
            self.f_detail.write(json.dumps(self.obj))
            self.f_detail.write("\n")

        while True:
            line = next(self.iter)
            self.line_no += 1
            try:
                if len(line) < 2:
                    continue
                data = json.loads(line)
            except:
                logger.error(f'Failed to load json text in line {self.line_no}: {line}')
                continue

            prediction = data["prediction"]
            answer = list(set(data["ground_truth"]))
            prediction = self._extract_prediction(prediction)
            self.obj = {"id": data["id"],
                        "prediction": prediction,
                        "ground_truth": answer}

            prediction_str = " ".join(prediction)
            acc = self._calc_accuracy(prediction_str, answer)
            hit = self._calc_hit(prediction_str, answer)
            self.acc.append(acc)
            self.hit.append(hit)
            self.obj['acc'] = acc
            self.obj['hit'] = hit

            return prediction, answer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f_in.close()
        self.f_detail.close()
        text = self._append_metric(f'Evaluated {len(self.acc)} items:', 'accuracy', self.acc)
        text = self._append_metric(text, 'hit', self.hit)
        text = self._append_metric(text, 'f1', self.metric.f1)
        text = self._append_metric(text, 'precision', self.metric.precision)
        text = self._append_metric(text, 'recall', self.metric.recall)
        logger.info(text)

    @staticmethod
    def _extract_prediction(prediction, k=-1):
        if isinstance(prediction, str):
            prediction = prediction.split("\n")
        results = {}
        for p in prediction:
            if p.strip() == "":
                continue
            if p in results:
                results[p] += 1
            else:
                results[p] = 1
        if k > len(results) or k < 0:
            k = len(results)
        results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:k]]

    @staticmethod
    def _append_metric(text, key, values):
        if len(values) > 0:
            if len(text) == 0:
                return f'{key} = {sum(values) * 100 / len(values):.2f}%'
            return f'{text}, {key} = {sum(values) * 100 / len(values):.2f}%'
        return text

    @staticmethod
    def _calc_accuracy(prediction, answer):
        if len(answer) == 0:
            return 0
        matched = 0
        for a in answer:
            if match(prediction, a):
                matched += 1
        return matched / len(answer)

    @staticmethod
    def _calc_hit(prediction, answer):
        if len(answer) == 0:
            return 0
        for a in answer:
            if match(prediction, a):
                return 1
        return 0

    @staticmethod
    def _calc_precision(prediction, answer):
        if len(prediction) == 0 or len(answer) == 0:
            return 0
        prediction_correct = 0
        for p in prediction:
            for a in answer:
                if match(p, a):
                    prediction_correct += 1
                    break
        return prediction_correct / len(prediction)

    @staticmethod
    def _calc_recall(prediction, answer):
        if len(prediction) == 0 or len(answer) == 0:
            return 0
        ans_recalled = 0
        prediction_str = " ".join(prediction)
        for a in answer:
            if match(prediction_str, a):
                ans_recalled += 1
        return ans_recalled / len(answer)

    @staticmethod
    def _calc_f1(precision, recall):
        value = precision + recall
        if value == 0:
            return 0
        return 2 * precision * recall / value


@dataclass
class F1Score:
    f1 = []
    precision = []
    recall = []


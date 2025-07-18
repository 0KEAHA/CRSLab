

import re
from collections import Counter

import math
import numpy as np
import jieba
from loguru import logger
import sys
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional

from evaluator.metrics.base import AverageMetric, SumMetric

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
re_space = re.compile(r'\s+')


class PPLMetric(AverageMetric):
    def value(self):
        return math.exp(super().value())


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = re_space.sub(' ', s)
    # s = ' '.join(s.split())
    return s


class ExactMatchMetric(AverageMetric):
    @staticmethod
    def compute(guess: str, answers: List[str]) -> 'ExactMatchMetric':
        if guess is None or answers is None:
            return None
        for a in answers:
            if guess == a:
                return ExactMatchMetric(1)
        return ExactMatchMetric(0)


class F1Metric(AverageMetric):
    """
    Helper class which computes token-level F1.
    """

    @staticmethod
    def _prec_recall_f1_score(pred_items, gold_items):
        """
        Compute precision, recall and f1 given a set of gold and prediction items.

        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values

        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def compute(guess: List[str], answers: List[List[str]]) -> 'F1Metric':
        if guess is None or answers is None:
            return AverageMetric(0, 0)
        scores = [
            F1Metric._prec_recall_f1_score(guess, a)
            for a in answers
        ]
        return F1Metric(max(scores), 1)


class BleuMetric(AverageMetric):
    @staticmethod
    def compute(guess: List[str], answers: List[List[str]], k: int) -> Optional['BleuMetric']:
        """
        Compute approximate BLEU score between guess and a set of answers.
        """
        tokenized_guess = guess
        tokenized_answers = answers
        if not tokenized_guess or not any(tokenized_answers):
            return BleuMetric(0.0)
        weights = [0] * 4
        if k < 1 or k > 4:
            logger.info(f"Warning: Invalid k={k} for BleuMetric. Using k=4.")
            k = 4
        weights[k - 1] = 1
        try:
            # 使用分词后的结果计算 BLEU
            score = sentence_bleu(
                tokenized_answers,
                tokenized_guess,
                weights=weights,
            )
        except ZeroDivisionError:
            # 如果 guess 过短 (例如少于 k 个词)，n-gram 分母可能为0
            score = 0.0
        except Exception as e:
            # 捕获其他潜在错误
            logger.error(f"Error calculating BLEU score: {e}")
            score = 0.0
        return BleuMetric(score)

class DistMetric(SumMetric):
    @staticmethod
    def compute(sent: List[str], k: int) -> 'DistMetric':
        token_set = set()
        for token in ngrams(sent, k):
            token_set.add(token)
        return DistMetric(len(token_set))


class EmbeddingAverage(AverageMetric):
    @staticmethod
    def _avg_embedding(embedding):
        return np.sum(embedding, axis=0) / (np.linalg.norm(np.sum(embedding, axis=0)) + 1e-12)

    @staticmethod
    def compute(hyp_embedding, ref_embeddings) -> 'EmbeddingAverage':
        hyp_avg_emb = EmbeddingAverage._avg_embedding(hyp_embedding).reshape(1, -1)
        ref_avg_embs = [EmbeddingAverage._avg_embedding(emb) for emb in ref_embeddings]
        ref_avg_embs = np.array(ref_avg_embs)
        return EmbeddingAverage(float(cosine_similarity(hyp_avg_emb, ref_avg_embs).max()))


class VectorExtrema(AverageMetric):
    @staticmethod
    def _extreme_embedding(embedding):
        max_emb = np.max(embedding, axis=0)
        min_emb = np.min(embedding, axis=0)
        extreme_emb = np.fromiter(
            map(lambda x, y: x if ((x > y or x < -y) and y > 0) or ((x < y or x > -y) and y < 0) else y, max_emb,
                min_emb), dtype=float)
        return extreme_emb

    @staticmethod
    def compute(hyp_embedding, ref_embeddings) -> 'VectorExtrema':
        hyp_ext_emb = VectorExtrema._extreme_embedding(hyp_embedding).reshape(1, -1)
        ref_ext_embs = [VectorExtrema._extreme_embedding(emb) for emb in ref_embeddings]
        ref_ext_embs = np.asarray(ref_ext_embs)
        return VectorExtrema(float(cosine_similarity(hyp_ext_emb, ref_ext_embs).max()))


class GreedyMatch(AverageMetric):
    @staticmethod
    def compute(hyp_embedding, ref_embeddings) -> 'GreedyMatch':
        hyp_emb = np.asarray(hyp_embedding)
        ref_embs = (np.asarray(ref_embedding) for ref_embedding in ref_embeddings)
        score_max = 0
        for ref_emb in ref_embs:
            sim_mat = cosine_similarity(hyp_emb, ref_emb)
            score_max = max(score_max, (sim_mat.max(axis=0).mean() + sim_mat.max(axis=1).mean()) / 2)
        return GreedyMatch(score_max)

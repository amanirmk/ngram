from math import log10
from typing import Union, List, Optional, Iterator, Tuple, Dict
from pathlib import Path
import h5py
import numpy as np
from ngram.abstract import Object
from ngram.processing import tokenize
from ngram.datatypes import NGram


class Model(Object):
    UNK = "<UNK>"
    BOS = "<S>"
    EOS = "</S>"
    COUNT = "COUNT"

    def __init__(self, model_file: Union[str, Path], read_only: bool = False) -> None:
        self._model_file = model_file
        self._model = h5py.File(model_file, "r" if read_only else "a")

    def __del__(self) -> None:
        self._model.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._model_file}, orders={self.orders()}, read_only={self._model.mode == 'r'})"

    def orders(self) -> List[int]:
        keys = self._model.keys()
        return sorted(int(key) for key in keys if key != Model.COUNT)

    def read_from_text(
        self,
        text_file: Union[str, Path],
        orders: Optional[List[int]] = None,
        include_sentence_boundaries: bool = False,
    ) -> None:
        if orders is None:
            Model.info("No orders specified, using orders from existing model")
            orders = self.orders()
        if len(orders) == 0:
            Model.error(
                "No orders specified and no orders present in the existing model"
            )
            raise ValueError(
                "No orders specified and no orders present in the existing model"
            )
        for order in orders:
            group = self._model.require_group(str(order))
            group.require_dataset(Model.COUNT, shape=(), dtype=int, exact=True)
        with open(text_file, "rt", encoding="utf-8") as f:
            for line in f:
                self._read_from_line(line, orders, include_sentence_boundaries)

    def _read_from_line(
        self, line: str, orders: List[int], include_sentence_boundaries: bool
    ) -> None:
        tokens = tokenize(line)
        if include_sentence_boundaries:
            tokens = [Model.BOS] + tokens + [Model.EOS]
        for order in orders:
            order_group = self._model.get(str(order))
            for i in range(len(tokens) - order + 1):
                order_group[Model.COUNT][()] += 1
                ngram_tokens = tokens[i : i + order]
                self._add_ngram_tokens(order_group, ngram_tokens)

    @staticmethod
    def _add_ngram_tokens(
        current_group: h5py.Group, ngram_tokens: List[str], index: int = 0
    ) -> None:
        if index < len(ngram_tokens):
            token = ngram_tokens[index]
            token_group = current_group.require_group(token)
            token_group.require_dataset(Model.COUNT, shape=(), dtype=int, exact=True)
            token_group[Model.COUNT][()] += 1
            Model._add_ngram_tokens(token_group, ngram_tokens, index + 1)

    def prune(self, min_counts: List[int]) -> None:
        orders = self.orders()
        # pad min_counts with the last value to match the number of orders
        min_counts = min_counts + [min_counts[-1]] * (max(orders) - len(min_counts))
        for order in orders:
            Model._prune(self._model[str(order)], min_counts[order - 1])

    @staticmethod
    def _prune(current_group: h5py.Group, min_count: int) -> None:
        for key in list(current_group.keys()):
            if key == Model.COUNT or key == Model.UNK:
                continue
            token_group = current_group[key]
            count = token_group[Model.COUNT][()]
            if count < min_count:
                unk_count = current_group.require_group(Model.UNK).require_dataset(
                    Model.COUNT, shape=(), dtype=int, exact=True
                )
                unk_count[()] += count
                del current_group[key]
            else:
                Model._prune(token_group, min_count)

    def iterate_ngrams(self, order: int) -> Iterator[NGram]:
        if order not in self.orders():
            raise ValueError(f"Order {order} not found in model")
        return Model._traverse(self._model[str(order)])

    @staticmethod
    def _traverse(current_group: h5py.Group) -> Iterator[NGram]:
        tokens = [
            key
            for key in current_group.keys()
            if key != Model.COUNT and key != Model.UNK
        ]
        if len(tokens) == 0:
            yield (
                NGram(tokens=current_group.name.split("/")[2:]),
                current_group[Model.COUNT][()],
            )
        else:
            for token in tokens:
                token_group = current_group[token]
                yield from Model._traverse(token_group)

    def get_count(self, ngram: NGram) -> int:
        order_group = self._model.get(str(len(ngram)))
        if order_group is None:
            return 0
        name = "/".join(ngram) + "/" + Model.COUNT
        count = order_group.get(name)
        if count is None:
            return 0
        return count[()]

    def get_frequency(self, ngram: NGram) -> float:
        count = self.get_count(ngram)
        if count == 0:
            return 0.0
        total = self._model.get(str(len(ngram)))[Model.COUNT][()]
        return count / total

    def get_fpm(self, ngram: NGram) -> float:
        return self.get_frequency(ngram) * 1_000_000

    def conditional_distribution(self, ngram: NGram, order: int) -> Dict[str, float]:
        assert order in self.orders(), f"Order {order} not found in model"
        if order > 1:
            last_k = NGram(tokens=ngram[-order + 1 :])
            group = self._model[str(order)].get("/".join(last_k))
        else:
            group = self._model[str(order)]
        if group is None:
            return {}
        total = group[Model.COUNT][()]
        next_tokens = {
            key: group[key][Model.COUNT][()] / total
            for key in group.keys()
            if key != Model.COUNT
        }
        return next_tokens

    def conditional_probability(self, ngram: NGram, order: int) -> float:
        assert order in self.orders(), f"Order {order} not found in model"
        if order > 1:
            last_k = NGram(tokens=ngram[-order:-1])
            group = self._model[str(order)].get("/".join(last_k))
        else:
            group = self._model[str(order)]
        if group is None:
            return 0.0
        token_group = group.get(ngram[-1])
        if token_group is None:
            return 0.0
        return token_group[Model.COUNT][()] / group[Model.COUNT][()]

    def conditional_probability_with_backoff(
        self, ngram: NGram
    ) -> Tuple[float, int, bool]:
        def get_prob(ngram, orders) -> Optional[Tuple[float, int]]:
            order_idx = len(orders) - 1
            while order_idx > -1:
                prob = self.conditional_probability(ngram, orders[order_idx])
                if prob > 0:
                    return prob, orders[order_idx]
                order_idx -= 1

        orders = [o for o in self.orders() if o <= len(ngram)]
        result = get_prob(ngram, orders)
        if result is not None:
            return result[0], result[1], False

        # try again for predicting unknown
        ngram = NGram(tokens=ngram[:-1] + (Model.UNK,))
        result = get_prob(ngram, orders)
        if result is not None:
            return result[0], result[1], True

        # prob, order, is_unk
        return 0.0, 0, True

    def estimate_logprobs_per_token(
        self, ngram: NGram
    ) -> List[Tuple[str, float, int, bool]]:
        # (token, logprob, order, is_unk)
        ngrams = [NGram(tokens=ngram[:i]) for i in range(1, len(ngram))]
        result = [
            (ngram[-1],) + self.conditional_probability_with_backoff(ngram)
            for ngram in ngrams
        ]
        return result

    def estimate_logprob(self, ngram: NGram) -> float:
        logprob = 0.0
        for _, prob, _, _ in self.estimate_logprobs_per_token(ngram):
            logprob += log10(prob)
        return logprob

    def generate(
        self,
        n_tokens: int,
        order: int,
        from_bos: bool = False,
        seed: Optional[int] = None,
    ) -> NGram:
        orders = self.orders()
        if order not in orders:
            Model.error(f"Order {order} not found in model")
            raise ValueError(f"Order {order} not found in model")
        if any(o not in orders for o in range(1, order)):
            Model.error(
                f"Generation requires all lower orders to be present in the model"
            )
            raise ValueError(
                f"Generation requires all lower orders to be present in the model"
            )

        tokens = []
        if from_bos:
            assert (
                Model.BOS in self._model[str(order)]
            ), f"Model does not contain {Model.BOS} tokens"
            tokens.append(Model.BOS)
        if seed:
            np.random.seed(seed)
        while len(tokens) < n_tokens + int(from_bos):
            order_to_use = min(len(tokens) + 1, order)
            next_tokens = self.conditional_distribution(
                NGram(tokens=tokens), order_to_use
            )
            if len(next_tokens) == 0:
                break
            new_token = np.random.choice(
                list(next_tokens.keys()), p=list(next_tokens.values())
            )
            if new_token == Model.UNK:
                if next_tokens[Model.UNK] == 1.0:
                    break
                continue
            tokens.append(new_token)
        return NGram(tokens=tokens)

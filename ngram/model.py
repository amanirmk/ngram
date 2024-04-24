from math import log10, inf
from typing import Union, List, Optional, Iterator, Tuple, Dict
from pathlib import Path
from collections import defaultdict, deque
import h5py
import numpy as np
from tqdm import tqdm
from ngram.abstract import Object
from ngram.processing import tokenize
from ngram.datatypes import NGram


class Model(Object):
    UNK = "<UNK>"
    BOS = "<S>"
    EOS = "</S>"
    COUNT = "COUNT"

    def __init__(self, model_file: Union[str, Path], read_only: bool = False) -> None:
        self._model_file = Path(model_file)
        self._model = h5py.File(model_file, "r" if read_only else "a")

    def __del__(self) -> None:
        self._model.close()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self._model_file.stem}, "
            + f"orders={self.orders()}, read_only={self._model.mode == 'r'})"
        )

    def orders(self) -> List[int]:
        keys = self._model.keys()
        return sorted(int(key) for key in keys if key != Model.COUNT)

    def read_from(
        self,
        thing_to_read: Union[str, Path],
        orders: Optional[List[int]] = None,
        include_sentence_boundaries: bool = False,
        disable_tqdm: bool = True,
    ) -> None:
        thing_to_read = Path(thing_to_read)
        if thing_to_read.is_dir():
            self.read_from_folder(
                thing_to_read, orders, include_sentence_boundaries, disable_tqdm
            )
        elif thing_to_read.is_file():
            self.read_from_text(
                thing_to_read, orders, include_sentence_boundaries, disable_tqdm
            )
        else:
            Model.error(f"Cannot read from {thing_to_read}")

    def read_from_folder(
        self,
        folder: Union[str, Path],
        orders: Optional[List[int]] = None,
        include_sentence_boundaries: bool = False,
        disable_tqdm: bool = True,
    ) -> None:
        for file in tqdm(
            list(Path(folder).rglob("*.txt")),
            desc="Reading from folder",
            unit="file",
            disable=disable_tqdm,
        ):
            self.read_from_text(file, orders, include_sentence_boundaries)

    def read_from_text(
        self,
        text_file: Union[str, Path],
        orders: Optional[List[int]] = None,
        include_sentence_boundaries: bool = False,
        disable_tqdm: bool = True,
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

        num_lines = 0
        if not disable_tqdm:
            with open(text_file, "rb") as f:
                num_lines = sum(1 for _ in f)
        with open(text_file, "rt", encoding="utf-8") as f:
            for line in tqdm(
                f,
                total=num_lines,
                desc="Reading from text",
                unit="line",
                disable=disable_tqdm,
            ):
                self._read_from_line(line, orders, include_sentence_boundaries)

    def _read_from_line(
        self, line: str, orders: List[int], include_sentence_boundaries: bool
    ) -> None:
        tokens = tokenize(line)
        if include_sentence_boundaries:
            tokens = [Model.BOS] + tokens + [Model.EOS]
        ngram = NGram(tokens=tokens)
        for order in orders:
            order_group = self._model.get(str(order))
            for subgram in ngram.subgrams(order):
                order_group[Model.COUNT][()] += 1
                self._add_ngram(order_group, subgram)

    @staticmethod
    def _add_ngram(current_group: h5py.Group, ngram: NGram, index: int = 0) -> None:
        if index < len(ngram):
            token = ngram[index]
            token_group = current_group.require_group(token)
            token_group.require_dataset(Model.COUNT, shape=(), dtype=int, exact=True)
            token_group[Model.COUNT][()] += 1
            Model._add_ngram(token_group, ngram, index + 1)

    def prune(self, min_counts: List[int]) -> None:
        orders = self.orders()
        # pad min_counts with the last value to match the number of orders
        min_counts = min_counts + [min_counts[-1]] * (max(orders) - len(min_counts))
        for order in orders:
            Model._prune(self._model[str(order)], min_counts[order - 1])

    @staticmethod
    def _prune(current_group: h5py.Group, min_count: int) -> None:
        for key in list(current_group.keys()):
            if key in [Model.COUNT, Model.UNK]:
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

    def iterate_ngrams(
        self,
        order: int,
        breadth_first: bool = False,
        shuffle: bool = False,
        seed: Optional[int] = None,
        with_counts: bool = False,
    ) -> Union[Iterator[Tuple[NGram, int]], Iterator[NGram]]:
        if order not in self.orders():
            Model.error(f"Order {order} not found in model")
            raise ValueError(f"Order {order} not found in model")
        if seed is not None:
            np.random.seed(seed)
        return Model._traverse(
            self._model[str(order)], breadth_first, shuffle, with_counts
        )

    @staticmethod
    def _traverse(
        group: h5py.Group,
        breadth_first: bool = False,
        shuffle: bool = False,
        with_counts: bool = False,
    ) -> Union[Iterator[Tuple[NGram, int]], Iterator[NGram]]:
        groups = deque([group])
        while groups:
            if breadth_first:
                current_group = groups.popleft()
            else:
                current_group = groups.pop()
            tokens = [
                key
                for key in current_group.keys()
                if key not in [Model.COUNT, Model.UNK]
            ]
            if len(tokens) == 0:
                if with_counts:
                    yield (
                        NGram(tokens=current_group.name.split("/")[2:]),
                        current_group[Model.COUNT][()],
                    )
                else:
                    yield NGram(tokens=current_group.name.split("/")[2:])
            else:
                if shuffle:
                    np.random.shuffle(tokens)
                for token in tokens:
                    groups.append(current_group[token])

    def get_count(self, ngram: NGram) -> int:
        count_query = ngram.to_query() + "/" + Model.COUNT
        count = self._model.get(count_query)
        if count is None:
            return 0
        return count[()]

    def get_frequency(self, ngram: NGram) -> float:
        count = self.get_count(ngram)
        if count == 0:
            return 0.0
        total = self._model[str(len(ngram))][Model.COUNT][()]
        return count / total

    def get_fpm(self, ngram: NGram) -> float:
        return self.get_frequency(ngram) * 1_000_000

    def conditional_distribution(self, ngram: NGram, order: int) -> Dict[str, float]:
        assert order in self.orders(), f"Order {order} not found in model"
        if order > 1:
            last_k = NGram(tokens=ngram[-order + 1 :])
            group = self._model.get(last_k.to_query(order))
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
            group = self._model.get(last_k.to_query(order))
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
            return 0.0, 0

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

    def logprobs_per_token(self, ngram: NGram) -> List[Tuple[str, float, int, bool]]:
        result = [
            (subgram[-1],) + self.conditional_probability_with_backoff(subgram)
            for subgram in ngram.subgrams(len(ngram), partial=True)
        ]
        result = [
            (token, log10(prob) if prob else -inf, order, is_unk)
            for token, prob, order, is_unk in result
        ]
        return result

    def estimate_logprob(self, ngram: NGram, alpha: float = 0.4) -> float:
        # uses stupid backoff (with whatever orders present in model)
        # using alpha=0.4 from Google paper (Brants et al., 2007)
        # could probably get a heuristic for alpha based on data
        assert alpha > 0.0, "Alpha must be greater than 0"
        max_order = max(o for o in self.orders() if o <= len(ngram))
        total_logprob = 0.0
        for _, logprob, order, _ in self.logprobs_per_token(ngram):
            total_logprob += logprob + log10(alpha) * (max_order - order)
        return total_logprob

    def generate(
        self,
        max_tokens: int,
        order: Optional[int] = None,
        from_bos: bool = False,
        seed: Optional[int] = None,
    ) -> NGram:
        orders = self.orders()
        if order is None:
            order = max(orders)
        if order not in orders:
            Model.error(f"Order {order} not found in model")
            raise ValueError(f"Order {order} not found in model")
        if any(o not in orders for o in range(1, order)):
            Model.error(
                "Generation requires all lower orders to be present in the model"
            )
            raise ValueError(
                "Generation requires all lower orders to be present in the model"
            )

        tokens = []
        if from_bos:
            if Model.BOS not in self._model[str(order)]:
                Model.error(f"Model does not contain {Model.BOS} tokens")
                raise ValueError(f"Model does not contain {Model.BOS} tokens")
            tokens.append(Model.BOS)
        if seed:
            np.random.seed(seed)
        while len(tokens) < max_tokens:
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

    def get_percentiles(
        self, orders: List[int], min_counts: Optional[List[int]] = None, num: int = 400
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        if min_counts is None:
            min_counts = [0] * len(orders)

        if len(min_counts) < len(orders):
            min_counts = min_counts + [min_counts[-1]] * (max(orders) - len(min_counts))

        percentile_dict = {}
        pctiles = np.linspace(0, 100, num=num)
        for order, min_count in zip(orders, min_counts):
            counts = np.array(
                [
                    count
                    for _, count in self.iterate_ngrams(order, with_counts=True)
                    if count >= min_count  # type: ignore[operator]
                ]
            )
            pctile_vals = np.percentile(counts, pctiles)
            percentile_dict[order] = (pctile_vals, pctiles)
        return percentile_dict

    def get_percentiles_of_pairwise_differences(
        self,
        orders: List[int],
        min_counts: Optional[List[int]] = None,
        num_bins: int = 10_000,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        if min_counts is None:
            min_counts = [0] * len(orders)

        if len(min_counts) < len(orders):
            min_counts = min_counts + [min_counts[-1]] * (max(orders) - len(min_counts))

        percentile_dict = {}
        for order, min_count in zip(orders, min_counts):
            counts = np.array(
                [
                    count
                    for _, count in self.iterate_ngrams(order, with_counts=True)
                    if count >= min_count  # type: ignore[operator]
                ]
            )
            # lognormal data, so better bins in log space
            logcounts = np.log10(counts)
            # divide into equal sized bins
            bin_cnts, bin_edges = np.histogram(logcounts, bins=num_bins)
            # use center for bin estimate
            bin_vals = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist = np.stack((bin_cnts, bin_vals), axis=1)
            # compute difference distribution based on binned estimates
            diff_cnts: Dict[float, int] = defaultdict(int)
            for i in range(len(hist)):
                cnt1, val1 = hist[i]
                # same bin, so just N choose 2 counts of 0 diff
                diff_cnts[0] += int(cnt1 * (cnt1 - 1) / 2)
                for j in range(i + 1, len(hist)):
                    cnt2, val2 = hist[j]
                    # for diff bins, N1 * N2 counts of diff
                    # measure diff in freq space, not log space
                    diff_cnts[abs(10**val1 - 10**val2)] += int(cnt1 * cnt2)
            # compute percentiles
            diffs = np.array(list(diff_cnts.keys()))
            counts = np.array(list(diff_cnts.values()))
            sorted_indices = np.argsort(diffs)
            sorted_counts = counts[sorted_indices]
            cumulative_counts = np.cumsum(sorted_counts)
            pctiles = cumulative_counts / np.sum(sorted_counts) * 100
            pctile_vals = diffs[sorted_indices]
            percentile_dict[order] = (pctile_vals, pctiles)
        return percentile_dict

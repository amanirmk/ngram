from math import log10, inf
from typing import Union, List, Optional, Iterator, Tuple, Dict
from pathlib import Path
from collections import defaultdict, deque
import json
import ijson
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

    def __init__(
        self,
        model_file: Union[str, Path],
        read_only: bool = False,
        autosave: bool = False,
    ) -> None:
        self._model_file = Path(model_file)
        self._read_only = read_only
        self._autosave = autosave
        self._model: Optional[dict] = None
        self._orders: Optional[List[int]] = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self._model_file.stem}, "
            + f"orders={self.orders()}, read_only={self._read_only})"
        )

    def __del__(self) -> None:
        if not self._read_only and self._autosave:
            if self._model is not None:
                try:
                    self.save()
                    Model.info("Auto-saved model.")
                except:
                    Model.error(
                        "Unable to auto-save model. "
                        + "This is typically caused by quitting Python."
                    )

    def save(self, model_file: Optional[Union[str, Path]] = None) -> None:
        Model.info("Saving model.")
        self._require_model()
        if model_file is None or Path(model_file) == self._model_file:
            if self._read_only:
                Model.error("Model is read-only")
                raise ValueError("Model is read-only")
            with open(self._model_file, "w", encoding="utf-8") as f:
                json.dump(self._model, f)
        else:
            with open(model_file, "w", encoding="utf-8") as f:
                json.dump(self._model, f)

    def load_into_memory(self) -> None:
        self._require_model()

    def _require_model(self) -> None:
        if self._model is None:
            if not self._model_file.exists():
                self._model = {}
            else:
                with open(self._model_file, "r", encoding="utf-8") as f:
                    self._model = json.load(f)

    @staticmethod
    def _require_group(group: dict, key: Union[int, str]) -> dict:
        return group.setdefault(str(key), {Model.COUNT: 0})

    def _get_group(self, query: str) -> Optional[dict]:
        if self._model is not None:
            return self._get_group_from_model(query)
        with open(self._model_file, "rb") as f:
            try:
                group = next(ijson.items(f, query))
            except:
                group = None
            return group

    def _get_group_from_model(self, query: str) -> Optional[dict]:
        assert self._model is not None
        group: Optional[dict] = self._model
        for key in query.split("."):
            if group is None:
                return None
            group = group.get(key)
        return group

    def orders(self) -> List[int]:
        if self._model is not None:
            return sorted([int(k) for k in self._model.keys()])
        if self._orders is None:
            with open(self._model_file, "rb") as f:
                self._orders = sorted(list(int(k) for k, _ in ijson.kvitems(f, "")))
        return self._orders

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
            Model.error(
                f"Cannot read from {thing_to_read}. "
                + "Currently only files and folders are supported."
            )

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
        self._require_model()
        if orders is None:
            Model.info("No orders specified, using orders from model.")
            orders = self.orders()
        if len(orders) == 0:
            Model.error("No orders specified and no orders present in the model.")
            raise ValueError("No orders specified and no orders present in the model.")

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
        assert self._model is not None
        tokens = tokenize(line)
        if include_sentence_boundaries:
            tokens = [Model.BOS] + tokens + [Model.EOS]
        ngram = NGram(tokens=tokens)
        for order in orders:
            order_group = Model._require_group(self._model, order)
            for subgram in ngram.subgrams(order):
                order_group[Model.COUNT] += 1
                self._add_ngram(order_group, subgram)

    @staticmethod
    def _add_ngram(current_group: dict, ngram: NGram, index: int = 0) -> None:
        if index < len(ngram):
            token = ngram[index]
            token_group = Model._require_group(current_group, token)
            token_group[Model.COUNT] += 1
            Model._add_ngram(token_group, ngram, index + 1)

    def prune(self, min_counts: List[int]) -> None:
        self._require_model()
        assert self._model is not None
        orders = self.orders()
        # pad min_counts with the last value to match the number of orders
        min_counts = min_counts + [min_counts[-1]] * (max(orders) - len(min_counts))
        for order in orders:
            Model._prune(
                Model._require_group(self._model, order), min_counts[order - 1]
            )

    @staticmethod
    def _prune(current_group: dict, min_count: int) -> None:
        for key in list(current_group.keys()):
            if key in [Model.UNK, Model.COUNT]:
                continue
            token_group = current_group[key]
            count = token_group[Model.COUNT]
            if count < min_count:
                unk_group = Model._require_group(current_group, Model.UNK)
                unk_group[Model.COUNT] += count
                del current_group[key]
            else:
                Model._prune(token_group, min_count)

    def iterate_ngrams(
        self,
        order: int,
        mode: str = "standard",
        seed: Optional[int] = None,
        with_counts: bool = False,
    ) -> Union[Iterator[Tuple[NGram, int]], Iterator[NGram]]:
        assert mode in [
            "standard",
            "greedy_shuffle",
            "greedy_ascend",
            "greedy_descend",
            "shuffle",
            "ascend",
            "descend",
        ]
        if order not in self.orders():
            Model.error(f"Order {order} not found in model.")
            raise ValueError(f"Order {order} not found in model.")
        if seed is not None:
            np.random.seed(seed)
        group = self._get_group(str(order))
        assert group is not None
        if mode == "standard" or mode.startswith("greedy"):
            yield from Model._traverse(group, mode=mode, with_counts=with_counts)
        else:
            ngrams: List[Tuple[NGram, int]] = list(
                Model._traverse(group, "standard", with_counts=True)  # type: ignore[arg-type]
            )
            if mode == "shuffle":
                np.random.shuffle(ngrams)  # type: ignore[arg-type]
            elif mode == "ascend":
                ngrams = sorted(ngrams, key=lambda x: x[1])
            elif mode == "descend":
                ngrams = sorted(ngrams, key=lambda x: x[1], reverse=True)
            for ngram, count in ngrams:
                if with_counts:
                    yield ngram, count
                else:
                    yield ngram

    @staticmethod
    def _traverse(
        group: dict,
        mode: str = "standard",
        with_counts: bool = False,
        init_tokens: Optional[List[str]] = None,
    ) -> Union[Iterator[Tuple[NGram, int]], Iterator[NGram]]:
        groups: deque = deque([(group, [] if init_tokens is None else init_tokens)])
        while groups:
            current_group, current_tokens = groups.pop()
            next_tokens = [key for key in current_group.keys() if key != Model.COUNT]
            if len(next_tokens) == 0:
                if with_counts:
                    yield (
                        NGram(tokens=current_tokens),
                        current_group[Model.COUNT],
                    )
                else:
                    yield NGram(tokens=current_tokens)
            else:
                if mode == "greedy_shuffle":
                    np.random.shuffle(next_tokens)
                elif mode in ["greedy_ascend", "greedy_descend"]:
                    next_tokens = sorted(
                        next_tokens,
                        key=lambda x: current_group[x][Model.COUNT],
                        reverse=mode.endswith("ascend"),
                    )
                for token in next_tokens:
                    if token != Model.UNK:
                        groups.append((current_group[token], current_tokens + [token]))

    def get_count(self, ngram: NGram) -> int:
        group = self._get_group(ngram.to_query())
        if group is None:
            return 0
        return group[Model.COUNT]

    def get_order_count(self, order: int) -> int:
        group = self._get_group(str(order))
        if group is None:
            return 0
        return group[Model.COUNT]

    def get_frequency(self, ngram: NGram) -> float:
        count = self.get_count(ngram)
        if count == 0:
            return 0.0
        total = self.get_order_count(len(ngram))
        return count / total

    def get_fpm(self, ngram: NGram) -> float:
        return self.get_frequency(ngram) * 1_000_000

    def get_logprob(self, ngram: NGram) -> float:
        return log10(self.get_frequency(ngram))

    def conditional_distribution(self, ngram: NGram, order: int) -> Dict[str, float]:
        assert order in self.orders(), f"Order {order} not found in model"
        if order > 1:
            last_k = NGram(tokens=ngram[-order + 1 :])
            group = self._get_group(last_k.to_query(order))
        else:
            group = self._get_group(str(order))
        if group is None:
            return {}
        total = group[Model.COUNT]
        next_tokens = {
            key: group[key][Model.COUNT] / total
            for key in group.keys()
            if key != Model.COUNT
        }
        return next_tokens

    def conditional_probability(self, ngram: NGram, order: int) -> float:
        assert order in self.orders(), f"Order {order} not found in model"
        if order > 1:
            last_k = NGram(tokens=ngram[-order:-1])
            group = self._get_group(last_k.to_query(order))
        else:
            group = self._get_group(str(order))
        if group is None:
            return 0.0
        token_group = group.get(ngram[-1])
        if token_group is None:
            return 0.0
        return token_group[Model.COUNT] / group[Model.COUNT]

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
            Model.error(f"Order {order} not found in model.")
            raise ValueError(f"Order {order} not found in model.")
        if any(o not in orders for o in range(1, order)):
            Model.error("Generation requires that all lower orders are in the model.")
            raise ValueError(
                "Generation requires that all lower orders are in the model."
            )

        tokens = []
        if from_bos:
            group = self._get_group(str(order))
            assert group is not None
            if Model.BOS not in group:
                Model.error(f"Model does not contain {Model.BOS} tokens.")
                raise ValueError(f"Model does not contain {Model.BOS} tokens.")
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
            min_counts = [0] * max(orders)
        elif len(min_counts) < max(orders):
            min_counts = min_counts + [min_counts[-1]] * (max(orders) - len(min_counts))

        percentile_dict = {}
        pctiles = np.linspace(0, 100, num=num)
        for order in orders:
            counts = np.array(
                [
                    count
                    for _, count in self.iterate_ngrams(order, with_counts=True)
                    if count >= min_counts[order - 1]  # type: ignore[operator]
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
            min_counts = [0] * max(orders)
        elif len(min_counts) < max(orders):
            min_counts = min_counts + [min_counts[-1]] * (max(orders) - len(min_counts))

        percentile_dict = {}
        for order in orders:
            counts = np.array(
                [
                    count
                    for _, count in self.iterate_ngrams(order, with_counts=True)
                    if count >= min_counts[order - 1]  # type: ignore[operator]
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

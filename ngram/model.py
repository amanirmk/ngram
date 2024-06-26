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


class Model(Object):  # pylint: disable=too-many-public-methods
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
                except Exception as e:  # pylint: disable=broad-exception-caught
                    Model.error(
                        "Unable to auto-save model. "
                        + "This is typically caused by quitting Python."
                        + f"Error: {e}"
                    )

    def save(self, model_file: Optional[Union[str, Path]] = None) -> None:
        Model.info("Saving model.")
        self._require_model()
        if model_file is None or Path(model_file) == self._model_file:
            if self._read_only:
                Model.error("Model is read-only, cannot write to its own file.")
                raise ValueError("Model is read-only, cannot write to its own file.")
            Path(self._model_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self._model_file, "w", encoding="utf-8") as f:
                json.dump(self._model, f)
        else:
            Path(model_file).parent.mkdir(parents=True, exist_ok=True)
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
            except StopIteration:
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
        location: Union[str, Path],
        orders: Optional[List[int]] = None,
        include_sentence_boundaries: bool = False,
        disable_tqdm: bool = True,
    ) -> None:
        location = Path(location)
        if location.is_dir():
            self._read_from_folder(
                location, orders, include_sentence_boundaries, disable_tqdm
            )
        elif location.is_file():
            self._read_from_text(
                location, orders, include_sentence_boundaries, disable_tqdm
            )
        else:
            Model.error(
                f"Cannot read from {location}. "
                + "Currently only files and folders are supported."
            )

    def _read_from_folder(
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
            self._read_from_text(file, orders, include_sentence_boundaries)

    def _read_from_text(
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

    def _get_prob(self, ngram, orders) -> Optional[Tuple[float, int]]:
        order_idx = len(orders) - 1
        while order_idx > -1:
            prob = self.conditional_probability(ngram, orders[order_idx])
            if prob > 0:
                return prob, orders[order_idx]
            order_idx -= 1
        return 0.0, 0

    def conditional_probability_with_backoff(
        self, ngram: NGram
    ) -> Tuple[float, int, bool]:
        orders = [o for o in self.orders() if o <= len(ngram)]
        result = self._get_prob(ngram, orders)
        if result is not None:
            return result[0], result[1], False

        # try again for predicting unknown
        ngram = NGram(tokens=ngram[:-1] + (Model.UNK,))
        result = self._get_prob(ngram, orders)
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

    def extend(  # pylint: disable=too-many-branches
        self,
        ngram: NGram,
        tokens_to_add: int,
        allow_eos: bool = False,
        order: Optional[int] = None,
        flexible_order: bool = True,
        min_prob: float = 0.0,
        max_prob: float = 1.0,
        mode: str = "sample",
        seed: Optional[int] = None,
    ) -> NGram:
        if mode not in ["sample", "maximize", "minimize"]:
            Model.error("Mode must be one of 'sample', 'maximize', or 'minimize'")
            raise ValueError("Mode must be one of 'sample', 'maximize', or 'minimize'")

        if seed is not None:
            np.random.seed(seed)

        while tokens_to_add > 0:
            if order is None:
                # start with longest order possible
                order_to_use = max(
                    o for o in [0] + self.orders() if o <= len(ngram) + 1
                )
            else:
                # start with fixed order
                order_to_use = order

            if order_to_use == 0:
                Model.error(
                    "There are no orders in the model that can be used to extend the ngram."
                )
                raise ValueError(
                    "There are no orders in the model that can be used to extend the ngram."
                )

            if order_to_use > len(ngram) + 1:
                Model.error("Order is too high to extend the ngram.")
                raise ValueError("Order is too high to extend the ngram.")

            # try decreasing orders until one is found that works (if flexible_order is True)
            while True:
                next_tokens = self.conditional_distribution(ngram, order_to_use)

                # remove special tokens
                next_tokens = {
                    k: v
                    for k, v in next_tokens.items()
                    if k
                    not in [Model.BOS, Model.UNK]
                    + ([Model.EOS] if not allow_eos else [])
                }
                # filter by probability
                if min_prob > 0.0 or max_prob < 1.0:
                    next_tokens = {
                        k: v
                        for k, v in next_tokens.items()
                        if min_prob <= v <= max_prob
                    }

                # tokens found, exit loop
                if len(next_tokens) > 0:
                    break

                # no tokens found, should i quit?
                if order_to_use == 0 or not flexible_order:
                    Model.warn(
                        "No tokens found to extend the stimulus. "
                        + "Returning the stimulus with <FILLER> tokens."
                    )
                    ngram = NGram(tokens=ngram.tokens() + ("<FILLER>",) * tokens_to_add)
                    return ngram

                # try with lower order
                order_to_use = max(o for o in [0] + self.orders() if o < order_to_use)

            # add token
            if mode == "sample":
                next_token = np.random.choice(
                    list(next_tokens.keys()), p=list(next_tokens.values())
                )
            else:
                next_token = min(
                    next_tokens,
                    key=lambda k: next_tokens[k] * (-1 if mode == "maximize" else 1),
                )
            ngram = NGram(tokens=ngram.tokens() + (next_token,))

            if next_token == Model.EOS:
                break

            tokens_to_add -= 1

        return ngram

    def generate(
        self,
        max_tokens: int,
        from_bos: bool = False,
        seed: Optional[int] = None,
    ) -> NGram:
        if from_bos:
            group = self._get_group(str(self.orders()[0]))
            assert group is not None
            if Model.BOS not in group:
                Model.error(f"Model does not contain {Model.BOS} tokens.")
                raise ValueError(f"Model does not contain {Model.BOS} tokens.")
            ngram = NGram(tokens=[Model.BOS])
        else:
            ngram = NGram(tokens=[])

        return self.extend(
            ngram,
            max_tokens,
            allow_eos=True,
            flexible_order=True,
            mode="sample",
            seed=seed,
        )

    def get_percentiles(
        self,
        orders: List[int],
        min_counts: Optional[List[int]] = None,
        chop_percent: float = 0.0,
        num: int = 400,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        percentile_dict = {}
        for order, counts in self._get_counts_for_percentiles(
            orders, min_counts, chop_percent
        ):
            percentile_dict[order] = self._percentiles(counts, num)
        return percentile_dict

    def get_percentiles_of_pairwise_differences(
        self,
        orders: List[int],
        min_counts: Optional[List[int]] = None,
        chop_percent: float = 0.0,
        num_bins: int = 10_000,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        percentile_of_difference_dict = {}
        for order, counts in self._get_counts_for_percentiles(
            orders, min_counts, chop_percent
        ):
            percentile_of_difference_dict[
                order
            ] = self._percentiles_of_pairwise_differences(counts, num_bins)
        return percentile_of_difference_dict

    def get_all_percentiles(
        self,
        orders: List[int],
        min_counts: Optional[List[int]] = None,
        chop_percent: float = 0.0,
        num: int = 400,
        num_bins: int = 10_000,
    ) -> Tuple[
        Dict[int, Tuple[np.ndarray, np.ndarray]],
        Dict[int, Tuple[np.ndarray, np.ndarray]],
    ]:
        percentile_dict = {}
        percentile_of_difference_dict = {}
        for order, counts in self._get_counts_for_percentiles(
            orders, min_counts, chop_percent
        ):
            # percentile
            percentile_dict[order] = self._percentiles(counts, num)
            # percentile of difference
            percentile_of_difference_dict[
                order
            ] = self._percentiles_of_pairwise_differences(counts, num_bins)
        return percentile_dict, percentile_of_difference_dict

    def _get_counts_for_percentiles(
        self,
        orders: List[int],
        min_counts: Optional[List[int]] = None,
        chop_percent: float = 0.0,
    ) -> Iterator[Tuple[int, List[int]]]:
        if min_counts is None:
            min_counts = [0] * max(orders)
        elif len(min_counts) < max(orders):
            min_counts = min_counts + [min_counts[-1]] * (max(orders) - len(min_counts))

        for order in orders:
            counts = [
                count
                for _, count in self.iterate_ngrams(order, with_counts=True)
                if count >= min_counts[order - 1]  # type: ignore[operator]
            ]
            if chop_percent > 0:
                counts.sort()
                counts = counts[int(len(counts) * (chop_percent / 100)) :]
            yield order, counts  # type: ignore[misc]

    @staticmethod
    def _percentiles(counts: List[int], num: int) -> Tuple[np.ndarray, np.ndarray]:
        pctiles = np.linspace(0, 100, num=num)
        pctile_vals = np.percentile(np.array(counts), pctiles)
        return pctile_vals, pctiles

    @staticmethod
    def _percentiles_of_pairwise_differences(
        counts: List[int], num_bins: int
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        counts_arr = np.array(list(diff_cnts.values()))
        sorted_indices = np.argsort(diffs)
        sorted_counts = counts_arr[sorted_indices]
        cumulative_counts = np.cumsum(sorted_counts)
        pctiles = cumulative_counts / np.sum(sorted_counts) * 100
        pctile_vals = diffs[sorted_indices]
        return pctile_vals, pctiles

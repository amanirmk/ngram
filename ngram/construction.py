from typing import Dict, Iterable, List, Optional, Union
from pathlib import Path
from itertools import combinations, islice
from ngram.abstract import Object
from ngram.model import Model
import numpy as np
import pandas as pd


class Construct(Object):
    pass


def construct_candidates(
    model: Model,
    order: int,
    n_candidates: int,
    max_per_prefix: int = 10,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)
    candidate_pairs = []
    for prefix in model.iterate_ngrams(
        order=order - 1, breadth_first=True, shuffle=True
    ):
        query = prefix.to_query(order=order)
        group = model._model.get(query)
        ngrams = islice(
            model._traverse(group, breadth_first=True, shuffle=True), max_per_prefix
        )
        for n1, n2 in combinations(ngrams, 2):
            if len(candidate_pairs) >= n_candidates:
                break
            candidate_pairs.append((n1, n2))
    return pd.DataFrame(candidate_pairs, columns=["high_item", "low_item"])


def construct():
    # construct
    # analyze(
    #     model_file: Union[str, Path],
    #     input_file: Union[str, Path],
    #     cols: List[str, str],
    #     output_file: Union[str, Path],
    #     min_counts_for_percentile: Optional[List[int]] = None,
    # )
    pass

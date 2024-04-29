from typing import List, Optional, Union, Tuple
from pathlib import Path
from itertools import combinations, islice
from tqdm import tqdm
import numpy as np
import pandas as pd
from ngram.abstract import Object
from ngram.model import Model
from ngram.analysis import analyze


class Construct(Object):
    pass


def construct_candidates(
    model: Model,
    length: int,
    n_candidates: int,
    max_per_prefix: int = 10,
    seed: Optional[int] = None,
    disable_tqdm: bool = False,
) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)
    candidate_pairs: List[Tuple[str, str]] = []
    pbar = tqdm(
        total=n_candidates,
        desc="Constructing candidates",
        unit="pair",
        disable=disable_tqdm,
    )
    for prefix in model.iterate_ngrams(
        order=length - 1, mode="descend"  # do greedy_descend for less memory
    ):
        query = prefix.to_query(order=length)  # type: ignore[union-attr]
        group = model._get_group(query)
        if group is not None:
            ngrams = islice(
                model._traverse(
                    group,
                    mode="greedy_descend",
                    init_tokens=list(prefix.tokens()),  # type: ignore[union-attr]
                ),
                max_per_prefix,
            )
            for n1, n2 in combinations(ngrams, 2):
                if len(candidate_pairs) >= n_candidates:
                    break
                candidate_pairs.append((n1.text(), n2.text()))  # type: ignore[attr-defined]
                pbar.update(1)
    return pd.DataFrame(candidate_pairs, columns=["high_item", "low_item"])


def construct(
    model_file: Union[str, Path],
    output_file: Union[str, Path],
    length: int,
    n_candidates: int,
    max_per_prefix: int = 10,
    min_counts_for_percentile: Optional[List[int]] = None,
    seed: Optional[int] = None,
    disable_tqdm: bool = False,
    load_into_memory: bool = True,
):
    model = Model(model_file, read_only=True)
    if load_into_memory:
        model.load_into_memory()
    construct_candidates(
        model=model,
        length=length,
        n_candidates=n_candidates,
        max_per_prefix=max_per_prefix,
        seed=seed,
        disable_tqdm=disable_tqdm,
    ).to_csv(output_file, index=False)
    analyze(
        model_file=model_file,
        input_file=output_file,
        cols=["high_item", "low_item"],
        output_file=output_file,
        min_counts_for_percentile=min_counts_for_percentile,
        disable_tqdm=disable_tqdm,
    )

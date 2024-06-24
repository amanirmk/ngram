from typing import List, Optional, Union, Tuple
from pathlib import Path
from itertools import combinations
from tqdm import tqdm
import pandas as pd
from ngram.abstract import Object
from ngram.model import Model
from ngram.analysis import analyze


class Construct(Object):
    pass


def construct_candidates(
    model: Model,
    length: int,
    n_candidates: Optional[int] = None,
    max_per_prefix: Optional[int] = None,
    min_candidate_fpm: float = 0.0,
    disable_tqdm: bool = False,
) -> pd.DataFrame:
    candidate_pairs: List[Tuple[str, str]] = []
    if n_candidates is not None:
        pbar = tqdm(
            total=n_candidates,
            desc="Constructing candidates",
            unit="pair",
            disable=disable_tqdm,
        )
    for prefix in model.iterate_ngrams(order=length - 1, mode="descend"):
        query = prefix.to_query(order=length)  # type: ignore[union-attr]
        group = model._get_group(query)  # pylint: disable=protected-access
        if group is not None:
            ngrams = []
            for ng in model._traverse(  # pylint: disable=protected-access
                group,
                mode="greedy_descend",
                init_tokens=list(prefix.tokens()),  # type: ignore[union-attr]
            ):
                if (
                    min_candidate_fpm == 0
                    or model.get_fpm(ng) >= min_candidate_fpm  # type: ignore[arg-type]
                ):
                    ngrams.append(ng)
                if max_per_prefix is not None and len(ngrams) >= max_per_prefix:
                    break
            for n1, n2 in combinations(ngrams, 2):
                if n_candidates is not None and len(candidate_pairs) >= n_candidates:
                    break
                candidate_pairs.append((n1.text(), n2.text()))  # type: ignore[union-attr]
                if n_candidates is not None:
                    pbar.update(1)
        if n_candidates is not None and len(candidate_pairs) >= n_candidates:
            break
    return pd.DataFrame(candidate_pairs, columns=["high_item", "low_item"])


def construct(
    model_file: Union[str, Path],
    output_file: Union[str, Path],
    length: int,
    n_candidates: Optional[int] = None,
    max_per_prefix: Optional[int] = None,
    min_counts_for_percentile: Optional[List[int]] = None,
    min_candidate_fpm: float = 0.0,
    chop_percent: float = 0.0,
    disable_tqdm: bool = False,
):
    model = Model(model_file, read_only=True)
    model.load_into_memory()
    construct_candidates(
        model=model,
        length=length,
        n_candidates=n_candidates,
        max_per_prefix=max_per_prefix,
        min_candidate_fpm=min_candidate_fpm,
        disable_tqdm=disable_tqdm,
    ).to_csv(output_file, index=False)
    analyze(
        model_file=model_file,
        input_file=output_file,
        cols=["high_item", "low_item"],
        output_file=output_file,
        min_counts_for_percentile=min_counts_for_percentile,
        chop_percent=chop_percent,
        disable_tqdm=disable_tqdm,
        actual_model=model,  # so won't load twice
    )

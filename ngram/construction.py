import typing
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from ngram.processing import read_ngram_file
from ngram.datatypes import NGram, StimulusPair
from ngram.abstract import Object
from ngram.analysis import analyze_stimuli_pair_data, load_models_and_percentiles
import pandas as pd


class Construct(Object):
    pass


def create_prefix_query(
    ngrams: typing.Iterable[NGram],
) -> typing.Dict[str, typing.List[str]]:
    prefix_query = defaultdict(list)
    for ngram in ngrams:
        prefix = " ".join(ngram.tokens()[:-1])
        prefix_query[prefix].append(ngram.text())
    return prefix_query


def create_candidates(
    ngram_file: typing.Union[str, Path],
    prefix_file: typing.Optional[typing.Union[str, Path]] = None,
    n_candidates: int = 10000,
    min_fpm: float = 0,
    disable_tqdm: bool = False,
) -> typing.Iterable[StimulusPair]:
    ngrams = (  # type: ignore[misc]
        ngram
        for _, ngram in read_ngram_file(
            ngram_file=ngram_file,
            min_fpm=min_fpm,
            only_freqs=False,
            exclude_bos=True,
            exclude_eos=True,
            disable_tqdm=disable_tqdm,
        )
    )
    prefix_query = create_prefix_query(ngrams)

    if prefix_file:
        # this has the advantage of pre-sorting by more likely prefixes
        prefixes = (  # type: ignore[misc]
            prefix.text()
            for _, prefix in read_ngram_file(
                ngram_file=prefix_file,
                min_fpm=min_fpm,  # prefix should always have greater fpm than full ngram, so fine to cut off
                only_freqs=False,
                exclude_bos=True,
                exclude_eos=True,
                disable_tqdm=disable_tqdm,
            )
        )
    else:
        prefixes = (prefix for prefix in prefix_query.keys())

    # TODO:
    # determine smart ways to sample from good candidates instead of enumerating since large space
    # maybe sample from prefixes proportional to their frequency without replacement
    # and only get first and last K from full_texts

    n = 0
    for prefix in prefixes:
        if prefix in prefix_query:
            full_texts = prefix_query[prefix]
            for (i, s1), (j, s2) in combinations(enumerate(full_texts), 2):
                # they're ordered by frequency, so high item is first
                if i < j:
                    yield StimulusPair(high_item=s1, low_item=s2)
                else:
                    yield StimulusPair(high_item=s2, low_item=s1)
                n += 1
                if n == n_candidates:
                    break
        if n == n_candidates:
            break
    if n < n_candidates:
        Construct.warn(
            f"Only created {n} candidates, fewer than requested {n_candidates}"
        )


def construct(
    model_files_folder: typing.Union[str, Path],
    ngram_file: typing.Union[str, Path],
    prefix_file: typing.Optional[typing.Union[str, Path]] = None,
    output_file: typing.Union[str, Path] = "constructed_pairs.csv",
    max_n: typing.Optional[int] = None,
    n_candidates: int = 10000,
    min_fpm: float = 0,
    disable_tqdm: bool = False,
):
    binary_files = list(Path(model_files_folder).rglob("*.binary"))
    Construct.info(
        f"Found {len(binary_files)} binary files to build models in {model_files_folder}"
    )
    ngram_files = list(Path(model_files_folder).rglob("*.ngram"))
    Construct.info(
        f"Found {len(ngram_files)} ngram files to compute percentiles in {model_files_folder}"
    )
    models, percentile_dict, diff_percentile_dict = load_models_and_percentiles(
        binary_files,
        ngram_files,
        max_n=max_n,
        include_diff=True,
        percentile_min_fpm=min_fpm,
        disable_tqdm=disable_tqdm,
    )
    pairs = create_candidates(
        ngram_file=ngram_file,
        prefix_file=prefix_file,
        n_candidates=n_candidates,
        min_fpm=min_fpm,
        disable_tqdm=disable_tqdm,
    )
    analyze_stimuli_pair_data(
        pairs=pairs,
        models=models,
        percentile_dict=percentile_dict,
        diff_percentile_dict=diff_percentile_dict,
        csv_file=output_file,
        disable_tqdm=disable_tqdm,
    )
    Construct.info(f"Constructed pairs saved to {output_file}")
    stimuli_pairs = pd.read_csv(output_file)

    orders = sorted((m._order for m in models), reverse=True)
    Construct.info(
        f"Maximizing difference for order {orders[0]} while minimizing for orders {orders[1:]}"
    )
    minimize_cols = [f"final_diff_percentile_{k}" for k in orders[1:]]
    max_to_minimize = stimuli_pairs[minimize_cols].max(axis=1)
    to_maximize = stimuli_pairs[f"final_diff_percentile_{orders[0]}"]
    goodness = to_maximize - max_to_minimize

    # TODO: better measure for goodness of fit
    stimuli_pairs["goodness"] = goodness
    stimuli_pairs.sort_values("goodness", ascending=False, inplace=True)

    stimuli_pairs.to_csv(output_file, index=False)
    Construct.info(f"Ordered pairs re-saved to {output_file}")

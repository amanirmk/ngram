from typing import List, Tuple, Dict, Any, Union, Iterable, Optional
from itertools import combinations
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from ngram.model import Model
from ngram.processing import process_text, read_ngram_file
from ngram.datatypes import StimulusPair
from ngram.abstract import Object


class Analyze(Object):
    pass


def get_percentiles(
    ngram_file: Union[str, Path],
    num: int = 400,
    min_fpm: float = 0,
    exclude_bos: bool = True,
    exclude_eos: bool = True,
    disable_tqdm: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    freqs = np.array(
        list(
            read_ngram_file(
                ngram_file=ngram_file,
                min_fpm=min_fpm,
                only_freqs=True,
                exclude_bos=exclude_bos,
                exclude_eos=exclude_eos,
                disable_tqdm=disable_tqdm,
            )
        )
    )
    pctiles = np.linspace(0, 100, num=num)
    pctile_vals = np.percentile(freqs, pctiles)
    return pctile_vals, pctiles


def get_diff_percentiles(
    ngram_file: Union[str, Path],
    num_bins: int = 10_000,
    min_fpm: float = 0,
    exclude_bos: bool = True,
    exclude_eos: bool = True,
    disable_tqdm: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    freqs = np.array(
        list(
            read_ngram_file(
                ngram_file=ngram_file,
                min_fpm=min_fpm,
                only_freqs=True,
                exclude_bos=exclude_bos,
                exclude_eos=exclude_eos,
                disable_tqdm=disable_tqdm,
            )
        )
    )
    logfreqs = np.log10(freqs)  # lognormal data, so better bins
    # divide into equal sized bins
    bin_cnts, bin_edges = np.histogram(logfreqs, bins=num_bins)
    # use center for bin estimate
    bin_vals = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist = np.stack((bin_cnts, bin_vals), axis=1)
    # compute difference distribution based on binned estimates
    diff_cnts: Dict[float, int] = defaultdict(int)
    with tqdm(
        total=int((len(hist) * (len(hist) - 1)) / 2),
        desc="Computing diff percentiles",
        disable=disable_tqdm,
    ) as pbar:
        for i in range(len(hist)):
            cnt1, val1 = hist[i]
            # same bin, so just N choose 2 counts of 0 diff
            diff_cnts[0] += int(cnt1 * (cnt1 - 1) / 2)
            for j in range(i + 1, len(hist)):
                cnt2, val2 = hist[j]
                # for diff bins, N1 * N2 counts of diff
                # measure diff in freq space, not log space
                diff_cnts[abs(10**val1 - 10**val2)] += int(cnt1 * cnt2)
                pbar.update(1)
    # compute percentiles
    diffs = np.array(list(diff_cnts.keys()))
    counts = np.array(list(diff_cnts.values()))
    sorted_indices = np.argsort(diffs)
    sorted_counts = counts[sorted_indices]
    cumulative_counts = np.cumsum(sorted_counts)
    pctiles = cumulative_counts / np.sum(sorted_counts) * 100
    pctile_vals = diffs[sorted_indices]
    return pctile_vals, pctiles


def percentile(
    freq_per_mil: float, pctile_vals: np.ndarray, pctiles: np.ndarray
) -> float:
    return float(np.interp(freq_per_mil, pctile_vals, pctiles).item())


def average_percentile(
    freq_per_mils: List[float], pctile_vals: np.ndarray, pctiles: np.ndarray
) -> float:
    return float(
        np.mean([percentile(fpm, pctile_vals, pctiles) for fpm in freq_per_mils])
    )


def analyze_single_stimulus_with_unigram(
    stimulus: str, any_model: Model, bos: bool = False, eos: bool = False
) -> Tuple[List[float], float, List[float], float, bool, bool]:
    scores = list(
        any_model.approximate_subgram_full_scores(stimulus, 1, bos=bos, eos=eos)
    )
    freq_per_mil_by_ngram = any_model.ngram_freqs(stimulus, n=1, bos=bos, eos=eos)
    any_oov = any(s[2] for s in scores)
    any_backed_off = False
    logprob = sum(s[0] for s in scores)
    freq_per_mil = 10**logprob * 1_000_000
    lobprob_by_token = [s[0] for s in scores]
    return (
        lobprob_by_token,
        logprob,
        freq_per_mil_by_ngram,
        freq_per_mil,
        any_oov,
        any_backed_off,
    )


def analyze_single_stimulus_with_model(
    stimulus: str,
    model: Model,
    bos: bool = False,
    eos: bool = False,
) -> Tuple[List[float], float, List[float], float, bool, bool]:
    scores = list(model.full_scores(stimulus, bos=bos, eos=eos))
    freq_per_mil_by_ngram = model.ngram_freqs(stimulus, bos=bos, eos=eos)
    any_oov = any(s[2] for s in scores)
    any_backed_off = any(s[1] < model._order for s in scores[model._order - 1 :])
    logprob = sum(s[0] for s in scores)
    freq_per_mil = 10**logprob * 1_000_000
    lobprob_by_token = [s[0] for s in scores]
    return (
        lobprob_by_token,
        logprob,
        freq_per_mil_by_ngram,
        freq_per_mil,
        any_oov,
        any_backed_off,
    )


def add_single_stimulus_results(
    results,
    analyze_output: Tuple[List[float], float, List[float], float, bool, bool],
    percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    k: int,
) -> Dict[str, Any]:
    (
        logprob_by_token,
        logprob,
        freq_per_mil_by_ngram,
        freq_per_mil,
        any_oov,
        any_backed_off,
    ) = analyze_output
    results[f"logprob_by_token_{k}"] = logprob_by_token
    results[f"logprob_{k}"] = logprob
    results[f"freq_per_mil_by_ngram_{k}"] = freq_per_mil_by_ngram
    results[f"freq_per_mil_{k}"] = freq_per_mil
    results[f"any_oov_{k}"] = any_oov
    results[f"any_backed_off_{k}"] = any_backed_off
    if k in percentile_dict:
        results[f"avg_percentile_{k}"] = average_percentile(
            freq_per_mil_by_ngram, *percentile_dict[k]
        )
    return results


def analyze_single_stimulus_with_multiple_models(
    stimulus: str,
    models: List[Model],
    percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    include_unigram: bool = True,
    bos: bool = False,
    eos: bool = False,
) -> Dict[str, Any]:
    tmp = process_text(stimulus)
    tokenized_stimulus = (models[0].BOS + " ") * bos + tmp + (" " + models[0].EOS) * eos
    results: Dict[str, Any] = {
        "stimulus": stimulus,
        "tokenized_stimulus": tokenized_stimulus,
    }
    if tmp == "":
        Analyze.warn(f"Tokenization left no stimulus text: {stimulus}")
        return results
    if include_unigram:
        partial_results = analyze_single_stimulus_with_unigram(
            stimulus, models[0], bos=bos, eos=eos
        )
        results = add_single_stimulus_results(
            results, partial_results, percentile_dict, 1
        )
    for model in models:
        partial_results = analyze_single_stimulus_with_model(
            stimulus, model, bos=bos, eos=eos
        )
        results = add_single_stimulus_results(
            results, partial_results, percentile_dict, model._order
        )
    return results


def analyze_single_stimuli_data(
    stimuli: Iterable[str],
    models: List[Model],
    percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    csv_file: Union[Path, str] = "results.csv",
    bos: bool = False,
    eos: bool = False,
    disable_tqdm: bool = False,
) -> pd.DataFrame:
    results = []
    for stimulus in tqdm(stimuli, desc="Analyzing stimuli", disable=disable_tqdm):
        results.append(
            analyze_single_stimulus_with_multiple_models(
                stimulus, models, percentile_dict=percentile_dict, bos=bos, eos=eos
            )
        )
    Path(csv_file).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(csv_file, index=False)


def add_stimuli_pair_results(
    results,
    analyze_output: Tuple[float, bool, float, float],
    percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    diff_percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    k: int,
):
    diff, both_in_vocab, fpm1, fpm2 = analyze_output
    results[f"high_item_final_freq_{k}"] = fpm1
    results[f"low_item_final_freq_{k}"] = fpm2
    results[f"final_freq_diff_{k}"] = diff
    results[f"both_in_vocab_{k}"] = both_in_vocab
    if k in diff_percentile_dict:
        pctile1 = percentile(fpm1, *percentile_dict[k])
        pctile2 = percentile(fpm2, *percentile_dict[k])
        results[f"high_item_final_percentile_{k}"] = pctile1
        results[f"low_item_final_percentile_{k}"] = pctile2
        results[f"final_percentile_diff_{k}"] = abs(pctile1 - pctile2)
        results[f"final_diff_percentile_{k}"] = percentile(
            diff, *diff_percentile_dict[k]
        )  # this is the main measure one should use
    return results


def analyze_stimuli_pair_with_models(
    pair: StimulusPair,
    models: List[Model],
    percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    diff_percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    include_unigram: bool = True,
) -> Dict[str, Any]:
    high_item_tokenized = process_text(pair.high_item)
    low_item_tokenized = process_text(pair.low_item)
    results: Dict[str, Any] = {
        "high_item": pair.high_item,
        "high_item_tokenized": high_item_tokenized,
        "low_item": pair.low_item,
        "low_item_tokenized": low_item_tokenized,
    }
    if high_item_tokenized == "" or low_item_tokenized == "":
        Analyze.warn(
            f"Tokenization left no stimulus text: high={pair.high_item}, low={pair.low_item}"
        )
        return results
    if include_unigram:
        partial_results = models[0].final_ngram_diff(pair, n=1)
        results = add_stimuli_pair_results(
            results, partial_results, percentile_dict, diff_percentile_dict, 1
        )
    for model in models:
        partial_results = model.final_ngram_diff(pair, n=model._order)
        results = add_stimuli_pair_results(
            results,
            partial_results,
            percentile_dict,
            diff_percentile_dict,
            model._order,
        )
    return results


def goodness_rerank(input_file: Union[str, Path], orders: Iterable[int]) -> None:
    stimuli_pairs = pd.read_csv(input_file)
    orders = sorted(orders, reverse=True)
    Analyze.info(
        f"Maximizing difference for order {orders[0]} while minimizing for orders {orders[1:]}"
    )
    minimize_cols = [f"final_diff_percentile_{k}" for k in orders[1:]]
    max_to_minimize = stimuli_pairs[minimize_cols].max(axis=1)
    to_maximize = stimuli_pairs[f"final_diff_percentile_{orders[0]}"]
    goodness = to_maximize - max_to_minimize
    stimuli_pairs["goodness"] = goodness
    stimuli_pairs.sort_values("goodness", ascending=False, inplace=True)
    stimuli_pairs.to_csv(input_file, index=False)
    Analyze.info(f"Ordered pairs re-saved to {input_file}")


def analyze_stimuli_pair_data(
    pairs: Iterable[StimulusPair],
    models: List[Model],
    percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    diff_percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    csv_file: Union[Path, str] = "results.csv",
    disable_tqdm: bool = False,
):
    results = []
    for pair in tqdm(pairs, desc="Analyzing pairs", disable=disable_tqdm):
        results.append(
            analyze_stimuli_pair_with_models(
                pair,
                models,
                percentile_dict=percentile_dict,
                diff_percentile_dict=diff_percentile_dict,
                include_unigram=True,
            )
        )
    Path(csv_file).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(csv_file, index=False)
    orders = [1] + [m._order for m in models]
    goodness_rerank(csv_file, orders)


def load_models_and_percentiles(
    binary_files: Union[List[str], List[Path]],
    ngram_files: Optional[Union[List[str], List[Path]]] = None,
    max_n: Optional[int] = None,
    include_diff: bool = False,
    percentile_min_fpm: float = 0,
    disable_tqdm: bool = False,
) -> Tuple[
    List[Model],
    Dict[int, Tuple[np.ndarray, np.ndarray]],
    Dict[int, Tuple[np.ndarray, np.ndarray]],
]:
    models = [
        Model(file)
        for file in tqdm(binary_files, desc="Loading models", disable=disable_tqdm)
    ]
    models = [m for m in models if max_n is None or m._order <= max_n]
    percentile_dict = {}
    diff_percentile_dict = {}
    if ngram_files:
        percentile_dict = {
            int(Path(f).stem[-1]): get_percentiles(
                f,
                min_fpm=percentile_min_fpm,
                exclude_bos=True,
                exclude_eos=True,
                disable_tqdm=disable_tqdm,
            )
            for f in ngram_files
            if max_n is None or int(Path(f).stem[-1]) <= max_n
        }
        if include_diff:
            diff_percentile_dict = {
                int(Path(f).stem[-1]): get_diff_percentiles(
                    f,
                    min_fpm=percentile_min_fpm,
                    exclude_bos=True,
                    exclude_eos=True,
                    disable_tqdm=disable_tqdm,
                )
                for f in ngram_files
                if max_n is None or int(Path(f).stem[-1]) <= max_n
            }
    return models, percentile_dict, diff_percentile_dict


def analyze_single(
    input_file: Union[str, Path],
    csv_file: Union[str, Path],
    models: List[Model],
    percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    col: str,
    bos: bool = False,
    eos: bool = False,
    disable_tqdm: bool = False,
):
    df = pd.read_csv(input_file)[col]
    analyze_single_stimuli_data(
        stimuli=df.values,
        models=models,
        percentile_dict=percentile_dict,
        csv_file=csv_file,
        bos=bos,
        eos=eos,
        disable_tqdm=disable_tqdm,
    )


def analyze_paired(
    input_file: Union[str, Path],
    csv_file: Union[str, Path],
    models: List[Model],
    percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    diff_percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    cols: List[str],
    disable_tqdm: bool = False,
):
    df = pd.read_csv(input_file)[cols]
    pairs = [
        StimulusPair(high_item=row[cols[0]], low_item=row[cols[1]])
        for _, row in df.iterrows()
    ]
    analyze_stimuli_pair_data(
        pairs=pairs,
        models=models,
        percentile_dict=percentile_dict,
        diff_percentile_dict=diff_percentile_dict,
        csv_file=csv_file,
        disable_tqdm=disable_tqdm,
    )


def analyze_pairwise(
    input_file: Union[str, Path],
    csv_file: Union[str, Path],
    models: List[Model],
    percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    diff_percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    cols: List[str],
    disable_tqdm: bool = False,
):
    df = pd.read_csv(input_file)[cols]
    stimuli = []
    for col in cols:
        stimuli.extend(df[col].values)
    pairs = [StimulusPair(high_item=a, low_item=b) for a, b in combinations(stimuli, 2)]
    analyze_stimuli_pair_data(
        pairs=pairs,
        models=models,
        percentile_dict=percentile_dict,
        diff_percentile_dict=diff_percentile_dict,
        csv_file=csv_file,
        disable_tqdm=disable_tqdm,
    )


def analyze(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    model_files_folder: Union[str, Path],
    cols: List[str],
    max_n: Optional[int] = None,
    percentile_min_fpm: float = 0,
    single_analysis: bool = False,
    paired_analysis: bool = False,
    pairwise_analysis: bool = False,
    disable_tqdm: bool = False,
):
    assert (
        single_analysis or paired_analysis or pairwise_analysis
    ), "At least one type of analysis must be selected"
    assert (
        not paired_analysis or len(cols) == 2
    ), "Paired analysis requires exactly two columns"

    files = list(Path(input_folder).rglob("*.csv"))
    Analyze.info(f"Found {len(files)} CSV files to analyze in {input_folder}")

    binary_files = list(Path(model_files_folder).rglob("*.binary"))
    Analyze.info(
        f"Found {len(binary_files)} binary files to build models in {model_files_folder}"
    )

    ngram_files = list(Path(model_files_folder).rglob("*.ngram"))
    Analyze.info(
        f"Found {len(ngram_files)} ngram files to compute percentiles in {model_files_folder}"
    )

    models, percentile_dict, diff_percentile_dict = load_models_and_percentiles(
        binary_files,
        ngram_files,
        max_n=max_n,
        include_diff=(paired_analysis or pairwise_analysis),
        percentile_min_fpm=percentile_min_fpm,
        disable_tqdm=disable_tqdm,
    )

    if single_analysis:
        for col in cols:
            Analyze.info("Beginning single analysis on stimuli")
            for f in files:
                analyze_single(
                    input_file=f,
                    csv_file=Path(output_folder)
                    / (f.stem + f"_single_analysis_col={col}.csv"),
                    models=models,
                    percentile_dict=percentile_dict,
                    col=col,
                    disable_tqdm=disable_tqdm,
                )
    if paired_analysis:
        assert len(cols) == 2
        Analyze.info("Beginning paired analysis on paired stimuli")
        for f in files:
            analyze_paired(
                input_file=f,
                csv_file=Path(output_folder) / (f.stem + "_paired_analysis.csv"),
                models=models,
                percentile_dict=percentile_dict,
                diff_percentile_dict=diff_percentile_dict,
                cols=cols,
                disable_tqdm=disable_tqdm,
            )
    if pairwise_analysis:
        Analyze.info("Beginning pairwise analysis on stimuli")
        for f in files:
            analyze_pairwise(
                input_file=f,
                csv_file=Path(output_folder) / (f.stem + "_pairwise_analysis.csv"),
                models=models,
                percentile_dict=percentile_dict,
                diff_percentile_dict=diff_percentile_dict,
                cols=cols,
                disable_tqdm=disable_tqdm,
            )

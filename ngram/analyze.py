import typing
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from ngram.model import Model
from ngram.processing import process_text
from ngram.datatypes import StimulusPair
from ngram.abstract import Object


class Analyze(Object):
    pass


def get_percentiles(
    ngram_file: typing.Union[str, Path],
    num: int = 400,
    min_fpm: float = 0,
    exclude_bos: bool = True,
    exclude_eos: bool = True,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    freqs = []
    with open(ngram_file, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading percentiles"):
            freq, *tokens = line.split()
            if exclude_bos and tokens[0] == "<s>":
                continue
            if exclude_eos and tokens[-1] == "</s>":
                continue
            freqs.append(float(freq))
    np_freqs = np.array(freqs)
    np_freqs = np_freqs[np_freqs >= min_fpm]
    pctiles = np.linspace(0, 100, num=num)
    pctile_vals = np.percentile(np_freqs, pctiles)
    return pctile_vals, pctiles


def get_diff_percentiles(
    ngram_file: typing.Union[str, Path],
    num_bins: int = 10000,
    min_fpm: float = 0,
    exclude_bos: bool = True,
    exclude_eos: bool = True,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    freqs = []
    with open(ngram_file, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading percentiles"):
            freq, *tokens = line.split()
            if exclude_bos and tokens[0] == "<s>":
                continue
            if exclude_eos and tokens[-1] == "</s>":
                continue
            freqs.append(float(freq))
    np_freqs = np.array(freqs)
    np_freqs = np_freqs[np_freqs >= min_fpm]
    logfreqs = np.log10(np_freqs)  # lognormal data, so better bins
    # divide into equal sized bins
    bin_cnts, bin_edges = np.histogram(logfreqs, bins=num_bins)
    # use center for bin estimate
    bin_vals = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist = np.stack((bin_cnts, bin_vals), axis=1)
    # compute difference distribution based on binned estimates
    diff_cnts: typing.Dict[float, int] = defaultdict(int)
    for i in tqdm(range(len(hist)), desc="Computing diff percentiles"):
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
    return pctile_vals, pctiles


def percentile(
    freq_per_mil: float, pctile_vals: np.ndarray, pctiles: np.ndarray
) -> float:
    return float(np.interp(freq_per_mil, pctile_vals, pctiles).item())


def average_percentile(
    freq_per_mils: typing.List[float], pctile_vals: np.ndarray, pctiles: np.ndarray
) -> float:
    return float(
        np.mean([percentile(fpm, pctile_vals, pctiles) for fpm in freq_per_mils])
    )


def analyze_single_stimulus_with_unigram(
    stimulus: str, any_model: Model, bos: bool = False, eos: bool = False
) -> typing.Tuple[typing.List[float], float, typing.List[float], float, bool, bool]:
    scores = list(
        any_model.approximate_subgram_full_scores(stimulus, 1, bos=bos, eos=eos)
    )
    freq_per_mil_by_ngram = any_model.ngram_freqs(stimulus, n=1, bos=bos, eos=eos)
    any_oov = any(s[2] for s in scores)
    any_backed_off = False
    logprob = sum(s[0] for s in scores)
    freq_per_mil = 10**logprob * 1000000
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
) -> typing.Tuple[typing.List[float], float, typing.List[float], float, bool, bool]:
    scores = list(model.full_scores(stimulus, bos=bos, eos=eos))
    freq_per_mil_by_ngram = model.ngram_freqs(stimulus, bos=bos, eos=eos)
    any_oov = any(s[2] for s in scores)
    any_backed_off = any(s[1] < model._order for s in scores[model._order - 1 :])
    logprob = sum(s[0] for s in scores)
    freq_per_mil = 10**logprob * 1000000
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
    analyze_output: typing.Tuple[
        typing.List[float], float, typing.List[float], float, bool, bool
    ],
    percentile_dict: typing.Dict[int, typing.Tuple[np.ndarray, np.ndarray]],
    k: int,
) -> typing.Dict[str, typing.Any]:
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
    models: typing.List[Model],
    percentile_dict: typing.Dict[int, typing.Tuple[np.ndarray, np.ndarray]],
    include_unigram: bool = True,
    bos: bool = False,
    eos: bool = False,
) -> typing.Dict[str, typing.Any]:
    tmp = process_text(stimulus)
    tokenized_stimulus = (models[0].BOS + " ") * bos + tmp + (" " + models[0].EOS) * eos
    results: typing.Dict[str, typing.Any] = {
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
    stimuli: typing.Iterable[str],
    binary_files: typing.List[typing.Union[str, Path]],
    ngram_files: typing.Optional[typing.List[typing.Union[str, Path]]] = None,
    csv_file: typing.Union[Path, str] = "results.csv",
    bos: bool = False,
    eos: bool = False,
) -> pd.DataFrame:
    models = [Model(file) for file in tqdm(binary_files, desc="Loading models")]
    if ngram_files:
        percentile_dict = {
            int(Path(f).stem[-1]): get_percentiles(
                f, exclude_bos=not bos, exclude_eos=not eos
            )
            for f in ngram_files
        }
    else:
        percentile_dict = {}
    results = []
    for stimulus in tqdm(stimuli, desc="Analyzing stimuli"):
        results.append(
            analyze_single_stimulus_with_multiple_models(
                stimulus, models, percentile_dict=percentile_dict, bos=bos, eos=eos
            )
        )
    pd.DataFrame(results).to_csv(csv_file)


def add_stimuli_pair_results(
    results,
    analyze_output: typing.Tuple[float, bool, float, float],
    percentile_dict: typing.Dict[int, typing.Tuple[np.ndarray, np.ndarray]],
    diff_percentile_dict: typing.Dict[int, typing.Tuple[np.ndarray, np.ndarray]],
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
    models: typing.List[Model],
    percentile_dict: typing.Dict[int, typing.Tuple[np.ndarray, np.ndarray]],
    diff_percentile_dict: typing.Dict[int, typing.Tuple[np.ndarray, np.ndarray]],
    include_unigram: bool = True,
) -> typing.Dict[str, typing.Any]:
    high_item_tokenized = process_text(pair.high_item)
    low_item_tokenized = process_text(pair.low_item)
    results: typing.Dict[str, typing.Any] = {
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


def analyze_stimuli_pair_data(
    pairs: typing.Iterable[StimulusPair],
    binary_files: typing.List[typing.Union[str, Path]],
    ngram_files: typing.Optional[typing.List[typing.Union[str, Path]]] = None,
    csv_file: typing.Union[Path, str] = "results.csv",
):
    models = [Model(file) for file in tqdm(binary_files, desc="Loading models")]
    if ngram_files:
        percentile_dict = {
            int(Path(f).stem[-1]): get_percentiles(f) for f in ngram_files
        }
        diff_percentile_dict = {
            int(Path(f).stem[-1]): get_diff_percentiles(f) for f in ngram_files
        }
    else:
        percentile_dict = {}
    results = []
    for pair in tqdm(pairs, desc="Analyzing pairs"):
        results.append(
            analyze_stimuli_pair_with_models(
                pair,
                models,
                percentile_dict=percentile_dict,
                diff_percentile_dict=diff_percentile_dict,
            )
        )
    pd.DataFrame(results).to_csv(csv_file)

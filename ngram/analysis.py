from typing import List, Tuple, Dict, Any, Union, Iterable, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from ngram.model import Model
from ngram.processing import process_text
from ngram.datatypes import NGram
from ngram.abstract import Object


class Analyze(Object):
    pass


def percentile(val: float, pctile_vals: np.ndarray, pctiles: np.ndarray) -> float:
    return float(np.interp(val, pctile_vals, pctiles).item())


def analyze_sentence_with_order(
    model: Model,
    sentence: str,
    order: int,
    percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    ngram = NGram(text=sentence)
    subgrams = ngram.subgrams(order=order)
    counts = [model.get_count(s) for s in subgrams]
    fpms = [model.get_fpm(s) for s in subgrams]
    percentiles = [
        percentile(
            fpm,
            *percentile_dict[order],
        )
        for fpm in fpms
    ]
    analysis = {
        f"count_{order}": counts,
        f"fpm_{order}": fpms,
        f"percentile_{order}": percentiles,
        f"avg_percentile_{order}": np.mean(percentiles),
        f"any_oov_{order}": any(fpm == 0 for fpm in fpms),
    }
    return analysis


def analyze_sentence(
    model: Model,
    sentence: str,
    percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    omit_unk: bool,
):
    ngram = NGram(text=sentence)
    analysis: Dict[str, Any] = {
        "sentence": sentence,
        "tokenized_sentence": ngram.text(),
        "estimated_logprob": model.estimate_logprob(ngram, omit_unk=omit_unk),
        "logprob_per_token": model.logprobs_per_token(ngram),
    }
    for order in model.orders():
        analysis.update(
            analyze_sentence_with_order(model, sentence, order, percentile_dict)
        )
    return analysis


def analyze_sentence_data(
    model: Model,
    stimuli: Iterable[str],
    min_counts: Optional[List[int]] = None,
    chop_percent: float = 0.0,
    omit_unk: bool = False,
    disable_tqdm: bool = False,
) -> pd.DataFrame:
    percentile_dict = model.get_percentiles(
        orders=model.orders(), min_counts=min_counts, chop_percent=chop_percent
    )
    analysis = []
    for stimulus in tqdm(
        stimuli, desc="Analyzing sentences", unit="sentence", disable=disable_tqdm
    ):
        analysis.append(analyze_sentence(model, stimulus, percentile_dict, omit_unk))
    return pd.DataFrame(analysis)


def analyze_stimulus_pair_with_order(
    model: Model,
    stimulus_pair: Tuple[str, str],
    order: int,
    percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    percentile_of_difference_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, List[Dict[str, Any]]]:
    item_analyses = []
    for stimulus in stimulus_pair:
        ngram = NGram(text=stimulus, last_n=order)
        count = model.get_count(ngram)
        fpm = model.get_fpm(ngram)
        percentile1 = percentile(fpm, *percentile_dict[order])
        conditional_probability = model.conditional_probability(ngram, order=order)
        item_analyses.append(
            {
                f"count_{order}": count,
                f"fpm_{order}": fpm,
                f"percentile_{order}": percentile1,
                f"conditional_probability_{order}": conditional_probability,
            }
        )
    count1, count2 = (
        item_analyses[0][f"count_{order}"],
        item_analyses[1][f"count_{order}"],
    )
    percentile_of_difference = percentile(
        abs(count1 - count2), *percentile_of_difference_dict[order]
    )
    return percentile_of_difference, item_analyses


def analyze_stimulus_pair(
    model: Model,
    stimulus_pair: Tuple[str, str],
    percentile_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    percentile_of_difference_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {}
    item1_analysis: Dict[str, Any] = {}
    item2_analysis: Dict[str, Any] = {}

    orders = model.orders()
    max_order = max(orders)
    for order in orders:
        pdiff, (item1, item2) = analyze_stimulus_pair_with_order(
            model, stimulus_pair, order, percentile_dict, percentile_of_difference_dict
        )
        analysis[f"percentile_of_difference_{order}"] = pdiff
        analysis[f"ngram_oov_{order}"] = (
            item1[f"fpm_{order}"] == 0 or item2[f"fpm_{order}"] == 0
        )
        item1_analysis.update(item1)
        item2_analysis.update(item2)

    if item1_analysis[f"count_{max_order}"] < item2_analysis[f"count_{max_order}"]:
        item1_analysis, item2_analysis = item2_analysis, item1_analysis
        stimulus_pair = stimulus_pair[::-1]

    analysis.update(
        {
            "high_item": stimulus_pair[0],
            "tokenized_high_item": process_text(stimulus_pair[0]),
            "low_item": stimulus_pair[1],
            "tokenized_low_item": process_text(stimulus_pair[1]),
        }
    )
    analysis.update({f"high_item_{k}": v for k, v in item1_analysis.items()})
    analysis.update({f"low_item_{k}": v for k, v in item2_analysis.items()})
    return analysis


def analyze_stimulus_pair_data(
    model: Model,
    stimulus_pairs: Iterable[Tuple[str, str]],
    min_counts: Optional[List[int]] = None,
    chop_percent: float = 0.0,
    disable_tqdm: bool = False,
) -> pd.DataFrame:
    percentile_dict, percentile_of_difference_dict = model.get_all_percentiles(
        orders=model.orders(), min_counts=min_counts, chop_percent=chop_percent
    )
    analysis = []
    for stimulus_pair in tqdm(
        stimulus_pairs, desc="Analyzing pairs", unit="pair", disable=disable_tqdm
    ):
        analysis.append(
            analyze_stimulus_pair(
                model,
                stimulus_pair,
                percentile_dict,
                percentile_of_difference_dict,
            )
        )
    return pd.DataFrame(analysis)


def add_goodness_cols(analyzed_df: pd.DataFrame) -> pd.DataFrame:
    orders = [
        int(c.split("_")[-1])
        for c in analyzed_df.columns
        if c.startswith("high_item_count_")
    ]
    orders.sort()

    pctiles = np.linspace(0, 100, num=10000)
    for k in orders:
        analyzed_df[f"d_{k}"] = np.abs(
            analyzed_df[f"high_item_fpm_{k}"] - analyzed_df[f"low_item_fpm_{k}"]
        )
        analyzed_df[f"r_{k}"] = np.abs(
            np.log(analyzed_df[f"high_item_fpm_{k}"])
            - np.log(analyzed_df[f"low_item_fpm_{k}"])
        )
        pctile_vals = np.percentile(analyzed_df[f"d_{k}"], pctiles)
        analyzed_df[f"p_{k}"] = analyzed_df[f"d_{k}"].apply(
            lambda v: np.interp(
                v, pctile_vals, pctiles  # pylint: disable=cell-var-from-loop
            )
            / 100
        )

    analyzed_df["p_score_hard"] = analyzed_df[f"p_{orders[-1]}"] - analyzed_df[
        [f"p_{k}" for k in orders[:-1]]
    ].max(axis=1)
    analyzed_df["r_score_hard"] = analyzed_df[f"r_{orders[-1]}"] - analyzed_df[
        [f"r_{k}" for k in orders[:-1]]
    ].max(axis=1)

    analyzed_df["goodness"] = (
        analyzed_df[f"p_{orders[-1]}"]
        * (1 - analyzed_df[[f"p_{k}" for k in orders[:-1]]]).prod(axis=1)
        * analyzed_df["p_score_hard"]
        * analyzed_df[f"d_{orders[-1]}"]
        * analyzed_df["r_score_hard"]
    )

    if orders == [1, 2, 3, 4]:
        analyzed_df["meets_thresholds"] = (
            (analyzed_df["r_score_hard"] > 0)
            & (analyzed_df["p_score_hard"] > 0)
            & (analyzed_df["d_1"] <= 256)
            & (analyzed_df["d_2"] <= 64)
            & (analyzed_df["d_3"] <= 16)
            & (analyzed_df["d_4"] >= 4)
        )
    return analyzed_df


def analyze(
    model_file: Union[str, Path],
    input_file: Union[str, Path],
    cols: List[str],
    output_file: Union[str, Path],
    min_counts_for_percentile: Optional[List[int]] = None,
    chop_percent: float = 0.0,
    omit_unk: bool = False,
    disable_tqdm: bool = False,
    actual_model: Optional[Model] = None,
) -> None:
    if actual_model is not None:
        model = actual_model
    else:
        model = Model(model_file, read_only=True)
        model.load_into_memory()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    if len(cols) == 1:
        Analyze.info("Analyzing sentences")
        sentences = pd.read_csv(input_file)[cols[0]]
        analyze_sentence_data(
            model,
            sentences,
            min_counts_for_percentile,
            chop_percent,
            omit_unk,
            disable_tqdm,
        ).to_csv(output_file)
    elif len(cols) == 2:
        Analyze.info("Analyzing stimulus pairs")
        stimulus_pairs = pd.read_csv(input_file)[cols].itertuples(index=False)
        add_goodness_cols(
            analyze_stimulus_pair_data(
                model,
                stimulus_pairs,
                min_counts_for_percentile,
                chop_percent,
                disable_tqdm,
            )
        ).to_csv(output_file)
    else:
        Analyze.error("cols must be a list of one or two column names")
        raise ValueError("cols must be a list of one or two column names")
    Analyze.info(f"Analysis saved to {output_file}")

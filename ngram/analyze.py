import typing
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from ngram.model import Model
from ngram.datatypes import NGram, StimulusPair
from ngram.processing import segment_text


def analyze_single_stimuli_with_unigram(
    stimulus: str, any_model: Model, bos: bool = False, eos: bool = False
) -> typing.Tuple[typing.List[float], float, float, bool, bool]:
    scores = list(
        any_model.approximate_subgram_full_scores(stimulus, 1, bos=bos, eos=eos)
    )
    any_oov = any(s[2] for s in scores)
    any_backed_off = False
    logprob = sum(s[0] for s in scores)
    freq_per_mil = 10**logprob * 1000000
    lobprob_by_token = [s[0] for s in scores]
    return lobprob_by_token, logprob, freq_per_mil, any_oov, any_backed_off


def analyze_single_stimulus_with_model(
    stimulus: str, model: Model, bos: bool = False, eos: bool = False
) -> typing.Tuple[typing.List[float], float, float, bool, bool]:
    scores = list(model.full_scores(stimulus, bos=bos, eos=eos))
    any_oov = any(s[2] for s in scores)
    any_backed_off = any(s[1] < model._order for s in scores[model._order - 1 :])
    logprob = sum(s[0] for s in scores)
    freq_per_mil = 10**logprob * 1000000
    lobprob_by_token = [s[0] for s in scores]
    return lobprob_by_token, logprob, freq_per_mil, any_oov, any_backed_off


def analyze_single_stimulus_with_multiple_models(
    stimulus: str,
    models: typing.List[Model],
    include_unigram: bool = True,
    bos: bool = False,
    eos: bool = False,
) -> typing.Dict[str, typing.Any]:
    tokenized_stimulus = models[0]._tokenizer.process_text_for_kenlm(stimulus)
    if bos:
        tokenized_stimulus = models[0].BOS + " " + tokenized_stimulus
    if eos:
        tokenized_stimulus += " " + models[0].EOS

    results: typing.Dict[str, typing.Any] = {
        "stimulus": stimulus,
        "tokenized_stimulus": tokenized_stimulus,
    }
    if include_unigram:
        (
            lobprob_by_token,
            logprob,
            freq_per_mil,
            any_oov,
            any_backed_off,
        ) = analyze_single_stimuli_with_unigram(stimulus, models[0], bos=bos, eos=eos)
        results["logprob_by_token_1"] = lobprob_by_token
        results["logprob_1"] = logprob
        results["freq_per_mil_1"] = freq_per_mil
        results["any_backed_off_1"] = any_backed_off
        results["any_oov_1"] = any_oov
    for model in models:
        (
            lobprob_by_token,
            logprob,
            freq_per_mil,
            any_oov,
            any_backed_off,
        ) = analyze_single_stimulus_with_model(stimulus, model, bos=bos, eos=eos)
        k = model._order
        results[f"logprob_by_token_{k}"] = lobprob_by_token
        results[f"logprob_{k}"] = logprob
        results[f"freq_per_mil_{k}"] = freq_per_mil
        results[f"any_backed_off_{k}"] = any_backed_off
        results[f"any_oov_{k}"] = any_oov
    return results


def analyze_single_stimuli_data(
    stimuli: typing.Iterable,
    model_files: typing.List[typing.Union[str, Path]],
    csv_file: typing.Union[Path, str] = "results.csv",
    bos: bool = False,
    eos: bool = False,
) -> pd.DataFrame:
    models = [Model(file) for file in model_files]
    results = []
    for stimulus in tqdm(stimuli, desc="Analyzing stimuli"):
        results.append(
            analyze_single_stimulus_with_multiple_models(
                stimulus, models, bos=bos, eos=eos
            )
        )
    pd.DataFrame(results).to_csv(csv_file)


def analyze_stimuli_pair_with_models(
    pair: StimulusPair,
    models: typing.List[Model],
    include_unigram: bool = True,
) -> typing.Dict[str, typing.Any]:
    tokenized_s1 = models[0]._tokenizer.process_text_for_kenlm(pair.s1)
    tokenized_s2 = models[0]._tokenizer.process_text_for_kenlm(pair.s2)
    results: typing.Dict[str, typing.Any] = {
        "s1": pair.s1,
        "tokenized_s1": tokenized_s1,
        "s2": pair.s2,
        "tokenized_s2": tokenized_s2,
    }
    if include_unigram:
        diff, both_in_vocab = models[0].final_ngram_diff(pair, n=1)
        results[f"final_diff_1"] = diff
        results[f"both_in_vocab_1"] = both_in_vocab
    for model in models:
        diff, both_in_vocab = model.final_ngram_diff(pair, n=model._order)
        results[f"final_diff_{model._order}"] = diff
        results[f"both_in_vocab_{model._order}"] = both_in_vocab
    return results


def analyze_stimuli_pair_data(
    pairs: typing.Iterable[StimulusPair],
    model_files: typing.List[typing.Union[str, Path]],
    csv_file: typing.Union[Path, str] = "results.csv",
):
    models = [Model(file) for file in model_files]
    results = []
    for pair in tqdm(pairs, desc="Analyzing pairs"):
        results.append(analyze_stimuli_pair_with_models(pair, models))
    pd.DataFrame(results).to_csv(csv_file)

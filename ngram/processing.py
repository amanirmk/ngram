import typing
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from ngram.model import Model
from ngram.datatypes import NGram
from ngram.tokenizer import Tokenizer


def create_model_files(
    input_folder: str,
    output_folder: str,
    n,
    file_prefix: str = "all_corpora",
    all_up_to=True,
):
    output_file = preprocess_text(
        input_folder, output_folder + "/" + file_prefix + ".txt"
    )
    create_arpa_and_binary(output_file, output_folder, n, all_up_to)
    if all_up_to:
        for k in range(2, n + 1):
            create_ngram_list(
                output_folder + "/" + str(Path(output_file).stem) + f"_{k}"
            )
    else:
        create_ngram_list(output_folder + "/" + str(Path(output_file).stem) + f"_{n}")


def preprocess_text(
    input_folder: str,
    output_file: str,
    lower: bool = True,
    remove_punc: bool = True,
    remove_meta: bool = True,
    split_contractions: bool = True,
    substitute_contractions: bool = False,
    make_newlines: bool = True,
) -> str:
    files = []
    for dirpath, _, filenames in os.walk(input_folder):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))

    tokenizer = Tokenizer(
        lower=lower,
        remove_punc=remove_punc,
        remove_meta=remove_meta,
        split_contractions=split_contractions,
        substitute_contractions=substitute_contractions,
    )
    for file in tqdm(files, desc="Preprocessing files"):
        tokenizer.tokenize_file(
            file, output_file, output_mode="a", make_newlines=make_newlines
        )
    return output_file


def create_arpa_and_binary(
    text_file: typing.Union[str, Path],
    output_folder: typing.Union[str, Path],
    n: int,
    all_up_to: bool = False,
) -> None:
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = str(Path(text_file).stem)
    if all_up_to:
        for k in range(2, n + 1):
            arpa_path = folder / Path(stem + f"_{k}.arpa")
            binary_path = folder / Path(stem + f"_{k}.binary")
            create_arpa(text_file, arpa_path, k)
            create_binary(arpa_path, binary_path)
    else:
        arpa_path = folder / Path(stem + f"_{n}.arpa")
        binary_path = folder / Path(stem + f"_{n}.binary")
        create_arpa(text_file, arpa_path, n)
        create_binary(arpa_path, binary_path)


def create_arpa(
    text_file: typing.Union[str, Path], arpa_path: typing.Union[str, Path], n
) -> None:
    flag = os.system(f"./kenlm/build/bin/lmplz -o {n} <{text_file} >{arpa_path}")
    if flag != 0:
        raise ValueError("Error in creating ARPA file")


def create_binary(
    arpa_path: typing.Union[str, Path], binary_path: typing.Union[str, Path]
) -> None:
    flag = os.system(f"./kenlm/build/bin/build_binary {arpa_path} {binary_path}")
    if flag != 0:
        raise ValueError("Error in creating binary file")


def create_ngram_list(file_path_and_stem: typing.Union[str, Path]) -> None:
    arpa_path = Path(file_path_and_stem).with_suffix(".arpa")
    binary_path = Path(file_path_and_stem).with_suffix(".binary")
    ngram_path = Path(file_path_and_stem).with_suffix(".ngram")

    t = tqdm(desc="Reading ngrams")
    with open(arpa_path, "rt", encoding="utf-8") as f:
        f.readline()
        line = f.readline()
        # find order of the model
        while "ngram" in line:
            n = int(line.split()[1][0])
            line = f.readline()

        # find start of the ngrams
        while True:
            if f"{n}-grams" in f.readline():
                break

        # read the ngrams
        ngrams = []
        while True:
            split_line = f.readline().split()
            if not split_line:
                break
            _, *tokens = split_line
            ngrams.append(NGram(tokens=tokens))
            t.update()
    t.close()

    m = Model(binary_path)
    scored_ngrams = [
        (ngram, m.freq_per_mil(ngram)[0])
        for ngram in tqdm(ngrams, desc="Getting ngram frequencies")
    ]
    print("Sorting ngrams")
    scored_ngrams.sort(key=lambda x: -x[1])

    with open(ngram_path, "wt", encoding="utf-8") as f:
        for ngram, fpm in tqdm(scored_ngrams, desc="Writing ngrams"):
            f.write(f"{fpm} {ngram.text()}\n")


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
    include_unigram=True,
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


def analyze_stimuli_pair_with_model():
    return NotImplemented

from typing import Optional, Union, List
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from ngram.model import Model
from ngram.datatypes import NGram
from ngram.abstract import Object
from ngram.analysis import analyze


class Extend(Object):
    pass


def extend_stimulus_data(
    model_file: Union[str, Path],
    input_file: Union[str, Path],
    sentence_column: str,
    output_file: Union[str, Path],
    length: int,
    min_prob: float = 0.0,
    max_prob: float = 1.0,
    extend_mode: str = "maximize",
    sampling_seed: Optional[int] = None,
    disable_tqdm: bool = False,
    do_analysis: bool = True,
    min_counts_for_percentile: Optional[List[int]] = None,
    chop_percent: float = 0.0,
):
    model = Model(model_file, read_only=True)
    model.load_into_memory()
    df = pd.read_csv(input_file)[[sentence_column]]

    def extend_sentence(sentence: str) -> str:
        ngram = NGram(text=sentence)
        ngram = model.extend(
            ngram=ngram,
            tokens_to_add=length - len(ngram),
            allow_eos=False,
            order=None,
            flexible_order=True,
            min_prob=min_prob,
            max_prob=max_prob,
            mode=extend_mode,
            seed=sampling_seed,
        )
        return ngram.text()

    if not disable_tqdm:
        tqdm.pandas()
        df["extended_sentence"] = df[sentence_column].progress_apply(extend_sentence)
    else:
        df["extended_sentence"] = df[sentence_column].apply(extend_sentence)

    df.to_csv(output_file, index=False)

    if do_analysis:
        analyze(
            model_file=model_file,
            input_file=output_file,
            cols=["extended_sentence"],
            output_file=output_file,  # pylint: disable=duplicate-code
            min_counts_for_percentile=min_counts_for_percentile,
            chop_percent=chop_percent,
            disable_tqdm=disable_tqdm,
            actual_model=model,  # so won't load twice
        )

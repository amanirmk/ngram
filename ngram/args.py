import dataclasses
from typing import List, Optional, Union
from pathlib import Path


@dataclasses.dataclass
class Arguments:
    #   -----------------------------------
    #       some general arguments
    #   -----------------------------------

    # which pipeline to use
    action: str = dataclasses.field(default="preprocess")

    # whether to disable tqdm
    disable_tqdm: bool = dataclasses.field(default=False)

    # folder locations
    corpora: str = dataclasses.field(
        default="./data/corpora",
    )
    processed_corpora: str = dataclasses.field(
        default="./data/corpora_processed",
    )
    stimuli: str = dataclasses.field(
        default="./data/stimuli",
    )
    stimuli_analyzed: str = dataclasses.field(
        default="./data/stimuli_analyzed",
    )
    model_files: str = dataclasses.field(
        default="./data/model_files",
    )

    #   -----------------------------------
    #       args for action = preprocess
    #   -----------------------------------

    # input folder
    # corpora: used as set in folder locations

    # output folder
    # processed_corpora: used as set in folder locations

    # combine all files into one (or keep folder structure)
    combine_files_as: Optional[str] = dataclasses.field(default=None)

    #   -----------------------------------
    #       args for action = train
    #   -----------------------------------

    # name of the model
    model_name: str = dataclasses.field(default="model_4_attrs")

    # file or folder containing preprocessed corpora
    # (none assumes args.processed_corpora)
    read_from: Optional[Union[str, Path]] = dataclasses.field(default=None)

    # which orders to compute statistics for
    orders: List[int] = dataclasses.field(default_factory=lambda: list(range(1, 5)))

    # whether to <s> and </s> in the model
    include_sentence_boundaries: bool = dataclasses.field(default=False)

    # whether/how to prune the model (like KenLM, based on absolute counts and
    # uses the last value for higher orders)
    min_counts: Optional[List[int]] = dataclasses.field(default=None)

    #   -----------------------------------
    #       args for action = analyze
    #   -----------------------------------

    # file containing the model
    model_file: Union[str, Path] = dataclasses.field(default="./data/model_files/model")

    # file containing the stimuli
    stimuli_file: Union[str, Path] = dataclasses.field(
        default="./data/stimuli/arnon/high_bin.csv"
    )

    # columns to analyze
    columns_for_analysis: List[str] = dataclasses.field(
        default_factory=lambda: ["high_item", "low_item"]
    )

    # output file
    # (none will keep file name but place in args.stimuli_analyzed)
    analyzed_file: Optional[Union[str, Path]] = dataclasses.field(default=None)

    # min counts for percentile calculations
    min_counts_for_percentile: Optional[List[int]] = dataclasses.field(default=None)

    #   -----------------------------------
    #       args for action = construct
    #   -----------------------------------

    # file containing the model
    # model_file: used as set in args for action=analyze

    # output file
    constructed_pairs_csv: Union[str, Path] = dataclasses.field(
        default="./data/stimuli_constructed/constructed_pairs.csv"
    )

    # length of the sentences to construct
    length: int = dataclasses.field(default=4)

    # number of candidate stimuli pairs to construct
    n_candidates: int = dataclasses.field(default=5_000_000)

    # number of sentences with same prefix to pair up
    # results in (n choose 2) pairs of similar form
    max_per_prefix: int = dataclasses.field(default=10)

    # min counts for percentile calculations
    # min_counts_for_percentile: used as set in args for action=analyze

    # seed for sampling ngrams for constructing pairs
    sampling_seed: Optional[int] = dataclasses.field(default=None)

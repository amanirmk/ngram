import dataclasses
import typing


@dataclasses.dataclass
class Arguments:
    # which pipeline to use
    action: typing.Optional[str] = dataclasses.field(default="process")

    # folder locations
    original_corpora: typing.Optional[str] = dataclasses.field(
        default="./data/corpora",
    )
    processed_corpora: typing.Optional[str] = dataclasses.field(
        default="./data/corpora_processed",
    )
    model_files: typing.Optional[str] = dataclasses.field(
        default="./data/model_files",
    )
    kenlm_bin_path: typing.Optional[str] = dataclasses.field(
        default="./kenlm/build/bin",
    )
    kenlm_tmp_path: typing.Optional[str] = dataclasses.field(default="./tmp")
    stimuli: typing.Optional[str] = dataclasses.field(
        default="./data/stimuli",
    )
    stimuli_analyzed: typing.Optional[str] = dataclasses.field(
        default="./data/stimuli_analyzed",
    )

    # general arguments (for all actions)
    disable_tqdm: typing.Optional[bool] = dataclasses.field(default=False)
    max_n: typing.Optional[int] = dataclasses.field(default=4)
    # note: processing max_n >= 7 requires kenlm rebuild
    # note: processing max_n == 1 will build a bigram model as proxy

    # arguments for action=process
    processed_filestem: typing.Optional[str] = dataclasses.field(default="all_corpora")
    all_up_to: typing.Optional[bool] = dataclasses.field(default=True)
    prune: typing.Optional[bool] = dataclasses.field(default=True)
    kenlm_ram_limit_mb: typing.Optional[int] = dataclasses.field(default=4_096)

    # arguments for action=analyze
    do_single_analysis: typing.Optional[bool] = dataclasses.field(default=True)
    do_paired_analysis: typing.Optional[bool] = dataclasses.field(default=True)
    do_pairwise_analysis: typing.Optional[bool] = dataclasses.field(default=False)
    columns_for_analysis: typing.Optional[typing.List[str]] = dataclasses.field(
        default_factory=lambda: ["high_item", "low_item"],
    )
    percentile_min_fpm: typing.Optional[float] = dataclasses.field(default=0)

    # arguments for action=construct
    ngram_file: typing.Optional[str] = dataclasses.field(
        default="./data/model_files/ngram/all_corpora_4.ngram",
    )
    prefix_file: typing.Optional[str] = dataclasses.field(
        default="./data/model_files/ngram/all_corpora_3.ngram",
    )
    constructed_pairs_csv: typing.Optional[str] = dataclasses.field(
        default="./data/stimuli_constructed/constructed_pairs.csv",
    )
    n_candidates: typing.Optional[int] = dataclasses.field(default=10_000)
    top_bottom_k: typing.Optional[int] = dataclasses.field(default=20)
    sampling_seed: typing.Optional[int] = dataclasses.field(default=42)
    # note: construct also uses percentile_min_fpm (as it analyzes constructed pairs)

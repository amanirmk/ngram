import dataclasses
import typing


@dataclasses.dataclass
class Arguments:
    # which pipeline to use
    action: typing.Optional[str] = dataclasses.field(default="analyze")

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

    # general arguments
    disable_tqdm: typing.Optional[bool] = dataclasses.field(default=False)

    # arguments for action=process
    processed_filestem: typing.Optional[str] = dataclasses.field(default="all_corpora")
    max_n: typing.Optional[int] = dataclasses.field(default=4)
    # note: max_n >= 7 requires kenlm rebuild
    # note: max_n == 1 will build a bigram model as proxy
    all_up_to: typing.Optional[bool] = dataclasses.field(default=True)
    prune: typing.Optional[bool] = dataclasses.field(default=True)
    kenlm_ram_limit_mb: typing.Optional[int] = dataclasses.field(default=4096)

    # arguments for action=analyze
    do_single_analysis: typing.Optional[bool] = dataclasses.field(default=True)
    do_paired_analysis: typing.Optional[bool] = dataclasses.field(default=True)
    do_pairwise_analysis: typing.Optional[bool] = dataclasses.field(default=False)
    columns_for_analysis: typing.Optional[typing.List[str]] = dataclasses.field(
        default_factory=lambda: ["high_item", "low_item"],
    )
    percentile_min_fpm: typing.Optional[float] = dataclasses.field(default=0)

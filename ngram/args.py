import dataclasses


@dataclasses.dataclass
class Arguments:
    # which pipeline to use
    action: str = dataclasses.field(default="process")

    # folder locations
    original_corpora: str = dataclasses.field(
        default="./data/corpora",
    )
    processed_corpora: str = dataclasses.field(
        default="./data/corpora_processed",
    )
    model_files: str = dataclasses.field(
        default="./data/model_files",
    )
    kenlm_bin_path: str = dataclasses.field(
        default="./kenlm/build/bin",
    )
    kenlm_tmp_path: str = dataclasses.field(default="./tmp")
    stimuli: str = dataclasses.field(
        default="./data/stimuli",
    )
    stimuli_analyzed: str = dataclasses.field(
        default="./data/stimuli_analyzed",
    )

    # general arguments (for all actions)
    disable_tqdm: bool = dataclasses.field(default=False)
    max_n: int = dataclasses.field(default=4)
    # note: processing max_n >= 7 requires kenlm rebuild
    # note: processing max_n == 1 will build a bigram model as proxy

    # arguments for action=process
    processed_filestem: str = dataclasses.field(default="all_corpora")
    all_up_to: bool = dataclasses.field(default=True)
    prune: bool = dataclasses.field(default=True)
    kenlm_ram_limit_mb: int = dataclasses.field(default=4_096)

    # arguments for action=analyze
    do_single_analysis: bool = dataclasses.field(default=True)
    do_paired_analysis: bool = dataclasses.field(default=True)
    do_pairwise_analysis: bool = dataclasses.field(default=False)
    columns_for_analysis: List[str] = dataclasses.field(
        default_factory=lambda: ["high_item", "low_item"],
    )
    percentile_min_fpm: float = dataclasses.field(default=0)

    # arguments for action=construct
    ngram_file: str = dataclasses.field(
        default="./data/model_files/ngram/all_corpora_4.ngram",
    )
    prefix_file: str = dataclasses.field(
        default="./data/model_files/ngram/all_corpora_3.ngram",
    )
    constructed_pairs_csv: str = dataclasses.field(
        default="./data/stimuli_constructed/constructed_pairs.csv",
    )
    n_candidates: int = dataclasses.field(default=10_000)
    top_bottom_k: int = dataclasses.field(default=20)
    sampling_seed: int = dataclasses.field(default=42)
    # note: construct also uses percentile_min_fpm (as it analyzes constructed pairs)

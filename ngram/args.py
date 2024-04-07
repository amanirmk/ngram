import dataclasses
import typing


@dataclasses.dataclass
class Arguments:
    action: typing.Optional[str] = dataclasses.field(default="process")

    # arguments for action=process
    original_corpora: typing.Optional[str] = dataclasses.field(
        default="./data/corpora",
    )
    processed_corpora: typing.Optional[str] = dataclasses.field(
        default="./data/corpora_processed",
    )
    processed_filestem: typing.Optional[str] = dataclasses.field(default="all_corpora")
    model_files: typing.Optional[str] = dataclasses.field(
        default="./data/model_files",
    )
    kenlm_bin_path: typing.Optional[str] = dataclasses.field(
        default="./kenlm/build/bin",
    )
    kenlm_tmp_path: typing.Optional[str] = dataclasses.field(default="./tmp")
    kenlm_ram_limit_mb: typing.Optional[int] = dataclasses.field(default=4096)
    max_n: typing.Optional[int] = dataclasses.field(default=4)
    all_up_to: typing.Optional[bool] = dataclasses.field(default=True)
    prune: typing.Optional[bool] = dataclasses.field(default=True)
    disable_tqdm: typing.Optional[bool] = dataclasses.field(default=False)

    # arguments for action=analyze
    stimuli: typing.Optional[str] = dataclasses.field(
        default="./data/stimuli",
    )

import dataclasses
import typing


@dataclasses.dataclass
class Arguments:
    original_corpora: typing.Optional[str] = dataclasses.field(
        default="./corpora_original",
    )
    processed_corpora: typing.Optional[str] = dataclasses.field(
        default="./corpora_processed",
    )
    max_n: typing.Optional[int] = dataclasses.field(default=4)
    all_up_to: typing.Optional[bool] = dataclasses.field(default=True)

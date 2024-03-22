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

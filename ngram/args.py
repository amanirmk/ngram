import dataclasses
import typing


@dataclasses.dataclass
class Arguments:
    model_path: typing.Optional[str] = dataclasses.field(
        default="./coca.binary",
    )

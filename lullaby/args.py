import dataclasses
import typing


@dataclasses.dataclass
class Arguments:
    example_arg: typing.Optional[str] = dataclasses.field(
        default="example value",
    )

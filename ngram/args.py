import dataclasses
from argparse import Namespace
from typing import List, Optional, Union
from pathlib import Path
from ngram.abstract import Object


class UNSET:
    def __eq__(self, other):
        return isinstance(other, UNSET)


@dataclasses.dataclass
class Arguments:
    action: str = dataclasses.field()
    disable_tqdm: bool = dataclasses.field(default=True)
    input_folder: Union[str, Path, UNSET] = dataclasses.field(default_factory=UNSET)
    output_folder: Union[str, Path, UNSET] = dataclasses.field(default_factory=UNSET)
    output_file: Union[str, Path, UNSET] = dataclasses.field(default_factory=UNSET)
    combine_files_as: Optional[str] = dataclasses.field(default=None)
    model_file: Union[str, Path, UNSET] = dataclasses.field(default_factory=UNSET)
    orders: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 3, 4])
    include_sentence_boundaries: bool = dataclasses.field(default=False)
    min_counts: Optional[Union[List[int]]] = dataclasses.field(default=None)
    stimuli_file: Union[str, Path, UNSET] = dataclasses.field(default_factory=UNSET)
    columns_for_analysis: List[str] = dataclasses.field(
        default_factory=lambda: ["high_item", "low_item"]
    )
    min_candidate_fpm: float = dataclasses.field(default=0.0)
    chop_percent: float = dataclasses.field(default=0.0)
    n_candidates: Optional[int] = dataclasses.field(default=None)
    length: int = dataclasses.field(default=4)
    max_per_prefix: Optional[int] = dataclasses.field(default=None)


def validate_args(args: Namespace) -> None:
    class Args(Object):
        pass

    changed_args = set()
    for field in dataclasses.fields(Arguments):
        if field.default_factory is not dataclasses.MISSING:
            default = field.default_factory()
        else:
            default = field.default
        if field.name != "action" and getattr(args, field.name) != default:
            changed_args.add(field.name)

    if args.action == "preprocess":
        required = {"input_folder", "output_folder"}
        optional = {"combine_files_as", "disable_tqdm"}
    elif args.action == "train":
        required = {"model_file", "read_from"}
        optional = {
            "min_counts",
            "include_sentence_boundaries",
            "orders",
            "disable_tqdm",
        }
    elif args.action == "analyze":
        required = {"model_file", "stimuli_file", "output_file"}
        optional = {
            "columns_for_analysis",
            "min_counts",
            "chop_percent",
            "disable_tqdm",
        }
    elif args.action == "construct":
        required = {"model_file", "stimuli_file", "output_file"}
        optional = {
            "length",
            "n_candidates",
            "max_per_prefix",
            "min_counts",
            "min_candidate_fpm",
            "chop_percent",
            "disable_tqdm",
        }
    else:
        Args.error(f"Invalid action: {args.action}")
        raise ValueError(f"Invalid action: {args.action}")

    missing_required = required - changed_args
    if missing_required:
        Args.error(f"Missing required arguments: {missing_required}")
        raise ValueError(f"Missing required arguments: {missing_required}")

    excess = changed_args - required - optional
    if excess:
        Args.warn(f"Unused arguments ignored: {excess}")

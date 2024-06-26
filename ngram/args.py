import dataclasses
from argparse import Namespace
from typing import List, Optional, Union
from ngram.abstract import Object


@dataclasses.dataclass
class Arguments:
    action: str = dataclasses.field()
    disable_tqdm: bool = dataclasses.field(default=False)
    input_folder: str = dataclasses.field(default="<UNSET>")
    output_folder: str = dataclasses.field(default="<UNSET>")
    read_from: str = dataclasses.field(default="<UNSET>")
    input_file: str = dataclasses.field(default="<UNSET>")
    output_file: str = dataclasses.field(default="<UNSET>")
    combine_files_as: Optional[str] = dataclasses.field(default=None)
    model_file: str = dataclasses.field(default="<UNSET>")
    orders: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 3, 4])
    include_sentence_boundaries: bool = dataclasses.field(default=False)
    min_counts: Optional[Union[List[int]]] = dataclasses.field(default=None)
    columns_for_analysis: List[str] = dataclasses.field(
        default_factory=lambda: ["high_item", "low_item"]
    )
    min_candidate_fpm: float = dataclasses.field(default=0.0)
    chop_percent: float = dataclasses.field(default=0.0)
    n_candidates: Optional[int] = dataclasses.field(default=None)
    length: int = dataclasses.field(default=4)
    max_per_prefix: Optional[int] = dataclasses.field(default=None)
    do_analysis: bool = dataclasses.field(default=False)
    min_prob: float = dataclasses.field(default=0.0)
    max_prob: float = dataclasses.field(default=1.0)
    extend_mode: str = dataclasses.field(default="maximize")
    sampling_seed: Optional[int] = dataclasses.field(default=None)
    sentence_column: str = dataclasses.field(default="<UNSET>")


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
        required = {"model_file", "input_file", "output_file"}
        optional = {
            "columns_for_analysis",
            "min_counts",
            "chop_percent",
            "disable_tqdm",
        }
    elif args.action == "construct":
        required = {"model_file", "output_file"}
        optional = {
            "length",
            "n_candidates",
            "max_per_prefix",
            "min_counts",
            "min_candidate_fpm",
            "chop_percent",
            "disable_tqdm",
            "do_analysis",
        }
    elif args.action == "extend":
        required = {"model_file", "input_file", "output_file", "sentence_column"}
        optional = {
            "length",
            "min_prob",
            "max_prob",
            "extend_mode",
            "sampling_seed",
            "disable_tqdm",
            "do_analysis",
            "min_counts",
            "chop_percent",
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
        Args.warn(
            f"You have changed the value for arguments that are not used: {excess}"
        )

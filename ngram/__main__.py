from pathlib import Path
from transformers import HfArgumentParser

from ngram.abstract import Object
from ngram.args import Arguments
from ngram.processing import preprocess_files
from ngram.model import Model
from ngram.analysis import analyze
from ngram.construction import construct


def main() -> None:
    class Main(Object):
        pass

    args = HfArgumentParser(Arguments).parse_args()
    if args.action == "preprocess":
        Main.info(
            f"Beginning to process corpora in {args.corpora}. "
            + f"Output will be saved in {args.corpora_processed}"
        )
        preprocess_files(
            input_folder=args.corpora,
            output_folder=args.corpora_processed,
            combine_files_as=args.combine_files_as,
            disable_tqdm=args.disable_tqdm,
        )
    elif args.action == "train":
        Main.info(
            f"Beginning to train model on {args.corpora_processed}. "
            + f"Output will be saved in {args.model_files}."
        )
        model = Model(Path(args.model_files) / args.model_name)
        if args.read_from is None:
            args.read_from = args.corpora_processed
        model.read_from(
            location=args.read_from,
            orders=args.orders,
            include_sentence_boundaries=args.include_sentence_boundaries,
            disable_tqdm=args.disable_tqdm,
        )
        if args.min_counts is not None:
            model.prune(min_counts=args.min_counts)
        model.save()
    elif args.action == "analyze":
        if args.stimuli_file is None:
            Main.info("Must provide stimuli file for analysis.")
            raise ValueError("Must provide stimuli file for analysis.")
        if args.model_file is None:
            Main.info("Must provide model file for analysis.")
            raise ValueError("Must provide model file for analysis.")
        Main.info(f"Beginning to analyze stimuli in {args.stimuli_file}.")
        if args.analyzed_file is None:
            args.analyzed_file = (
                Path(args.stimuli_analyzed) / Path(args.stimuli_file).name
            )
        analyze(
            model_file=args.model_file,
            input_file=args.stimuli_file,
            cols=args.columns_for_analysis,
            output_file=args.analyzed_file,
            min_counts_for_percentile=args.min_counts_for_percentile,
            chop_percent=args.chop_percent,
            disable_tqdm=args.disable_tqdm,
        )
    elif args.action == "construct":
        if args.model_file is None:
            Main.info("Must provide model file for construction.")
            raise ValueError("Must provide model file for construction.")
        Main.info(
            f"Beginning to construct stimuli pairs with model={args.model_file}. "
            + f"Output will be saved in {args.constructed_pairs_csv}."
        )
        construct(
            model_file=args.model_file,
            output_file=args.constructed_pairs_csv,
            length=args.length,
            n_candidates=args.n_candidates,
            max_per_prefix=args.max_per_prefix,
            min_counts_for_percentile=args.min_counts_for_percentile,
            min_candidate_fpm=args.min_candidate_fpm,
            chop_percent=args.chop_percent,
            disable_tqdm=args.disable_tqdm,
        )
    else:
        raise ValueError(f"Invalid action: {args.action}")


if __name__ == "__main__":
    main()

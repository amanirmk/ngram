from transformers import HfArgumentParser

from ngram.abstract import Object
from ngram.args import Arguments, validate_args
from ngram.processing import preprocess_files
from ngram.model import Model
from ngram.analysis import analyze
from ngram.construction import construct
from ngram.extension import extend_stimulus_data


def main() -> None:
    class Main(Object):
        pass

    args = HfArgumentParser(Arguments).parse_args()
    validate_args(args)

    if args.action == "preprocess":
        Main.info(
            f"Beginning to process corpora in {args.input_folder}. "
            + f"Output will be saved in {args.output_folder}"
        )
        preprocess_files(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            combine_files_as=args.combine_files_as,
            disable_tqdm=args.disable_tqdm,
        )
    if args.action == "train":
        Main.info(
            f"Beginning to train model on {args.read_from}. "
            + f"Output will be saved in {args.model_file}."
        )
        model = Model(args.model_file)
        model.read_from(
            location=args.read_from,
            orders=args.orders,
            include_sentence_boundaries=args.include_sentence_boundaries,
            disable_tqdm=args.disable_tqdm,
        )
        if args.min_counts is not None:
            model.prune(min_counts=args.min_counts)
        model.save()
    if args.action == "analyze":
        Main.info(
            f"Beginning to analyze stimuli from {args.input_file}. "
            + f"Analysis will be done with model {args.model_file}. "
            + f"Output will be saved in {args.output_file}."
        )
        analyze(
            model_file=args.model_file,
            input_file=args.input_file,
            cols=args.columns_for_analysis,
            output_file=args.output_file,
            min_counts_for_percentile=args.min_counts,
            chop_percent=args.chop_percent,
            disable_tqdm=args.disable_tqdm,
        )
    if args.action == "construct":
        Main.info(
            f"Beginning to construct stimuli pairs with model {args.model_file}. "
            + f"Output will be saved in {args.output_file}."
        )
        construct(
            model_file=args.model_file,
            output_file=args.output_file,
            length=args.length,
            n_candidates=args.n_candidates,
            max_per_prefix=args.max_per_prefix,
            min_counts_for_percentile=args.min_counts,
            min_candidate_fpm=args.min_candidate_fpm,
            chop_percent=args.chop_percent,
            disable_tqdm=args.disable_tqdm,
        )
    if args.action == "extend":
        Main.info(
            f"Beginning to extend stimuli from {args.input_file} "
            + f"to length {args.length}. "
            + f"Output will be saved in {args.output_file}."
        )
        extend_stimulus_data(
            model_file=args.model_file,
            input_file=args.input_file,
            sentence_column=args.sentence_column,
            output_file=args.output_file,
            length=args.length,
            min_prob=args.min_prob,
            max_prob=args.max_prob,
            extend_mode=args.extend_mode,
            sampling_seed=args.sampling_seed,
            disable_tqdm=args.disable_tqdm,
            do_analysis=args.do_analysis,
            min_counts_for_percentile=args.min_counts,
            chop_percent=args.chop_percent,
        )
    Main.info("All complete!")


if __name__ == "__main__":
    main()

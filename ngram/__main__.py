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
            f"Beginning to process corpora in {args.original_corpora}. "
            + f"Output will be saved in {args.processed_corpora}"
        )
        preprocess_files(
            input_folder=args.corpora,
            output_folder=args.processed_corpora,
            combine_files_as=args.combine_files_as,
            disable_tqdm=args.disable_tqdm,
        )
    elif args.action == "train":
        Main.info(
            f"Beginning to train model on {args.processed_corpora}. "
            + f"Output will be saved in {args.model_files}."
        )
        model = Model(Path(args.model_files) / args.model_name)
        model.read_from_folder(
            input_folder=args.processed_corpora,
            orders=args.orders,
            include_sentence_boundaries=args.include_sentence_boundaries,
        )
        if args.prune:
            model.prune(min_counts=args.min_counts)
        del model

    elif args.action == "analyze":
        Main.info(
            f"Beginning to analyze stimuli in {args.stimuli}. "
            + f"Output will be saved in {args.stimuli_analyzed}."
        )
        analyze(
            input_folder=args.stimuli,
            output_folder=args.stimuli_analyzed,
            model_files_folder=args.model_files,
            cols=args.columns_for_analysis,
            max_n=args.max_n,
            percentile_min_fpm=args.percentile_min_fpm,
            single_analysis=args.do_single_analysis,
            paired_analysis=args.do_paired_analysis,
            pairwise_analysis=args.do_pairwise_analysis,
            disable_tqdm=args.disable_tqdm,
        )
    elif args.action == "construct":
        Main.info(
            f"Beginning to construct stimuli pairs from {args.ngram_file}. "
            + f"Output will be saved in {args.constructed_pairs_csv}."
        )
        construct(
            model_files_folder=args.model_files,
            ngram_file=args.ngram_file,
            prefix_file=args.prefix_file,
            output_file=args.constructed_pairs_csv,
            max_n=args.max_n,
            n_candidates=args.n_candidates,
            top_bottom_k=args.top_bottom_k,
            min_fpm=args.percentile_min_fpm,
            disable_tqdm=args.disable_tqdm,
            seed=args.sampling_seed,
        )
    else:
        raise ValueError(f"Invalid action: {args.action}")


if __name__ == "__main__":
    main()

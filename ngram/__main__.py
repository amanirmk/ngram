from transformers import HfArgumentParser

from ngram.abstract import Object
from ngram.args import Arguments
from ngram.processing import process
from ngram.analysis import analyze
from ngram.construction import construct


def main() -> None:
    class Main(Object):
        pass

    args = HfArgumentParser(Arguments).parse_args()
    if args.action == "process":
        Main.info(
            f"Beginning to process corpora in {args.original_corpora}. Output will be saved in {args.processed_corpora} and {args.model_files}."
        )
        Main.info(
            f"Building model files {'up to' if args.all_up_to else 'for'} n={args.max_n}."
        )
        process(
            input_folder=args.original_corpora,
            processed_corpora_folder=args.processed_corpora,
            model_output_folder=args.model_files,
            n=args.max_n,
            kenlm_bin_path=args.kenlm_bin_path,
            kenlm_tmp_path=args.kenlm_tmp_path,
            kenlm_ram_limit_mb=args.kenlm_ram_limit_mb,
            proxy_n_for_unigram=(2 if args.max_n == 1 else None),
            filestem=args.processed_filestem,
            all_up_to=args.all_up_to,
            prune=args.prune,
        )
    elif args.action == "analyze":
        Main.info(
            f"Beginning to analyze stimuli in {args.stimuli}. Output will be saved in {args.stimuli_analyzed}."
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
            f"Beginning to construct stimuli pairs from {args.ngram_file}. Output will be saved in {args.constructed_pairs_csv}."
        )
        construct(
            model_files_folder=args.model_files,
            ngram_file=args.ngram_file,
            prefix_file=args.prefix_file,
            output_file=args.constructed_pairs_csv,
            max_n=args.max_n,
            n_candidates=args.n_candidates,
            min_fpm=args.percentile_min_fpm,
            disable_tqdm=args.disable_tqdm,
        )
    else:
        raise ValueError(f"Invalid action: {args.action}")


if __name__ == "__main__":
    main()

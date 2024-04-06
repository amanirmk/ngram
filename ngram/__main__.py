from transformers import HfArgumentParser

from ngram.abstract import Object
from ngram.args import Arguments
from ngram.processing import create_model_files


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
        create_model_files(
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
    else:
        raise ValueError(f"Invalid action: {args.action}")


if __name__ == "__main__":
    main()

from transformers import HfArgumentParser

from ngram.abstract import Object
from ngram.args import Arguments
from ngram.processing import create_model_files


def main() -> None:
    class Main(Object):
        pass

    args = HfArgumentParser(Arguments).parse_args()
    if args.action == "process":
        Main.info(f"Beginning to process corpora in {args.original_corpora}. Output will be saved in {args.processed_corpora} and {args.model_files}.")
        Main.info(f"Building model files {"up to" if args.all_up_to else "for"} n={args.max_n}.")
        create_model_files(
            args.original_corpora,
            args.processed_corpora,
            args.model_files,
            args.max_n,
            args.processed_filestem,
            args.all_up_to,
        )
    else:
        raise ValueError(f"Invalid action: {args.action}")


if __name__ == "__main__":
    main()

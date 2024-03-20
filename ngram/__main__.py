from transformers import HfArgumentParser

from ngram.abstract import Object
from ngram.args import Arguments


def main() -> None:
    class Main(Object):
        pass

    args = HfArgumentParser(Arguments).parse_args()
    # do things with args
    Main.info(f"Ran module with args: {args.__dict__}")


if __name__ == "__main__":
    main()

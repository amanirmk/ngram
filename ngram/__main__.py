from transformers import HfArgumentParser

from ngram.abstract import Object
from ngram.args import Arguments

from ngram.processing import arpa_to_ngrams


def main() -> None:
    class Main(Object):
        pass

    _ = HfArgumentParser(Arguments).parse_args()
    arpa_to_ngrams("./corpora_processed/coca.arpa")


if __name__ == "__main__":
    main()

from transformers import HfArgumentParser

from lullaby.abstract import Object
from lullaby.args import Arguments


def main() -> None:
    class Main(Object):
        pass

    args = HfArgumentParser(Arguments).parse_args()
    # do things with args
    Main.info(f"Ran module with args: {args.__dict__}")


if __name__ == "__main__":
    main()

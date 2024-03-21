from transformers import HfArgumentParser

from ngram.abstract import Object
from ngram.args import Arguments

from ngram.datatypes import Model, StimulusPair


def main() -> None:
    class Main(Object):
        pass

    args = HfArgumentParser(Arguments).parse_args()
    
    model = Model(args.model_path)
    pair = StimulusPair(s1="click on ads", s2="click on it")
    match, final = model.evaluate_pair(pair)
    Main.info(f"Match: {match}, Final: {final}")


if __name__ == "__main__":
    main()

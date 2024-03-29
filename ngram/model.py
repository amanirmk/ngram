import typing
from pathlib import Path
import kenlm
from ngram.abstract import Object
from ngram.datatypes import StimulusPair, NGram
from ngram.processing import process_text


class Model(Object):
    BOS = "<s>"
    EOS = "</s>"

    def __init__(self, path: typing.Union[str, Path]) -> None:
        super().__init__()
        self._path = path
        self._model = kenlm.Model(str(self._path))  # type: ignore[attr-defined]
        self._order = self._model.order

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._path})"

    def freq_per_mil(self, ngram: NGram) -> typing.Tuple[float, bool]:
        scores = list(self.full_scores(ngram.text()))
        fpm = 10 ** sum(s[0] for s in scores) * 1000000
        _, n_used, _ = scores[-1]
        oov = sum(s[2] for s in scores) > 0
        in_vocab = n_used == len(ngram) and not oov
        return fpm, in_vocab

    def ngram_diffs(self, ngram1: NGram, ngram2: NGram) -> typing.Tuple[float, bool]:
        fpm1, in_vocab1 = self.freq_per_mil(ngram1)
        fpm2, in_vocab2 = self.freq_per_mil(ngram2)
        both_in_vocab = in_vocab1 and in_vocab2
        return abs(fpm1 - fpm2), both_in_vocab

    def final_ngram_diff(self, pair: StimulusPair, n: int) -> typing.Tuple[float, bool]:
        ng1 = NGram(text=pair.highItem, last_n=n)
        ng2 = NGram(text=pair.lowItem, last_n=n)
        return self.ngram_diffs(ng1, ng2)

    def approximate_final_diffs_by_n(self, pair: StimulusPair) -> typing.List[float]:
        return [self.final_ngram_diff(pair, n)[0] for n in range(1, self._order + 1)]

    def score(self, text: str, bos: bool = False, eos: bool = False) -> float:
        return self._model.score(process_text(text), bos=bos, eos=eos)

    def full_scores(
        self, text: str, bos: bool = False, eos: bool = False
    ) -> typing.Iterable[typing.Tuple[float, int, bool]]:
        return self._model.full_scores(process_text(text), bos=bos, eos=eos)

    def approximate_subgram_full_scores(
        self, text: str, n: int, bos: bool = False, eos: bool = False
    ) -> typing.Iterable[typing.Tuple[float, int, bool]]:
        tokens = [self.BOS] * bos + text.split() + [self.EOS] * eos
        ngrams = [NGram(tokens=tokens[:i], last_n=n) for i in range(1, len(tokens) + 1)]
        scores = [list(self.full_scores(ngram.text()))[-1] for ngram in ngrams][
            int(bos) :
        ]
        return scores

    def approximate_subgram_score(
        self, text: str, n: int, bos: bool = False, eos: bool = False
    ) -> float:
        return sum(
            score[0]
            for score in self.approximate_subgram_full_scores(text, n, bos=bos, eos=eos)
        )

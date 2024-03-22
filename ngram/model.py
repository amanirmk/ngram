import typing
from pathlib import Path
import kenlm
from ngram.abstract import Object
from ngram.datatypes import StimulusPair, NGram
from ngram.tokenizer import Tokenizer


class Model(Object):
    BOS = "<s>"
    EOS = "</s>"

    def __init__(self, path: typing.Union[str, Path]) -> None:
        super().__init__()
        self._path = path
        self._model = kenlm.Model(str(self._path))  # type: ignore[attr-defined]
        self._order = self._model.order
        self._tokenizer = Tokenizer()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._path})"

    def freq_per_mil(self, ngram: NGram) -> typing.Tuple[float, bool]:
        scores = list(self.full_scores(ngram.text()))
        fpm = 10 ** sum(s[0] for s in scores) * 1000000
        _, n_used, oov = scores[-1]
        in_vocab = n_used == len(ngram) and not oov
        return fpm, in_vocab

    def ngram_diffs(self, ngram1: NGram, ngram2: NGram) -> float:
        fpm1, in_vocab1 = self.freq_per_mil(ngram1)
        fpm2, in_vocab2 = self.freq_per_mil(ngram2)
        assert in_vocab1 and in_vocab2, "One or both ngrams are not in the vocabulary"
        return abs(fpm1 - fpm2)

    def final_ngram_diff(self, pair: StimulusPair, n: int) -> float:
        ng1 = NGram(text=pair.s1, last_n=n)
        ng2 = NGram(text=pair.s2, last_n=n)
        return self.ngram_diffs(ng1, ng2)

    def final_diffs_by_n(self, pair: StimulusPair) -> typing.List[float]:
        return [self.final_ngram_diff(pair, n) for n in range(1, self._order + 1)]

    def validate_pair(
        self, pair: StimulusPair, max_difference: float, min_difference: float
    ) -> bool:
        diffs = self.final_diffs_by_n(pair)
        valid = max(diffs[:-1]) <= max_difference and diffs[-1] >= min_difference
        return valid

    def score(self, text: str, bos: bool = False, eos: bool = False) -> float:
        return self._model.score(self._tokenizer.process_text_for_kenlm(text), bos=bos, eos=eos)

    def full_scores(
        self, text: str, bos: bool = False, eos: bool = False
    ) -> typing.Iterable[typing.Tuple[float, int, bool]]:
        return self._model.full_scores(self._tokenizer.process_text_for_kenlm(text), bos=bos, eos=eos)

    # I was hoping these would be exact, but it appears the lower order information is unrecoverable (except for unigram)
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

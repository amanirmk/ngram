import abc
import typing

from pydantic import BaseModel
import kenlm

from ngram.abstract import Object, classproperty

class StimulusPair(BaseModel):
    s1: str
    s2: str

class Model(Object):
    def __init__(self, path) -> None:
        super().__init__()
        self._path: str = path
        self._model: kenlm.Model = kenlm.Model(self._path)
        self._order = self._model.order

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._path})"

    def validate_pair(self, pair: StimulusPair, max_difference: float, min_difference: float) -> bool:
        match, final = self.evaluate_pair(pair)
        return match <= max_difference and final >= min_difference

    def evaluate_pair(self, pair: StimulusPair) -> typing.Tuple[float, float]:
        diffs = []
        try:
            for n in range(1, self._order+1):
                fpm1 = self.get_final_fpm(pair.s1, n)
                fpm2 = self.get_final_fpm(pair.s2, n)
                d = abs(fpm1 - fpm2)
                diffs.append(d)
        except AssertionError as e:
            Model.error(f"Issue encountered while evaluating pair: {e}")
            raise e
        match = max(diffs[:-1])
        final = diffs[-1]
        return (match, final)
    
    def get_final_fpm(self, stimulus: str, n: int) -> float:
        """ THIS NEEDS FIXING - GET NGRAM LOG PROB, NO AVG(?)"""
        assert 0 < n <= self._order, f"Invalid order ({n}) for this model (order={self._order})"
        
        tokens = stimulus.split()
        assert len(tokens) >= n, f"Stimulus ({stimulus}) is too short for order {n}"

        final_ngram = " ".join(tokens[-n:])
        scores = list(self._model.full_scores(final_ngram, bos=True, eos=False))

        _, length, oov = scores[-1]
        assert length == n, f"Final ngram ({final_ngram}) is not in vocabulary"
        assert not oov, f"Final token ({final_ngram[-1]}) is not in vocabulary"

        ngram_logprob = sum(s[0] for s in scores)
        avg_ngram_fpm = (10 ** (ngram_logprob / n)) * 1000000
        return avg_ngram_fpm
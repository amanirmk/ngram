import typing
from pydantic import BaseModel
from ngram.abstract import Object


class StimulusPair(BaseModel):
    s1: str
    s2: str


class NGram(Object):
    def __init__(self, text: str, last_n=0):
        self._tokens = tuple(text.split()[-last_n:])
        self._n = len(self._tokens)
        self._text = " ".join(self._tokens)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._text})"

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> str:
        return self._tokens[i]

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self._tokens)

    def __contains__(self, token: str) -> bool:
        return token in self._tokens

    def tokens(self) -> typing.Tuple[str, ...]:
        return self._tokens

    def text(self) -> str:
        return self._text

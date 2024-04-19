from typing import Optional, Tuple, Iterator, Iterable
from pydantic import BaseModel
from ngram.abstract import Object
from ngram.processing import tokenize


class StimulusPair(BaseModel):
    high_item: str
    low_item: str


class NGram(Object):
    def __init__(
        self,
        text: Optional[str] = None,
        tokens: Optional[Iterable[str]] = None,
        last_n=0,
    ):
        assert (text is not None or tokens is not None) and not (
            text is not None and tokens is not None
        ), "Text or tokens must be provided, and not both"
        if text is not None:
            self._tokens = tuple(tokenize(text)[-last_n:])
        else:
            self._tokens = tuple(tokens[-last_n:])  # type: ignore[index]
        self._n = len(self._tokens)
        self._text = " ".join(self._tokens)
        if self._n == 0 or self._text == "":
            NGram.warn(f"No tokens found: input={text or tokens}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._text})"

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> str:
        return self._tokens[i]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tokens)

    def __contains__(self, token: str) -> bool:
        return token in self._tokens

    def tokens(self) -> Tuple[str, ...]:
        return self._tokens

    def text(self) -> str:
        return self._text

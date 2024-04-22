from typing import Optional, Tuple, Iterator, List, Iterable
from ngram.abstract import Object
from ngram.processing import tokenize


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

    def subgrams(self, order: int, partial: bool = False) -> List["NGram"]:
        return [
            NGram(tokens=self[:i], last_n=order)
            for i in range(1 if partial else order, len(self) + 1)
        ]

    def to_query(self, order: Optional[int] = None) -> str:
        o = str(len(self)) if order is None else str(order)
        return o + "/" + "/".join(self)

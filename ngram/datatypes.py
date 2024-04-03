import typing
from pydantic import BaseModel
from ngram.abstract import Object


class StimulusPair(BaseModel):
    high_item: str
    low_item: str


class NGram(Object):
    def __init__(
        self,
        text: typing.Optional[str] = None,
        tokens: typing.Optional[typing.Iterable[str]] = None,
        last_n=0,
    ):
        # avoiding circular import
        from ngram.processing import tokenize

        assert (text or tokens) and not (
            text and tokens
        ), "Text or tokens must be provided, and not both"
        if text:
            self._tokens = tuple(tokenize(text)[-last_n:])
        else:
            self._tokens = tuple(tokens[-last_n:])  # type: ignore[index]
        self._n = len(self._tokens)
        self._text = " ".join(self._tokens)
        if self._n == 0 or self._text == "":
            raise ValueError(
                f"No tokens found: input={text or tokens}, output={self._tokens}"
            )

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

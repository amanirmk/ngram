import abc
import logging
import pathlib
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class classproperty(Generic[T]):  # pylint: disable=invalid-name
    def __init__(self, func: Callable[[Any], T]) -> None:
        self.func = func

    def __get__(self, obj: Any, cls: Any) -> T:
        return self.func(cls)


class Object(abc.ABC):
    basedir: pathlib.Path = pathlib.Path(__file__).parents[1]

    @classproperty
    def _name(cls) -> str:  # pylint: disable=no-self-argument
        return cls.__name__

    @classproperty
    def _logger(cls) -> logging.Logger:  # pylint: disable=no-self-argument
        return logging.getLogger(cls._name)

    @classmethod
    def _log(cls, message: str, level: str) -> None:
        assert hasattr(cls._logger, level)
        if "\n" in message:
            lines = message.split("\n")
        else:
            lines = [message]
        for line in lines:
            getattr(cls._logger, level)(line)

    @classmethod
    def info(cls, message: str) -> None:
        cls._log(message, "info")

    @classmethod
    def warn(cls, message: str) -> None:
        cls._log(message, "warning")

    @classmethod
    def error(cls, message: str) -> None:
        cls._log(message, "error")

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

import abc
import typing

from pydantic import BaseModel

from lullaby.abstract import Object, classproperty


class SimpleType(BaseModel):
    attr1: float
    attr2: str


class ComplexType(Object):
    def __init__(self) -> None:
        super().__init__()

    @classproperty
    def class_method(cls) -> None:  # pylint: disable=no-self-argument
        pass

    @abc.abstractmethod
    def abstract_method(self) -> None:
        raise NotImplementedError()

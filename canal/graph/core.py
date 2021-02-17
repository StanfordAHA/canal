import abc
import enum
from typing import List, Tuple


class CoreConnectionType(enum.Flag):
    CB = 1 << 0
    SB = 1 << 1
    Core = 1 << 2
    Default = CB | SB


class InterconnectCore(abc.ABC):
    @abstractmethod
    def inputs(self) -> List[Tuple[int, str]]:
        pass

    @abstractmethod
    def outputs(self) -> List[Tuple[int, str]]:
        pass

    @abstractmethod
    def get_port_ref(self, port_name: str):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

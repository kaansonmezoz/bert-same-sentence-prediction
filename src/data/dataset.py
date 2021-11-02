from abc import ABC, abstractmethod


class Dataset(ABC):

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def split_into_sentences(self):
        pass

    @abstractmethod
    def filter_sentences(self):
        pass

    @abstractmethod
    def save(self):
        pass

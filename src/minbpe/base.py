from abc import abstractmethod, ABC
import typing


TokenList = typing.List[int]
'''
A list of tokens, where each token is represented as an integer.
'''

TokenPair = typing.Tuple[int, int]
'''
A pair of tokens, where each token is represented as an integer.`
'''

PairCounts = typing.Dict[typing.Tuple[int, int], int]
'''
The pair counts dictionary maps pairs of tokens to the number of times they
occur consecutively.
'''

Merges = typing.OrderedDict[typing.Tuple[int, int], int]
'''
The merges that were applied during the byte pair encoding algorithm, in order.
Each merge replaces a pair of tokens with a new token.
'''

VocabDict = typing.Dict[int, bytes]
'''
Mapping of each of the replacement tokens to the bytes that it replaced.
'''


class Tokenizer(ABC):

    @abstractmethod
    def train(self, text: str, vocab_size: int):
        pass

    @abstractmethod
    def encode(self, text: str) -> TokenList:
        pass

    @abstractmethod
    def decode(self, encoded_tokens: TokenList)-> str:
        pass

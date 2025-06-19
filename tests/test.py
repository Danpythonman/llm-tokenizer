import abc
import unittest
from pathlib import Path
import typing

from minbpe.base import Tokenizer
from minbpe.basic import BasicTokenizer
from minbpe.regex import RegexTokenizer


class TokenizerTest(unittest.TestCase, abc.ABC):

    skip: bool = True

    taylor_swift_wikipedia_text: typing.Optional[str]
    llama_text: typing.Optional[str]

    abc.abstractmethod
    def make_instance(self) -> Tokenizer:
        pass

    def setUp(self):
        data_path = Path(__file__).parent / 'data'

        taylor_swift_wikipedia_path = data_path / 'taylorswift.txt'

        with open(taylor_swift_wikipedia_path, 'r') as f:
            self.taylor_swift_wikipedia_text = f.read()

        llama_path = data_path / 'llamatext.txt'

        with open(llama_path, 'r') as f:
            self.llama_text = f.read()

    def test_encode_decode(self):
        if self.skip:
            self.skipTest('Base class does not need to be tested')

        training_text = self.llama_text
        if training_text is None:
            self.fail('Training data not present')

        text = self.taylor_swift_wikipedia_text
        if text is None:
            self.fail('Testing data not present')

        tokenizer = self.make_instance()

        tokenizer.train(training_text, 400)

        tokens = tokenizer.encode(text)

        decoded = tokenizer.decode(tokens)

        self.assertEqual(text, decoded)


class BasicTokenizerTest(TokenizerTest):

    skip = False

    def make_instance(self) -> Tokenizer:
        return BasicTokenizer()


class RegexTokenizerTest(TokenizerTest):

    skip = False

    def make_instance(self) -> Tokenizer:
        return RegexTokenizer()


if __name__ == '__main__':
    unittest.main()

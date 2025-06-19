import collections
import typing

import regex as re

from .base import Tokenizer, TokenList, TokenPair, PairCounts, VocabDict, Merges


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    _vocab: typing.Optional[VocabDict]
    _merges: typing.Optional[Merges]

    def __init__(self):
        super().__init__()
        self._vocab = None
        self._merges = None
        self._compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)

    def train(self, text: str, vocab_size: int):
        '''
        Trains the tokenizer by splitting the text with the GPT-4 regex and
        using the byte pair encoding algorithm.
        '''

        text_chunks = re.findall(self._compiled_pattern, text)

        id_chunks = [list(str(chunk).encode('utf-8')) for chunk in text_chunks]

        tokens = [[int(id) for id in chunk] for chunk in id_chunks]

        number_of_merges = vocab_size - 256

        vocab, merges = self._byte_pair_encode(
            tokens,
            vocab_size,
            number_of_merges
        )

        self._vocab = vocab
        self._merges = merges

    def encode(self, text: str) -> TokenList:
        if self._vocab is None or self._merges is None:
            raise Exception('Tokenizer has not been trained, cannot encode')

        text_chunks = re.findall(self._compiled_pattern, text)
        text_chunks = [str(chunk) for chunk in text_chunks]

        tokens = []

        for chunk in text_chunks:
            chunk_bytes = chunk.encode('utf-8')
            token_chunk = list(map(int, chunk_bytes))
            for pair, replacement_token in self._merges.items():
                token_chunk = self._merge(token_chunk, pair, replacement_token)

            tokens.extend(token_chunk)

        return tokens

    def decode(self, encoded_tokens: TokenList)-> str:
        if self._vocab is None or self._merges is None:
            raise Exception('Tokenizer has not been trained, cannot decode')

        encoded_token_bytes = [self._vocab[token] for token in encoded_tokens]

        tokens = b''.join(encoded_token_bytes)

        text = tokens.decode('utf-8', errors='replace')

        return text

    def _count_pairs(
        self,
        tokens: TokenList,
        pair_counts: PairCounts
    ) -> PairCounts:
        '''
        Counts the number of times each pair of tokens occurs consecutively in
        `tokens`, updating the counts in `pair_counts`.

        Note that if a pair of tokens does not occur consecutively in tokens,
        then it will not be present in the output dictionary.

        Args:
            tokens (List[int]): The list of tokens to be processed.

            pair_counts (Dict[Tuple[int, int], int]): The dictionary of pair
                counts to be updated.

        Returns:
            PairCounts: The updated dictionary mapping every consecutive pair of
            tokens that occurs consecutively in `tokens` to the number of times
            it does so.
        '''

        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        return pair_counts

    def _merge(
        self,
        tokens: TokenList,
        pair_to_replace: TokenPair,
        replacement_token: int
    ) -> TokenList:
        '''
        Merges a list of tokens with a conditional replacement token. Every
        occurrence of `pair_to_replace` in `tokens` will be replaced by
        `replacement_token`.

        Note that the replacement does not happen in place. A new list of tokens
        is created and returned.

        Args:
            tokens (List[int]): The list of tokens to be merged.

            pair_to_replace (Tuple[int, int]): The pair of tokens to be replaced
                by the replacement token wherever it occurs in `tokens`.

            replacement_token (int): The token with which to replace occurrences
                of the replacement pair in `tokens`.

        Returns:
            List[int]: The new list of tokens where every occurrence of
            `pair_to_replace` in `tokens` has been replaced with
            `replacement_token`.
        '''

        new_tokens = []
        i = 0
        can_replace: bool

        while i < len(tokens):
            if i < len(tokens) - 1:
                can_replace = (
                    pair_to_replace[0] == tokens[i]
                    and pair_to_replace[1] == tokens[i+1]
                )
            else:
                can_replace = False

            if can_replace:
                new_tokens.append(replacement_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def _byte_pair_encode(
        self,
        token_chunks: typing.List[TokenList],
        original_vocab_size: int,
        number_of_merges: int
    ) -> typing.Tuple[VocabDict, Merges]:
        '''
        Applies the byte pair encoding algorithm to a list of chunks of tokens
        by merging the most frequently occurring pair in the tokens with a new
        token `number_of_merges` times.

        Args:
            token_chunks (List[List[int]]): The list of chunks of tokens on
                which to apply the byte pair encoding algorithm.

            original_vocab_size (int): The original number of unique tokens in
                the vocabulary.

            number_of_merges (int): The number of iterations to run the byte
                pair encoding algorithm. More merges means more pairs of tokens
                will be replaced by a new token.

        Returns:
            Tuple[VocabDict, Merges]: A tuple containing:
            - `VocabDict`: a dictionary mapping each replacement token to the
              bytes it replaced. Note that the bytes may be more than a pair
              (like if a replacement token is a replacement of a replacement).
            - `Merges`: a dictionary mapping pairs of tokens to the new
              replacement token that replaced the pair, in the order they were
              applied in.
        '''

        merges = collections.OrderedDict()
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(number_of_merges):
            pair_counts = dict()
            for chunk in token_chunks:
                # Update the counts of the consecutive pairs of tokens
                pair_counts = self._count_pairs(chunk, pair_counts)

            # Get the most commonly-occurring consecutive pair of tokens
            top_pair = max(pair_counts, key=pair_counts.get)

            # Make a new token
            new_index = original_vocab_size + i

            # Replace the top pair with the new token in each token chunk
            token_chunks = [
                self._merge(chunk, top_pair, new_index)
                for chunk in token_chunks
            ]

            # Map the top pair to the new token created for it
            merges[top_pair] = new_index

            # Map the new token to the pair of tokens that it replaced
            vocab[new_index] = vocab[top_pair[0]] + vocab[top_pair[1]]

        return vocab, merges

import typing


PairCounts = typing.Dict[typing.Tuple[int, int], int]
'''
The pair counts dictionary maps pairs of tokens to the number of times they
occur consecutively.
'''

class BasicTokenizer:

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        pass

    def encode(self, text: str) -> typing.List[int]:
        pass

    def decode(self, ids: typing.List[int]) -> str:
        pass

    def _count_pairs(self, tokens: typing.List[int]) -> PairCounts:
        '''
        Counts the number of times each pair of tokens occurs consecutively in
        `tokens`.

        Note that if a pair of tokens does not occur consecutively in tokens,
        then it will not be present in the output dictionary.

        Args:
            tokens (List[int]): The list of tokens to be processed.

        Returns:
            PairCounts: A dictionary mapping every consecutive pair of tokens
            that occurs consecutively in `tokens` to the number of times it does
            so.
        '''

        counts: PairCounts = dict()

        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1

        return counts

    def _merge(
        self,
        tokens: typing.List[int],
        pair_to_replace: typing.Tuple[int, int],
        replacement_token: int
    ) -> typing.List[int]:
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
                    pair_to_replace[i] == tokens[i]
                    and pair_to_replace[i+1] == tokens[i+1]
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

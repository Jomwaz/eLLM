import re


# Simple text tokenizer

# Step 1 -- Tokenizing Text
# Because we are building a 'word' embedder, we need to tokenize
# the words of our input dataset. This means that the first step
# in preparing the dataset is to split the words and punctions
# and hold them within a structure - for our purposes, it will be
# a list.

# We refrain from making all text upper/lower- case because
# capitalization helps the LLM distinguish between proper
# nouns and common nouns, understand sentence structure,
# and learn to generate text with proper capitalization.

# Step 2 -- Converting Tokens into Token IDs
# To map the previously generated tokens into token IDs, we have
# to build a vocabulary first. This vocabulary defines how we map
# each unique word and special character to a unique integer.

# Take your training set, seperate by words/punctiations, sort by
# alphabetic order. Assigned an integer value to each word and throw
# this into a key/value structure - dictionary in python.

# Step 3 -- Adding special context tokens
# We add special tokens to a vocabulary to deal with certain context.
# For instance, we add an <|unk|> token to represent new and
# unknown words that were not part of the training data and thus not
# part of the existing vocabulary. Furthermore, we add the
# <|endoftext|> token that we can use to separate two unrelated
# text sources.

# There are other special tokens that researchers may use depending
# on the model they are building, but the tokenizer used for GPT
# models do not need any of these tokens, it only uses an <|endoftext|>
# token for simplicity. The tokenizer used for GPT models also doesn't
# use an <|unk|> token for out-of-vocabulary words. Instead, the model
# uses byte pair encoding.


_RE_ENCODE = r'([,.:;?_!"()\']|--|\s)'
_RE_DECODE = r'\s+([,.?!"()\'])'

_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|unk|>",
]


class Text:
    def __init__(self, text: str = ""):
        self._text = text

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def set_text(self, text: str) -> None:
        self._text = text

    def extend_text(self, text: str) -> None:
        self._text = " <|endoftext|> ".join((self._text, text))


class Vocab:
    def __init__(self, text: str):
        self._re_encode = _RE_ENCODE
        self._text = text
        self._vocab = self._vocab_f_tokens()

    @property
    def vocab(self) -> dict[str:int]:
        return self._vocab

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def set_text(self, text: str) -> None:
        self._text = text
        self._vocab_f_tokens(self._text)

    def _vocab_f_tokens(self) -> dict[str:int]:
        output = re.split(self._re_encode, self._text)
        tokens = [item.strip() for item in output if item.strip()]
        all_words = sorted(set(tokens))
        all_words.extend(_SPECIAL_TOKENS)
        vocab = {token: integer for integer, token in enumerate(all_words)}
        return vocab


class Tokenizer:
    def __init__(self, vocab: dict[str:int]):
        self._re_encode = _RE_ENCODE
        self._re_decode = _RE_DECODE
        self._str_to_int = vocab
        self._int_to_str = {i: s for s, i in vocab.items()}

    def _tokenize(self, text: str) -> list[str]:
        output = re.split(self._re_encode, text)
        return [item.strip() for item in output if item.strip()]

    def encode(self, text) -> list[int]:
        preprocessed = self._tokenize(text)

        # Unknown words are converted to <|unk|> special token
        preprocessed = [
            item if item in self._str_to_int else "<|unk|>" for item in preprocessed
        ]

        return [self._str_to_int[s] for s in preprocessed]

    def decode(self, ids) -> str:
        text = " ".join([self._int_to_str[i] for i in ids])
        return re.sub(self._re_decode, r"\1", text)

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
# each unique word and special character to a unique integer

# Step 3 -- Adding special context tokens


_RE_ENCODE = r'([,.:;?_!"()\']|--|\s)'
_RE_DECODE = r'\s+([,.?!"()\'])'


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
        return [self._str_to_int[s] for s in preprocessed]

    def decode(self, ids) -> str:
        text = " ".join([self._int_to_str[i] for i in ids])
        return re.sub(self._re_decode, r"\1", text)

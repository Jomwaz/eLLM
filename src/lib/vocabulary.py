import re

from .config import (
    _RE_ENCODE,
    _SPECIAL_TOKENS,
)


class Vocabulary:
    def __init__(self, text: str):
        self._re_encode = _RE_ENCODE
        self._text = text
        self._vocab = self._vocab_f_tokens()

    @property
    def vocabulary(self) -> dict[str:int]:
        return self._vocab

    @property
    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text
        self._vocab_f_tokens()

    def _vocab_f_tokens(self) -> dict[str:int]:
        output = re.split(self._re_encode, self._text)
        tokens = [item.strip() for item in output if item.strip()]
        all_words = sorted(set(tokens))
        all_words.extend(_SPECIAL_TOKENS)
        vocab = {token: integer for integer, token in enumerate(all_words)}
        return vocab

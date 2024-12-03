class Text:
    def __init__(self, text: str = ""):
        self._text = text

    @property
    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text

    def extend_text(self, text: str) -> None:
        self._text = " <|endoftext|> ".join((self._text, text))

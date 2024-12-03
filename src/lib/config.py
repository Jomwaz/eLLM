_RE_ENCODE = r'([,.:;?_!"()\']|--|\s)'
_RE_DECODE = r'\s+([,.?!"()\'])'
_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|unk|>",
]
_ENCODINGS = [
    "gpt2",
]
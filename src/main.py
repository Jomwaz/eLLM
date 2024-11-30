import tokenizer
import tiktoken

with open("datasets/literature/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tiktokenizer = tiktoken.get_encoding("gpt2")
enc_text = tiktokenizer.encode(raw_text)
print(len(enc_text))
enc_sample = enc_text[50:]

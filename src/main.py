import tiktoken


def data_sampling_with_sliding_window():

    with open("datasets/literature/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))
    enc_sample = enc_text[50:]
    print(enc_sample)

    context_size = 4 # The context size determines how many tokens are included in the input
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size + 1]

    print(f"x: {x}")
    print(f"y:      {y}")



if __name__ == "__main__":
    data_sampling_with_sliding_window()

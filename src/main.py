import tokenizer

with open("datasets/literature/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

    vocab_model = tokenizer.Vocab(raw_text)
    token_engine = tokenizer.Tokenizer(vocab_model.vocab)

    # Test the vocab encoding.
    test_text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
    ids = token_engine.encode(test_text)
    print(ids)
    print(token_engine.decode(ids))

    # This test demonstrates need for large vocab sets.
    # Will produce a KeyError
    test_text_2 = "Hello, do you like tea?"
    print(token_engine.encode(test_text_2))

from lib.vocabulary import Vocabulary


def test_get_vocabulary():
    text = "Hello there, this is a test."
    vocab_instance = Vocabulary(text)

    assert vocab_instance.text == "Hello there, this is a test."


def test_get_vocab():
    text = "Hello there, this is a test."
    vocab_instance = Vocabulary(text)

    assert vocab_instance.vocabulary == {
        ",": 0,
        ".": 1,
        "<|endoftext|>": 8,
        "<|unk|>": 9,
        "Hello": 2,
        "a": 3,
        "is": 4,
        "test": 5,
        "there": 6,
        "this": 7,
    }


def test_set_text():
    text = "Hello there, this is a test."
    vocab_instance = Vocabulary(text)
    new_text = "Hello, test this is."
    vocab_instance.set_text(new_text)

    assert vocab_instance.text == "Hello, test this is."

    assert vocab_instance.vocabulary == {
        ",": 0,
        ".": 1,
        "<|endoftext|>": 8,
        "<|unk|>": 9,
        "Hello": 2,
        "a": 3,
        "is": 4,
        "test": 5,
        "there": 6,
        "this": 7,
    }

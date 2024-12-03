from lib.text import Text


def test_get_text():
    text = "Hello, world. This, is a test."
    text_instance = Text(text)

    assert text_instance.text == "Hello, world. This, is a test."


def test_set_text():
    text = "Hello, world. This, is a test."
    text_instance = Text(text)
    new_text = "I am overwriting here."
    text_instance.set_text(new_text)

    assert text_instance.text == "I am overwriting here."


def test_extend_text():
    text = "Hello, world. This, is a test."
    text_instance = Text(text)
    extension_text = "And I am another text test."
    text_instance.extend_text(extension_text)

    assert (
        text_instance.text
        == "Hello, world. This, is a test. <|endoftext|> And I am another text test."
    )

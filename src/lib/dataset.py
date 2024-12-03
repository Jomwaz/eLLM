import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from tiktoken import Encoding

from .config import (
    _ENCODINGS,
)


class GPTDataset(Dataset):
    def __init__(self, text: str, tokenizer: Encoding, max_length: int, stride: int):
        self.input_ids = []
        self.output_ids = []

        # Tokenize the text using tiktoken tokenizer encoding.
        token_ids = tokenizer.encode(text)

        # Use a sliding window to chunk the entire text into overlapping sequences of max_length length.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.output_ids.append(torch.tensor(target_chunk))

    def _len_(self):
        return len(self.input_ids)

    def _getitem(self, idx: int):
        """Returns a single row from the dataset.

        # Args:
            **idx**: *int*
            Desired row index.

        """
        return self.input_ids[idx], self.output_ids[idx]


def create_data_loader(
    text: str,
    encoding_name: str,
    batch_size: int = 1,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    r"""Creates a pyTorch DataLoader class.

    # Args:
        **text**: *str*
        Training text to load into tokenizer.

        **encoding_name**: *str*
        Name of encoding to use when instantiating tokenizer.

        **batch_size**: *int*, *optional*
        How many samples per a batch to load.

        **max_length**: *int*, *optional*
        Token ID count per tensor.

        **stride**: *bool*, *optional*
        Determines overlap between input/output tensors.

        **shuffle**: *bool*, *optional*
        set to `True` to have data shuffled at every epoch.

        **drop_last**: *bool*, *optional*
        set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
        If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.

        **num_workers**: *int*, *optional*
        how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)

    """

    if encoding_name not in _ENCODINGS:
        raise ValueError(f"Unknown encoding {encoding_name}.\n")

    tokenizer = tiktoken.get_encoding(encoding_name)
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader

"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

import torch
from .base import Tokenizer, get_stats, merge, merge_gpu,merge_gpu_vectorized,get_stats_with_numpy
from tqdm import tqdm
import time
from typing import List


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
        

    def train(self, text:str, vocab_size:int, verbose=False, prompt_interval: int = 10):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255
        ids_tensor = torch.tensor(ids, dtype=torch.int32).cuda()

        # iteratively merge the most common pairs to create new tokens
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes
        
        merges = {}  # (int, int) -> int

        with tqdm(total=num_merges) as pbar:
            for i in range(num_merges):
                # count up the number of times every consecutive pair appears
                start_time = time.perf_counter()
                stats = get_stats_with_numpy(ids_tensor.cpu().numpy())
                end_time = time.perf_counter()
                get_stats_time = (end_time - start_time) / 60
                # find the pair with the highest count
                pair = max(stats, key=stats.get)
                # mint a new token: assign it the next available id
                idx = 256 + i
                # replace all occurrences of pair in ids with idx
                start_time = time.perf_counter()
                ids_tensor = merge_gpu_vectorized(ids_tensor, pair, idx)
                end_time = time.perf_counter()
                merge_time = (end_time - start_time) / 60
                # save the merge
                merges[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
                self.vocab = vocab
                # prints
                if verbose:
                    pbar.set_description(
                        f"get_stats took {get_stats_time:.2f} minutes, merge took {merge_time:.2f} minutes. merge {i+1}/{num_merges}: {pair}({self.decode(pair)}) -> {idx} ({vocab[idx]}) had {stats[pair]:,} occurrences"
                    )
                pbar.update(1)
                if i % prompt_interval == 0 and i > 0:
                    answer = input(
                        f"After {i+1} merges vocab size is {len(vocab)}. Continue? (y,yes or n,no)"
                    )
                    if answer in ["n", "no"]:
                        break

        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab  # used in decode()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids



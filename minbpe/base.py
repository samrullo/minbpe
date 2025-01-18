"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""

import unicodedata
import torch
from typing import Tuple, List

# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer


def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):  # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

import numpy as np

def get_stats_with_numpy(ids: np.ndarray, exclude_values:np.ndarray):
    # Compute pairs
    # We are using numpy and bit operations to speed up calculation of pair counts. numpy array achieves parallelization through vector operations

    #The code effectively pairs consecutive elements of ids into single integers, 
    # with the earlier element occupying the higher 32 bits and the subsequent element the lower 32 bits of each 64-bit integer. 
    # This technique could be useful for creating unique identifiers from pairs of numbers, ensuring that the pair (a, b) is distinct 
    # from (b, a) if a and b are different.
    pairs = ids[:-1] * (1 << 32) + ids[1:]  # Pack consecutive pairs into a single integer
    
    # Mask to exclude pairs where either element is a space or punctuation mark represented by array exclude_values
    mask = ~np.isin(ids[:-1], exclude_values) & ~np.isin(ids[1:], exclude_values)
    # filtered_pairs now excludes pairs where at least one component is in exclude_values
    # we want to ignore pairs where one component is a space or punctuation mark. we don't want to treat combination of character and punctuation mark as a token
    # instead punctuation mark is treated as a separate token
    filtered_pairs = pairs[mask]  # Apply the mask to pairs

    # Count unique pairs
    unique, pair_counts = np.unique(filtered_pairs, return_counts=True)
    # p>>32 extracts first number from 64 bit integer, p & (1<<32-1) extracts second number from 64 bit integer
    # our filtered_pairs array combines pair like (356, 1230) into a single 64 bit integer like 345678987234555 (not accurate)
    # then from that big 64 bit integer like 345678987234555 we extract (356, 1230)
    pair_dict = {(p >> 32, p & ((1 << 32) - 1)): count for p, count in zip(unique, pair_counts)}

    return pair_dict


def merge(ids: List[int], pair: Tuple[int, int], idx: int):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def merge_gpu(ids: torch.Tensor, pair: Tuple[int, int], idx: int) -> torch.Tensor:
    """
    Replace all consecutive occurrences of `pair` in `ids` with `idx` using GPU acceleration.

    Args:
        ids (torch.Tensor): 1D tensor of integers representing the list.
        pair (Tuple[int, int]): The pair of integers to replace.
        idx (int): The new token ID to insert.

    Returns:
        torch.Tensor: 1D tensor with the replaced tokens.
    """
    # Create a mask where the pair is detected
    match_mask = (ids[:-1] == pair[0]) & (ids[1:] == pair[1])

    # Create a result array with the size accounting for replacements
    result = torch.full(
        (ids.size(0) - match_mask.sum().item(),), -1, dtype=ids.dtype, device=ids.device
    )

    i, j = 0, 0  # Pointers for the original and new arrays
    while i < len(ids):
        if i < len(ids) - 1 and match_mask[i]:  # If pair matches
            result[j] = idx
            i += 2  # Skip the matched pair
        else:
            result[j] = ids[i]
            i += 1
        j += 1

    return result[:j]  # Truncate the result tensor to its final size

def merge_gpu_vectorized(ids: torch.Tensor, pair: Tuple[int, int], idx: int) -> torch.Tensor:
    """
    Fully vectorized GPU implementation to replace all consecutive occurrences
    of `pair` in `ids` with `idx`.

    Args:
        ids (torch.Tensor): 1D tensor of integers representing the list.
        pair (Tuple[int, int]): The pair of integers to replace.
        idx (int): The new token ID to insert.

    Returns:
        torch.Tensor: 1D tensor with the replaced tokens.
    """
    # Identify where the pair starts
    match_starts = (ids[:-1] == pair[0]) & (ids[1:] == pair[1])

    # Create an output tensor for indices
    keep = torch.ones(ids.size(0), dtype=torch.bool, device=ids.device)

    # ~ inverts boolean values. keep[1:] will represent second element of each pair.
    # match_starts has True where first element of pair is detected. By inverting that 
    # and doing AND operation against second elements we set indices wher second element of pair occurs to False
    keep[1:] &= ~match_starts  # Remove the second element of each pair

    # Assign replacement indices. nonzero() returns a 2D tensor by default. each row has indices along dimensions where input_tensor is non-zero.
    # as_tuple will return all those rows as elements of tuple. that way we can extract 1D tensor with locations where pair starts in the original ids list
    ids[match_starts.nonzero(as_tuple=True)[0]] = idx

    # Filter the output to keep non-removed elements
    return ids[keep]


# first two helper functions...
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)  # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}")  # escape
    return "".join(chars)


def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode("utf-8", errors="replace")
    s = replace_control_characters(s)
    return s


# -----------------------------------------------------------------------------
# the base Tokenizer class


class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {}  # (int, int) -> int
        self.pattern = ""  # str
        self.special_tokens = {}  # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab()  # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, "w") as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, "r", encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

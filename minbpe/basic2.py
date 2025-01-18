"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""
import numpy as np
import torch
from .base import Tokenizer, get_stats, merge, merge_gpu,merge_gpu_vectorized,get_stats_with_numpy
from tqdm import tqdm
import time
from typing import List


class BasicTokenizer(Tokenizer):

    def __init__(self, all_chars:List[str], exclude_chars:List[str]):
        super().__init__()
        self.all_chars= all_chars
        self.exclude_chars=exclude_chars
        self.all_chars_bytes = [letter.encode("utf-8") for letter in all_chars]
        self.exclude_chars_bytes = [letter.encode("utf-8") for letter in exclude_chars]
        cyrillic_text="АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяЎҒҚҲўғқҳ"
        self.cyrillic_chars = list(cyrillic_text)

        word_to_ids = {word:idx for idx,word in enumerate(self.all_chars_bytes)}
        id_to_words = {id:word for word,id in word_to_ids.items()}

        # when counting pairs in the corpus I don't want to count pairs with certain bytes that represent punctuation marks, space or new line
        # that's the purpose of exclude_ids
        self.exclude_ids = np.array([id for word,id in word_to_ids.items() if word in self.exclude_chars_bytes])

        self.word_to_ids=word_to_ids
        self.id_to_words=id_to_words


    def train(self, text:str, num_merges:int, verbose=False, prompt_interval: int = 10):
        # input text preprocessing
        # first we want to assign id s to every single byte. there are 256 of them in total
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes
        merges={} # (int,int) -> int

        # process every letter in all_chars. there are those that consists of more than one byte
        # when encounter such letters made of 2 bytes or 3 bytes, add a record in merges and update vocab with new id and word                
        for letter in self.all_chars:
            # first express a letter as a list of integers where each integer represents a single byte
            ids = list(letter.encode("utf-8"))
            while len(ids) > 1:
                # if a letter consists of more than one byte then retrieve pairs from its bytes
                pairs = list(zip(ids[:-1],ids[1:]))
                for pair in pairs:
                    if pair not in merges:
                        # assign the pair a new id and add that id to vocabulary
                        idx = len(vocab)
                        merges[pair] = idx
                        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
                ids = merge(ids, pair, idx)

        
        word_to_id = {word : id for id,word in vocab.items()}
        exclude_ids = np.array([id for id,word in vocab.items() if word in self.exclude_chars_bytes])
        ids = [word_to_id[letter.encode("utf-8")] for letter in list(text)]
        ids_tensor = torch.tensor(ids, dtype=torch.int32).cuda()

        with tqdm(total=num_merges) as pbar:
            for i in range(num_merges):
                # count up the number of times every consecutive pair appears
                start_time = time.perf_counter()
                stats = get_stats_with_numpy(ids_tensor.cpu().numpy(), exclude_ids)
                end_time = time.perf_counter()
                get_stats_time = (end_time - start_time) / 60
                # find the pair with the highest count
                pair = max(stats, key=stats.get)
                # mint a new token: assign it the next available id
                idx = len(vocab)
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
            #stats = get_stats(ids)
            stats = get_stats_with_numpy(np.array(ids),self.exclude_ids)
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
    
    # def load(self, model_file):
    #     # first execute load method of the parent
    #     super().load(model_file)

    #     # now extend the vocabulary with self.id_to_word


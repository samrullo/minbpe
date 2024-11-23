class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        # Add Uzbek Cyrillic characters to the vocabulary
        self.cyrillic_chars = [
            'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 
            'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я', 'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 
            'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 
            'Ў', 'Ғ', 'Қ', 'Ҳ', 'ў', 'ғ', 'қ', 'ҳ'
        ]

    def train(self, text, vocab_size, verbose=False, prompt_interval: int = 10):
        assert vocab_size >= 256 + len(self.cyrillic_chars)
        num_merges = vocab_size - 256 - len(self.cyrillic_chars)

        # Initialize vocab with bytes and Cyrillic characters
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255

        # Add Cyrillic characters to the vocabulary
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes
        for i, char in enumerate(self.cyrillic_chars, start=256):
            vocab[i] = char.encode("utf-8")  # Add Cyrillic chars to vocab

        # Merge byte pairs representing Cyrillic characters into single tokens
        cyrillic_byte_pairs = {
            tuple(char.encode("utf-8")): 256 + i for i, char in enumerate(self.cyrillic_chars)
        }
        new_ids = []
        skip_next = False
        for j in range(len(ids)):
            if skip_next:
                skip_next = False
                continue
            if j + 1 < len(ids) and (ids[j], ids[j + 1]) in cyrillic_byte_pairs:
                # Merge the byte pair into a single token ID
                new_ids.append(cyrillic_byte_pairs[(ids[j], ids[j + 1])])
                skip_next = True  # Skip the next byte since it's part of the pair
            else:
                new_ids.append(ids[j])
        ids = new_ids  # Replace the old ids with the processed ones

        merges = {}  # (int, int) -> int

        with tqdm(total=num_merges) as pbar:
            for i in range(num_merges):
                start_time = time.perf_counter()
                stats = get_stats(ids)
                end_time = time.perf_counter()
                get_stats_time = (end_time - start_time) / 60

                # Find the pair with the highest count
                pair = max(stats, key=stats.get)
                idx = 256 + len(self.cyrillic_chars) + i

                # Replace occurrences of the pair in ids with idx
                start_time = time.perf_counter()
                ids = merge(ids, pair, idx)
                end_time = time.perf_counter()
                merge_time = (end_time - start_time) / 60

                # Save the merge
                merges[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
                self.vocab = vocab

                if verbose:
                    pbar.set_description(
                        f"get_stats took {get_stats_time:.2f} minutes, merge took {merge_time:.2f} minutes. "
                        f"merge {i + 1}/{num_merges}: {pair}({self.decode(pair)}) -> {idx} ({vocab[idx]}) had {stats[pair]:,} occurrences"
                    )
                pbar.update(1)
                if i % prompt_interval == 0 and i > 0:
                    answer = input(f"After {i + 1} merges vocab size is {len(vocab)}. Continue? (y,yes or n,no)")
                    if answer in ["n", "no"]:
                        break

        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        # Use Cyrillic tokens if the character is in the vocabulary
        ids = []
        for char in text:
            if char in self.cyrillic_chars:
                idx = 256 + self.cyrillic_chars.index(char)
                ids.append(idx)
            else:
                ids += list(char.encode("utf-8"))
        return ids

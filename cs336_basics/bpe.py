import regex as re
from collections import defaultdict
import multiprocessing as mp
from .pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def process_chunk(args):
    input_path, start, end, special_tokens, special_token_to_id = args
    local_word_counting = defaultdict(int)
    special_token_set = set(special_tokens)

    with open(input_path, mode='rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    if "\r" in chunk:
        chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")

    parts = split_by_special(chunk, special_tokens, drop_special=False)

    for part in parts:
        if part in special_token_set:
            local_word_counting[(special_token_to_id[part],)] += 1
        else:
            chunk_word_counting = pre_tokenization(part)
            for word, count in chunk_word_counting.items():
                local_word_counting[word] += count

    return local_word_counting

def bpe(input_path, vocab_size, special_tokens):
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size is too small")
    merges = []
    vocab = {i: bytes([i]) for i in range(256)}
    special_token_to_id = vocab_add_special_tokens(special_tokens, vocab)
    word_counting = defaultdict(int)
    num_processes = min(8, mp.cpu_count())

    with open(input_path, mode='rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = [
        (input_path, start, end, special_tokens, special_token_to_id)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, tasks)

    for local_word_counting in results:
        for word, count in local_word_counting.items():
            word_counting[word] += count
    while len(vocab) < vocab_size: 
        pair_counting = count_pair(word_counting) 
        if not pair_counting: 
            break
        max_pair = find_largest_pair(pair_counting, vocab)
        new_token_id = len(vocab)
        word_counting = apply_merge(max_pair, new_token_id, word_counting)
        vocab[new_token_id] = vocab[max_pair[0]] + vocab[max_pair[1]]
        merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
    return vocab, merges

def vocab_add_special_tokens(special_tokens, vocab):
    index = len(vocab)
    special_token_to_id = {}
    for special_token in special_tokens:
        vocab[index] = special_token.encode("utf-8")
        special_token_to_id[special_token] = index
        index += 1
    return special_token_to_id

def pre_tokenization(text):
    matches = re.finditer(PAT, text)
    word_counting = defaultdict(int)
    for match in matches:
        match_text = match.group()
        word_counting[word_to_byte(match_text)] += 1
    return word_counting
##  convert documents to words

def count_pair(word_counting):
    pair_counting = defaultdict(int)
    for word, count in word_counting.items():
        word_length = len(word)
        if (word_length < 2):
            continue
        for i in range(word_length - 1):
            pair_counting[(word[i], word[i + 1])] += count
    return pair_counting

def find_largest_pair(pair_counting, vocab):
    max_pair, _ = max(pair_counting.items(), key=lambda x: (x[1], (vocab[x[0][0]], vocab[x[0][1]])))
    return max_pair

def apply_merge(max_pair, new_token_id, word_counting):
    new_word_counting = defaultdict(int)
    a, b = max_pair

    for word, count in word_counting.items():

        if (a, b) not in zip(word[:-1], word[1:]):
            new_word_counting[word] += count
            continue
        n = len(word)
        i = 0
        new_word = None 

        while i < n:
            if i < n - 1 and word[i] == a and word[i + 1] == b:
                if new_word is None:
                    new_word = list(word[:i])  
                new_word.append(new_token_id)
                i += 2
            else:
                if new_word is not None:
                    new_word.append(word[i])
                i += 1

        if new_word is None:
            new_word_counting[word] += count
        else:
            new_word_counting[tuple(new_word)] += count

    return new_word_counting

def word_to_byte(word):
    return tuple(word.encode('utf-8'))

def split_by_special(text, special_tokens, drop_special=True):
    if not special_tokens:
        return [text]

    special_tokens = sorted(special_tokens, key=len, reverse=True)

    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special:
        pattern = f"({pattern})"

    pattern = re.compile(pattern)
    chunks = pattern.split(text)
    return [c for c in chunks if c]




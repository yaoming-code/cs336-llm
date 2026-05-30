from typing import Iterator, Iterable
import json
from .bpe import split_by_special
import regex as re
PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
def word2bytes(word):
    a = list(word.encode('utf-8'))
    return tuple(bytes([i]) for i in a)
def split_to_words(text):
    "Split text into words."
    return PAT.findall(text)

def apply_merges(word_bytes, merges_set, vocab_to_id):
    word_bytes = list(word_bytes)
    
    while True:
        min_token_id = float('inf')
        best_pair_idx = -1
        merged = None

        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            if pair in merges_set:
                combined = pair[0] + pair[1]
                token_id = vocab_to_id.get(combined)
                if token_id is not None and token_id < min_token_id:
                    min_token_id = token_id
                    best_pair_idx = i
                    merged = combined

        if best_pair_idx == -1:
            break

        # Apply best merge
        word_bytes = (
            word_bytes[:best_pair_idx]
            + [merged]
            + word_bytes[best_pair_idx + 2:]
        )

    return tuple(word_bytes)

def encode_merged(text,merges,vocab_to_id):
    word_list = split_to_words(text)
    tokens=[]
    for word in word_list:
        word_bytes=word2bytes(word)
        merged_word_bytes = apply_merges(word_bytes,merges,vocab_to_id)
        tokens.extend(vocab_to_id[i] for i in merged_word_bytes)
    return tokens

# Can merge the above functions into below class

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = set(merges)
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [i.encode('utf-8') for i in self.special_tokens]
        

        self.vocab_to_id={v:k for k,v in vocab.items()}

        # Ensure special tokens are in the vocabulary
        for token_bytes in self.special_tokens_bytes:
            if token_bytes not in self.vocab_to_id:
                # Add to vocab if not already present
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.vocab_to_id[token_bytes] = new_id


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # Load vocab (assumed to be a JSON file: {token_id: byte_string})
        with open(vocab_filepath, 'r', encoding='utf-8') as vf:
            vocab_data = json.load(vf)
            # Optional: convert keys to int if stored as strings
            vocab = {int(k): bytes(v, 'latin1') if isinstance(v, str) else bytes(v) 
                     for k, v in vocab_data.items()}

        # Load merges (assumed to be a list of pairs like: "a b")
        with open(merges_filepath, 'r', encoding='utf-8') as mf:
            lines = mf.readlines()
            # Optional: skip headers like "#version: 0.2"
            merge_pairs = [tuple(line.strip().split()) for line in lines if not line.startswith('#') and line.strip()]
            # Convert to byte-pairs
            merges = [(a.encode('utf-8'), b.encode('utf-8')) for a, b in merge_pairs]

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def encode(self, text: str) -> list[int]:
        chunks = split_by_special(text, self.special_tokens, drop_special=False)
        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.vocab_to_id[chunk.encode('utf-8')])
            else:
                tokens.extend(encode_merged(chunk, self.merges, self.vocab_to_id))
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        "Decode a sequence of token IDs into text."
        return b''.join([self.vocab[t] for t in ids]).decode('utf-8',errors='replace')
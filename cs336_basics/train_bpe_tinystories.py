from pathlib import Path
import json
import time
from bpe import bpe

input_path = Path("/mnt/d/CS336/data/TinyStoriesV2-GPT4-train.txt")
output_dir = Path("/mnt/d/CS336/tokenizer")
output_dir.mkdir(parents=True, exist_ok=True)

start = time.perf_counter()
vocab, merges = bpe(
    input_path=input_path,
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
)
elapsed = time.perf_counter() - start

# 保存 merges
with open(output_dir / "merges.txt", "w", encoding="utf-8") as f:
    for a, b in merges:
        f.write(f"{a!r}\t{b!r}\n")

# 保存 vocab
serializable_vocab = {k: v.hex() for k, v in vocab.items()}
with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
    json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)

print("seconds:", elapsed)
print("hours:", elapsed / 3600)
print("longest token:", max(vocab.values(), key=len))
print("saved to:", output_dir)
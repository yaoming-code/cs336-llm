from pathlib import Path
import cProfile
import pstats

from cs336_basics.bpe import bpe

input_path = Path("tests/fixtures/corpus.en")

profiler = cProfile.Profile()
profiler.enable()

bpe(
    input_path=input_path,
    vocab_size=500,
    special_tokens=["<|endoftext|>"],
)

profiler.disable()


output_filepath = "D:\CS336\profile"


with open(output_filepath, "w", encoding="utf-8") as f:

    stats = pstats.Stats(profiler, stream=f)

    stats.sort_stats("cumtime").print_stats(30)

print(f"性能分析结果已成功保存至: {output_filepath}")
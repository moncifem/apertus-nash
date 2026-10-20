import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OPEN, CLOSE, END = "<|inner_prefix|>", "<|inner_suffix|>", "<|assistant_end|>"


class CharTok:
    """Character-level fake tokenizer with Apertus-style special tokens."""

    SPECIAL = {OPEN: 1_000_001, CLOSE: 1_000_002, END: 1_000_003}

    def __init__(self):
        self.inv = {v: k for k, v in self.SPECIAL.items()}

    def encode(self, s: str) -> list[int]:
        ids, i = [], 0
        while i < len(s):
            for k, v in self.SPECIAL.items():
                if s.startswith(k, i):
                    ids.append(v)
                    i += len(k)
                    break
            else:
                ids.append(ord(s[i]))
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(("" if skip_special_tokens else self.inv[i]) if i in self.inv else chr(i) for i in ids)

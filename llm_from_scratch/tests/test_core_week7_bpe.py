import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from bpe_tokenizer import BPETokenizer  # noqa: E402


class TestWeek7BPETokenizer(unittest.TestCase):
    def test_bpe_encode_decode_wordwise_roundtrip(self) -> None:
        text = "hello   world\nthis is\na test"
        tok = BPETokenizer.train(text, num_merges=50)
        ids = tok.encode(text)
        out = tok.decode(ids)
        self.assertEqual(out.split(), text.split())


if __name__ == "__main__":
    unittest.main()


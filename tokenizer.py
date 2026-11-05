"""
STAGE 1 — A byte-level BPE tokenizer, written from scratch.

The model only works with integers, so text has to become a list of token ids
and back. BPE (Byte Pair Encoding) starts from the 256 possible byte values and
repeatedly glues the most frequent adjacent pair into a new token. After enough
merges, common words (" que", " não", " porque") are single tokens and rare
words stay split into pieces. Nothing is ever "unknown": any text is bytes, so
the worst case is more tokens, never an error. Same family as GPT-2/3/4.

    text --regex split--> chunks (" palavra", ",", " 1899") --utf-8--> bytes --merges--> token ids

Why it matters: the model sees ids, not letters. "Capitu" -> C|ap|itu and
"Capitú" -> C|ap|it|ú (run-2 tokenizer) are different inputs: the first leads to
"Capítulo", the second to Machado's heroine.

Explained step by step in docs/01-tokenizer.md.
Tests: pytest tests/test_stage1_tokenizer.py
"""
import json
from collections import Counter

import regex as re  # the `regex` package: supports \p{L} (any Unicode letter, incl. é, ã, ç)

# Pre-tokenization: cut the text into chunks before BPE, so a merge can never glue
# the end of one word to the start of the next. The leading space stays with the
# word (" casa"), like GPT-2, so " da" and "da" are different tokens.
SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT = "<|endoftext|>"  # special token separating documents (books)


def get_stats(words: dict) -> Counter:
    """Count every adjacent pair of ids, weighted by how often the word occurs.

    words: {(104, 105): 3, (104, 97): 1}  ->  Counter({(104, 105): 3, (104, 97): 1})
    """
    # `words` holds each distinct chunk once, with its frequency in the corpus.
    # Adding `freq` (instead of 1) gives the same counts as scanning the whole text,
    # while visiting each distinct chunk only once — this is what makes a pure-Python
    # BPE fast enough on Machado (4 M characters, run 1) or a 10 M-character sample
    # of the library (run 2, 444 s).
    stats = Counter()
    for word, freq in words.items():
        for a, b in zip(word, word[1:]):  # zip(w, w[1:]) -> (w0,w1), (w1,w2), ...
            stats[(a, b)] += freq
    return stats


def merge(ids: tuple, pair: tuple, new_id: int) -> tuple:
    """Replace every occurrence of `pair` in `ids` by `new_id`, scanning left to right.

    merge((1, 2, 3, 1, 2), (1, 2), 99) -> (99, 3, 99)
    """
    # On a match, emit the new id and skip BOTH positions, so merges never overlap:
    # merge((1, 1, 1), (1, 1), 7) -> (7, 1), not (7, 7).
    # Returns a tuple because the result is used as a dict key in `train`.
    result = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return tuple(result)


class BPETokenizer:
    def __init__(self):
        self.merges: dict = {}                               # (id_a, id_b) -> new_id, in creation order
        self.vocab: dict = {i: bytes([i]) for i in range(256)}  # id -> the bytes it stands for
        self.special: dict = {}                              # "<|endoftext|>" -> id
        self._pat = re.compile(SPLIT_PATTERN)
        self._cache: dict = {}                               # chunk -> its token ids (encode speed-up)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special)

    @property
    def eot_id(self) -> int:
        return self.special[EOT]

    # ------------------------------------------------------------------ training
    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert vocab_size > 257, "need room for 256 bytes + 1 special token"
        n_merges = vocab_size - 256 - 1  # the last slot is reserved for <|endoftext|>

        # Count unique chunks once; we merge over {chunk_bytes: frequency}, not the raw
        # text — that is what makes pure-Python BPE fast enough on a few MB.
        text = text.replace(EOT, "")
        chunk_counts = Counter(self._pat.findall(text))
        words = {tuple(chunk.encode("utf-8")): f for chunk, f in chunk_counts.items()}

        # One BPE step per loop: count pairs -> pick the most frequent -> merge it
        # everywhere -> record it. Each merge changes the words, which creates new
        # pairs (e.g. "n"+"ão" only exists after "ã"+"o" became "ão"), so the counts
        # are recomputed every round.
        for i in range(n_merges):
            new_id = 256 + i
            stats = get_stats(words)
            if not stats:  # every chunk is already a single token: nothing left to merge
                break
            pair = max(stats, key=stats.get)
            # Rewrite every word with the new merge; frequencies stay the same,
            # only the shape of each word changes.
            words = {merge(word, pair, new_id): freq for word, freq in words.items()}
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]  # bytes behind the new id
            if verbose and (i % 100 == 0 or i == n_merges - 1):
                shown = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"merge {i + 1}/{n_merges}: {pair} -> {new_id} {shown!r}")

        self.special = {EOT: len(self.vocab)}
        self._cache = {}

    # ------------------------------------------------------------------ encoding
    def _encode_chunk(self, chunk: bytes) -> list:
        """Apply the learned merges to one chunk, in the order they were learned."""
        # Start from raw bytes and replay the merges. At each round, apply the pair
        # with the LOWEST merge id (= learned earliest), not the most frequent one:
        # later merges are built from earlier ones ("não" = "n" + "ão"), so replaying
        # them in order cuts new text exactly the way training would have.
        # Pairs never learned get id = infinity, so they are never chosen; when the
        # best pair is one of those, nothing more can merge.
        ids = list(chunk)
        while len(ids) >= 2:
            pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
            best_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best_pair not in self.merges:
                break
            new_id = self.merges[best_pair]
            ids = list(merge(ids, best_pair, new_id))  # merge returns a tuple; keep a list
        return ids

    def encode(self, text: str) -> list:
        ids = []
        # <|endoftext|> must become its single special id, never be split into bytes.
        parts = text.split(EOT) if self.special else [text]
        for k, part in enumerate(parts):
            if k > 0:
                ids.append(self.eot_id)
            for chunk in self._pat.findall(part):
                # Common chunks (" que", ",") appear tens of thousands of times:
                # encode each distinct chunk once and reuse the result.
                cached = self._cache.get(chunk)
                if cached is None:
                    cached = self._encode_chunk(chunk.encode("utf-8"))
                    self._cache[chunk] = cached
                ids.extend(cached)
        return ids

    def decode(self, ids) -> str:
        # Look up the bytes behind each id and join them. errors="replace" turns an
        # incomplete character (e.g. half of "ã" picked alone by an untrained model)
        # into "�" instead of crashing.
        inv_special = {v: k.encode("utf-8") for k, v in self.special.items()}
        data = b"".join(self.vocab[i] if i in self.vocab else inv_special[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def save(self, path: str):
        # Only the merges need saving: the vocabulary can be rebuilt from them.
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": [[a, b, n] for (a, b), n in self.merges.items()],
                       "special": self.special}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        tok = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for a, b, n in data["merges"]:  # rebuild in creation order
            tok.merges[(a, b)] = n
            tok.vocab[n] = tok.vocab[a] + tok.vocab[b]
        tok.special = data["special"]
        return tok

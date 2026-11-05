from collections import Counter

from tokenizer import EOT, BPETokenizer, get_stats, merge

SAMPLE = ("Não tive filhos, não transmiti a nenhuma criatura o legado da nossa miséria. "
          "L'homme qui rit. Élémentaire, mon cher Watson! Ação, coração, emoção, nação. ") * 20


def test_1_1_get_stats():
    words = {(1, 2, 3): 2, (1, 2): 5}
    assert get_stats(words) == Counter({(1, 2): 7, (2, 3): 2})


def test_1_2_merge():
    assert merge((1, 2, 3, 1, 2), (1, 2), 99) == (99, 3, 99)
    assert merge((1, 1, 1), (1, 1), 7) == (7, 1)      # left to right, no overlap
    assert merge((5,), (1, 2), 9) == (5,)
    assert merge((), (1, 2), 9) == ()


def test_1_3_train_reaches_vocab_and_learns_frequent_pieces():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    assert tok.vocab_size == 300
    learned = {b.decode("utf-8", errors="ignore") for i, b in tok.vocab.items() if i >= 256}
    assert any("ção" in s for s in learned), "a very frequent Portuguese suffix should be merged"


def test_1_4_roundtrip_and_compression():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=320)
    text = "Ação e emoção: o coração da nação. 🙂 日本"   # includes bytes never seen in training
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert len(tok.encode(SAMPLE)) < len(SAMPLE.encode("utf-8")) * 0.6, "BPE should compress"


def test_special_token_and_save_load(tmp_path):
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    ids = tok.encode(f"livro um{EOT}livro dois")
    assert ids.count(tok.eot_id) == 1
    path = tmp_path / "tok.json"
    tok.save(str(path))
    tok2 = BPETokenizer.load(str(path))
    assert tok2.encode(SAMPLE[:200]) == tok.encode(SAMPLE[:200])
    assert tok2.decode(ids) == f"livro um{EOT}livro dois"

"""
STAGE 6 — Build the pretraining corpus: open 19th-century Portuguese, Machado excluded.

    python prepare_pretrain.py --source gutenberg                    # all Portuguese books on Project Gutenberg
    python prepare_pretrain.py --source pleias --target_words 60e6   # cleanest part of PleIAs/Portuguese-PD
    python prepare_pretrain.py --source all                          # both, in that order

Writes, under data/pretrain/:
    corpus.txt       documents separated by <|endoftext|>
    manifest.csv     one line per document: source, id, creator, title, date, words, quality score
    raw/gutenberg/   cached downloads, so an interrupted run resumes where it stopped

Why two sources: Gutenberg texts are human-proofread but there are only ~1,000 of them;
Portuguese-PD has 7,840 titles (672 M words) but is OCR text with transcription errors.
The Gutenberg vocabulary is used as a dictionary to score each PD document, and only the
cleanest documents are kept.

Machado de Assis is excluded from both sources: his books are the fine-tuning set and the
validation set, so they must not be seen during pretraining.

See docs/06-training-runs.md.
"""
import argparse
import sys
import csv
import json
import os
import re
import time
import unicodedata
from collections import Counter

import requests

from prepare_data import HEADERS, norm, strip_gutenberg

EOT = "<|endoftext|>"
OUT_DIR = "data/pretrain"
EXCLUDE_AUTHORS = ["machado de assis", "assis machado", "assis, machado"]
# Reference works add little to a language model and skew the vocabulary.
EXCLUDE_TITLE_WORDS = ["diccionario", "dicionario", "vocabulario", "grammatica", "gramatica",
                       "orthographia", "ortografia", "catalogo", "almanach", "almanaque"]

WORD_RE = re.compile(r"[a-záàâãéêíóôõúçüñ]+")


# ------------------------------------------------------------------ helpers
def is_excluded_author(name: str) -> bool:
    n = norm(name or "")
    return any(x in n for x in ["machado de assis", "assis machado"]) or \
        ("machado" in n and "assis" in n)


def is_reference_work(title: str) -> bool:
    t = norm(title or "")
    return any(w in t for w in EXCLUDE_TITLE_WORDS)


def words_of(text: str) -> list:
    return WORD_RE.findall(text.lower())


def quality_score(text: str, dictionary: set) -> float:
    """Share of words (3+ letters) that exist in the clean reference vocabulary."""
    ws = [w for w in words_of(text) if len(w) >= 3]
    if len(ws) < 200:
        return 0.0
    return sum(w in dictionary for w in ws) / len(ws)


def letter_ratio(text: str) -> float:
    """Share of characters that are letters, spaces or common punctuation (OCR noise is the rest)."""
    if not text:
        return 0.0
    ok = sum(c.isalpha() or c.isspace() or c in ".,;:!?-—«»\"'()" for c in text)
    return ok / len(text)


def clean_ocr(text: str) -> str:
    """Light cleanup for scanned text: join hyphenated line breaks, rebuild paragraphs."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)                 # "senti-\nmento" -> "sentimento"
    paragraphs = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paragraphs = [re.sub(r"[ \t]+", " ", p) for p in paragraphs if len(p.strip()) > 0]
    return "\n\n".join(paragraphs)


class CorpusWriter:
    """Appends documents to corpus.txt and lines to manifest.csv; tracks totals."""

    def __init__(self, out_dir: str, append: bool):
        os.makedirs(out_dir, exist_ok=True)
        self.corpus_path = os.path.join(out_dir, "corpus.txt")
        self.manifest_path = os.path.join(out_dir, "manifest.csv")
        mode = "a" if append and os.path.exists(self.corpus_path) else "w"
        self.corpus = open(self.corpus_path, mode, encoding="utf-8")
        self.first = mode == "w" or os.path.getsize(self.corpus_path) == 0
        new_manifest = mode == "w" or not os.path.exists(self.manifest_path)
        self.manifest = open(self.manifest_path, "a" if not new_manifest else "w",
                             newline="", encoding="utf-8")
        self.writer = csv.writer(self.manifest)
        if new_manifest:
            self.writer.writerow(["source", "id", "creator", "title", "date", "words", "quality"])
        self.docs, self.words = 0, 0
        if not new_manifest:  # resume totals from what is already on disk
            with open(self.manifest_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.docs += 1
                    self.words += int(row["words"])

    def add(self, text: str, source: str, doc_id, creator: str, title: str, date: str, quality: float):
        if not self.first:
            self.corpus.write(f"\n{EOT}\n")
        self.first = False
        self.corpus.write(text)
        n_words = len(words_of(text))
        self.writer.writerow([source, doc_id, (creator or "")[:120], (title or "")[:160],
                              date or "", n_words, f"{quality:.3f}"])
        self.docs += 1
        self.words += n_words

    def close(self):
        self.corpus.close()
        self.manifest.close()


# ------------------------------------------------------------------ source 1: Gutenberg
def http_get(url: str, tries: int = 4, **kw) -> requests.Response:
    """GET with retries and backoff: Gutenberg's servers occasionally stall."""
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=kw.pop("timeout", 60), **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == tries - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"  retry in {wait}s ({e})")
            time.sleep(wait)


def gutenberg_catalog(out_dir: str, lang: str = "pt") -> list:
    """Every Gutenberg text in the language, from the official catalogue CSV (one download).

    https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv has one row per ebook:
    Text#, Type, Issued, Title, Language, Authors, Subjects, LoCC, Bookshelves.
    """
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    cache = os.path.join(raw_dir, "pg_catalog.csv")
    if not os.path.exists(cache) or time.time() - os.path.getmtime(cache) > 30 * 86400:
        print("downloading the Project Gutenberg catalogue (~15 MB) ...")
        r = http_get("https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv", timeout=120)
        with open(cache, "wb") as f:
            f.write(r.content)
    books = []
    with open(cache, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["Type"] != "Text" or row["Language"].strip() != lang:
                continue
            books.append({"id": int(row["Text#"]), "title": row["Title"],
                          "creator": row["Authors"] or "Anonymous", "date": row["Issued"]})
    print(f"Gutenberg catalogue: {len(books)} Portuguese-language texts")
    return books


def run_gutenberg(writer: CorpusWriter, out_dir: str, max_books: int):
    raw_dir = os.path.join(out_dir, "raw", "gutenberg")
    os.makedirs(raw_dir, exist_ok=True)
    books = gutenberg_catalog(out_dir)
    skipped = Counter()
    seen_titles = set()
    kept = 0
    for i, b in enumerate(books[:max_books]):
        if is_excluded_author(b["creator"]):
            skipped["machado"] += 1
            continue
        if is_reference_work(b["title"]):
            skipped["reference work"] += 1
            continue
        key = norm(b["title"])
        if key in seen_titles:
            skipped["duplicate title"] += 1
            continue
        seen_titles.add(key)

        cache = os.path.join(raw_dir, f"{b['id']}.txt")
        if os.path.exists(cache):
            raw = open(cache, encoding="utf-8").read()
        else:
            try:
                r = http_get(f"https://www.gutenberg.org/ebooks/{b['id']}.txt.utf-8")
                r.encoding = "utf-8"
                raw = r.text
            except requests.RequestException as e:
                print(f"  skip {b['id']} ({e})")
                skipped["download error"] += 1
                continue
            with open(cache, "w", encoding="utf-8") as f:
                f.write(raw)
            time.sleep(1.0)  # be polite to Gutenberg's servers
        text = strip_gutenberg(raw)
        if len(text) < 20_000:
            skipped["too short"] += 1
            continue
        writer.add(text, "gutenberg", b["id"], b["creator"], b["title"], b["date"], 1.0)
        kept += 1
        if kept % 25 == 0 or kept == 1:
            print(f"  [{kept:4d}] {i + 1}/{len(books)} | {b['title'][:50]:50s} | "
                  f"{writer.words / 1e6:6.1f}M words so far")
    print(f"Gutenberg: kept {kept}, skipped {dict(skipped)}")


# ------------------------------------------------------------------ source 2: PleIAs/Portuguese-PD
def build_dictionary(corpus_path: str, min_count: int = 3) -> set:
    """Words seen at least `min_count` times in the clean Gutenberg text."""
    counts = Counter()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            counts.update(w for w in words_of(line) if len(w) >= 3)
    return {w for w, c in counts.items() if c >= min_count}


def run_pleias(writer: CorpusWriter, out_dir: str, target_words: float, min_quality: float,
               max_doc_words: int, dictionary: set):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo = "PleIAs/Portuguese-PD"
    files = [f"portuguese_pd_{i}.parquet" for i in range(1, 8)]
    stats = Counter()
    scores = []
    for fname in files:
        if writer.words >= target_words:
            break
        print(f"PleIAs: downloading {fname} (~700 MB; cached by huggingface_hub) ...")
        path = hf_hub_download(repo, fname, repo_type="dataset",
                               cache_dir=os.path.join(out_dir, "raw", "hf_cache"))
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64, columns=["identifier", "creator", "title",
                                                             "publication_date", "word_count", "text"]):
            rows = batch.to_pylist()
            for row in rows:
                if writer.words >= target_words:
                    break
                if is_excluded_author(row.get("creator") or "") or \
                        is_excluded_author(row.get("title") or ""):
                    stats["machado"] += 1
                    continue
                if is_reference_work(row.get("title") or ""):
                    stats["reference work"] += 1
                    continue
                text = clean_ocr(row.get("text") or "")
                if letter_ratio(text) < 0.93:
                    stats["ocr noise"] += 1
                    continue
                q = quality_score(text, dictionary)
                scores.append(q)
                if q < min_quality:
                    stats["low quality"] += 1
                    continue
                ws = text.split(" ")
                if len(ws) > max_doc_words:  # cap very long periodicals so no single title dominates
                    text = " ".join(ws[:max_doc_words])
                    stats["truncated"] += 1
                writer.add(text, "pleias", row.get("identifier"), row.get("creator"),
                           row.get("title"), str(row.get("publication_date") or ""), q)
                stats["kept"] += 1
                if stats["kept"] % 100 == 0:
                    print(f"  kept {stats['kept']:5d} | {writer.words / 1e6:6.1f}M words | "
                          f"last quality {q:.2f} | {(row.get('title') or '')[:50]}")
            if writer.words >= target_words:
                break
    if scores:
        scores.sort()
        print(f"PleIAs quality scores: median {scores[len(scores) // 2]:.2f}, "
              f"10th pct {scores[len(scores) // 10]:.2f}, 90th pct {scores[9 * len(scores) // 10]:.2f}")
    print(f"PleIAs: {dict(stats)}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["gutenberg", "pleias", "all"], default="all")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--max_books", type=int, default=5000, help="Gutenberg: cap on books to consider")
    ap.add_argument("--target_words", type=float, default=60e6,
                    help="PleIAs: stop when the whole corpus reaches this many words")
    ap.add_argument("--min_quality", type=float, default=0.88,
                    help="PleIAs: keep documents whose share of dictionary words is at least this")
    ap.add_argument("--max_doc_words", type=int, default=150_000)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # show progress live, even through `tee`

    t0 = time.time()
    if args.source in ("gutenberg", "all"):
        writer = CorpusWriter(args.out_dir, append=False)
        run_gutenberg(writer, args.out_dir, args.max_books)
        writer.close()
        print(f"after Gutenberg: {writer.docs} documents, {writer.words / 1e6:.1f}M words")

    if args.source in ("pleias", "all"):
        corpus_path = os.path.join(args.out_dir, "corpus.txt")
        if not os.path.exists(corpus_path) or os.path.getsize(corpus_path) < 1_000_000:
            raise SystemExit("Run --source gutenberg first: its text is the dictionary used to "
                             "score the OCR documents.")
        dictionary = build_dictionary(corpus_path)
        print(f"dictionary from Gutenberg text: {len(dictionary):,} words")
        writer = CorpusWriter(args.out_dir, append=True)
        run_pleias(writer, args.out_dir, args.target_words, args.min_quality,
                   args.max_doc_words, dictionary)
        writer.close()

    with open(os.path.join(args.out_dir, "manifest.csv"), encoding="utf-8", newline="") as f:
        total = sum(1 for _ in csv.DictReader(f))  # rows, not lines: some titles contain newlines
    size = os.path.getsize(os.path.join(args.out_dir, "corpus.txt"))
    print(f"\n{total} documents, {size / 1e6:.0f} MB -> {args.out_dir}/corpus.txt  "
          f"({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()

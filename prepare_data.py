"""
STAGE 0 — Download a public-domain corpus from Project Gutenberg.

    python prepare_data.py                  # Machado de Assis (Portuguese) — default
    python prepare_data.py --preset brasil  # Machado + other 19th-c. Brazilian classics (more data)
    python prepare_data.py --preset verne   # Jules Verne (French)
    python prepare_data.py --preset doyle   # Arthur Conan Doyle (English)

Writes data/corpus.txt — books separated by <|endoftext|>. This is run 1's corpus and,
re-encoded with the run-2 tokenizer, the fine-tuning set of run 2. The pretraining
library of run 2 is built by prepare_pretrain.py.
Uses the free Gutendex API (https://gutendex.com) to find the books.
"""
import argparse
import os
import re
import time
import unicodedata

import requests

EOT = "<|endoftext|>"
HEADERS = {"User-Agent": "tiny-llm-from-scratch (educational project)"}

PRESETS = {
    "machado": dict(lang="pt", authors=["Machado de Assis"],
                    # Known Gutenberg IDs, used only if the Gutendex API is unreachable
                    fallback_ids=[54829, 55682, 55797, 56737, 57001, 53101, 33056, 61653]),
    "brasil": dict(lang="pt", authors=["Machado de Assis", "Azevedo, Aluísio", "Alencar, José de",
                                       "Barreto, Lima", "Pompéia, Raul", "Almeida, Júlia Lopes de",
                                       "Taunay, Alfredo d'Escragnolle Taunay"]),
    "verne": dict(lang="fr", authors=["Verne, Jules"]),
    "doyle": dict(lang="en", authors=["Doyle, Arthur Conan"]),
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", " ", s.lower()).strip()


def find_books(author: str, lang: str) -> list:
    """Return [(id, title, txt_url)] for an author in a language, via Gutendex."""
    surname = norm(author.split(",")[0])
    url = "https://gutendex.com/books"
    params = {"search": author.replace(",", ""), "languages": lang}
    found = []
    while url:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        for b in data["results"]:
            if not any(surname in norm(a["name"]) for a in b["authors"]):
                continue
            if any(m.startswith("text/plain") for m in b["formats"]):
                # Gutenberg serves a UTF-8 plain-text version of every text book at this URL
                found.append((b["id"], b["title"],
                              f"https://www.gutenberg.org/ebooks/{b['id']}.txt.utf-8"))
        url, params = data.get("next"), None
    return found


def strip_gutenberg(raw: str) -> str:
    """Keep only the book: drop the license header/footer and tidy whitespace."""
    raw = raw.replace("\r\n", "\n")
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", raw)
    text = raw[start.end() if start else 0: end.start() if end else len(raw)]

    lines = text.split("\n")
    lines = [l for i, l in enumerate(lines)
             if not (i < 40 and re.match(r"\s*(Produced by|E-text prepared|Credits)", l))]
    lines = [l for l in lines if not re.fullmatch(r"\s*\[(Ilustra|Illustra|Image|Gravura).*\]\s*", l)]
    text = "\n".join(lines)

    # Gutenberg hard-wraps lines at ~70 chars: re-join lines inside a paragraph.
    paragraphs = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paragraphs = [re.sub(r"[ \t]+", " ", p).replace("_", "") for p in paragraphs if p]
    return "\n\n".join(paragraphs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="machado", choices=PRESETS)
    ap.add_argument("--max_books", type=int, default=200)
    ap.add_argument("--out", default="data/corpus.txt")
    args = ap.parse_args()
    cfg = PRESETS[args.preset]

    books = []
    try:
        for author in cfg["authors"]:
            hits = find_books(author, cfg["lang"])
            print(f"{author}: {len(hits)} books found")
            books += hits
    except requests.RequestException as e:
        if not cfg.get("fallback_ids"):
            raise
        print(f"Gutendex unreachable ({e}); using the built-in list of Gutenberg IDs.")
        books = [(i, f"#{i}", f"https://www.gutenberg.org/ebooks/{i}.txt.utf-8")
                 for i in cfg["fallback_ids"]]

    seen, texts = set(), []
    for book_id, title, url in books[: args.max_books]:
        key = norm(title)
        if key in seen:  # same title, different edition
            continue
        seen.add(key)
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            r.encoding = "utf-8"
        except requests.RequestException as e:
            print(f"  skip {book_id} ({e})")
            continue
        text = strip_gutenberg(r.text)
        if len(text) < 5000:
            continue
        texts.append(text)
        print(f"  [{len(texts):3d}] {title[:60]:60s} {len(text) / 1e3:7.0f}k chars")
        time.sleep(1.0)  # be polite to Gutenberg's servers

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    corpus = f"\n{EOT}\n".join(texts)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"\n{len(texts)} books, {len(corpus) / 1e6:.2f}M characters -> {args.out}")


if __name__ == "__main__":
    main()

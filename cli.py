"""
cli.py — interactive command-line interface for the patent search engine.

Run:  python cli.py

Type a natural-language query. Optionally add filters with flags at the end:
    airless tire --class B60C
    wheel --title wheel
    --exact "NON-PNEUMATIC TIRE"
    robot mobility --class B60B --alpha 0.7

Flags:
  --class PREFIX   only patents whose classification starts with PREFIX
  --title WORD     only patents whose title contains WORD
  --abstract WORD  only patents whose abstract contains WORD
  --exact "TITLE"  exact (case-insensitive) title match
  --alpha 0.0-1.0  semantic vs keyword weight (1=semantic, 0=keyword)
  --k N            number of results (default 10)

Type 'help' for this message, 'quit' to exit.
"""

import shlex
from loader import load_patents
from search import PatentSearchEngine


def parse_line(line: str) -> dict:
    """Split a raw input line into query text + filter kwargs."""
    tokens = shlex.split(line)
    query_parts, kwargs = [], {}
    i = 0
    flag_map = {
        "--class": "classification_prefix",
        "--title": "title_contains",
        "--abstract": "abstract_contains",
        "--exact": "exact_title",
        "--alpha": "alpha",
        "--k": "top_k",
    }
    while i < len(tokens):
        tok = tokens[i]
        if tok in flag_map:
            if i + 1 >= len(tokens):
                break
            value = tokens[i + 1]
            key = flag_map[tok]
            if key == "alpha":
                value = float(value)
            elif key == "top_k":
                value = int(value)
            kwargs[key] = value
            i += 2
        else:
            query_parts.append(tok)
            i += 1
    kwargs["query"] = " ".join(query_parts)
    return kwargs


def print_results(results: list[dict]):
    if not results:
        print("  No matches (filters may have excluded everything).\n")
        return
    for rank, r in enumerate(results, 1):
        flag = "  [!] claims may contain garble" if r["has_garbled_claims"] else ""
        score_str = f"[{r['score']:.3f}] " if r["score"] is not None else ""
        print(f"  {rank:>2}. {score_str}{r['title']}")
        print(f"      {r['doc_number']}  |  {r['classification']}{flag}")
        if r["snippet"]:
            print(f"      \"{r['snippet']}\"")
        print()


def main():
    print("Loading patents and building the search index...")
    patents = load_patents()

    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
        print("Loading semantic model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Loading cross-encoder re-ranker (ms-marco-MiniLM-L-6-v2)...")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        print(f"  Semantic model unavailable ({e}); running keyword-only.")
        model = None
        reranker = None

    engine = PatentSearchEngine(patents, embed_model=model, reranker=reranker)
    if reranker:
        mode = "two-phase (semantic retrieval + cross-encoder re-rank)"
    elif model:
        mode = "hybrid (keyword + semantic)"
    else:
        mode = "keyword-only"
    print(f"\nReady. {len(patents)} patents indexed. Mode: {mode}.")
    print("Type a query, 'help', or 'quit'.\n")

    while True:
        try:
            line = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        if line.lower() == "help":
            print(__doc__)
            continue

        kwargs = parse_line(line)
        if not kwargs.get("query") and not any(
            k in kwargs for k in ("classification_prefix", "title_contains",
                                  "abstract_contains", "exact_title")
        ):
            print("  Empty query.\n")
            continue

        results = engine.search(**kwargs)
        print()
        print_results(results)


if __name__ == "__main__":
    main()

"""
profile_breakdown.py — decompose where a single search spends its time.

This tells us WHY filtering didn't speed things up, using measured numbers
rather than assumptions. Run:  python profile_breakdown.py
"""

import time
import statistics
import numpy as np
from loader import load_patents
from search import PatentSearchEngine, tokenize


def median_ms(fn, runs=50):
    fn()  # warm-up
    ts = []
    for _ in range(runs):
        s = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - s) * 1000)
    return round(statistics.median(ts), 3)


def main():
    print("Loading + indexing...")
    patents = load_patents()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    engine = PatentSearchEngine(patents, embed_model=model)

    query = "airless non-pneumatic tire for heavy vehicles"
    qtok = tokenize(query)
    all_idx = list(range(len(patents)))

    # 1. cost of encoding the query alone (happens every search, any pool size)
    t_encode = median_ms(
        lambda: model.encode([query], normalize_embeddings=True,
                             show_progress_bar=False)
    )

    # 2. cost of the filter loop over all 640 patents
    t_filter = median_ms(
        lambda: engine._candidate_indices(classification_prefix="B60B")
    )

    # 3. cost of BM25 scoring over the full corpus
    t_bm25 = median_ms(lambda: engine.bm25.score(qtok, all_idx))

    # 4. cost of the embedding cosine over the full corpus (pre-computed docs)
    q_emb = model.encode([query], normalize_embeddings=True,
                         show_progress_bar=False)[0]
    t_cos = median_ms(lambda: engine.doc_embeddings @ q_emb)

    print("\nPER-SEARCH TIME BREAKDOWN (median over 50 runs)")
    print("-" * 50)
    print(f"  Encode the query (fixed cost, any pool): {t_encode:>7} ms")
    print(f"  Filter loop over all 640 patents:        {t_filter:>7} ms")
    print(f"  BM25 scoring over full corpus:           {t_bm25:>7} ms")
    print(f"  Embedding cosine over full corpus:       {t_cos:>7} ms")
    print("-" * 50)
    scoring = t_bm25 + t_cos
    print(f"  => Query encoding is ~{t_encode/scoring:.0f}x the cost of scoring"
          f" all 640 docs.")
    print(f"  => Filtering ADDS {t_filter} ms while saving only a fraction")
    print(f"     of {scoring:.3f} ms of scoring. That is why filtering")
    print(f"     did not speed up search at this corpus size.")


if __name__ == "__main__":
    main()

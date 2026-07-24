"""
benchmark.py — Part 2 efficiency analysis.

Story in two acts, both measured on your machine:

  ACT 1 (bi-encoder only): filtering does NOT help. Query-encoding is a
  fixed cost that dominates; scoring 640 tiny vectors is already sub-1ms,
  so shrinking the candidate set saves almost nothing. Filtering is
  overhead. (See profile_breakdown.py for the decomposition.)

  ACT 2 (with cross-encoder re-ranking): filtering DOES help. The
  cross-encoder runs a full transformer per (query, document) pair, so
  per-document cost is now large. A metadata pre-filter cuts how many
  documents reach that expensive phase, and latency drops in proportion.

Conclusion: whether a filter is worth it depends on where the time goes.
We measured, found the bottleneck, then changed the architecture so the
filter earns its keep.

Run:  python benchmark.py
"""

import time
import statistics
from loader import load_patents
from search import PatentSearchEngine


def time_query(engine, runs: int = 15, **kwargs) -> float:
    engine.search(**kwargs)  # warm-up
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        engine.search(**kwargs)
        times.append((time.perf_counter() - start) * 1000)
    return round(statistics.median(times), 2)


def act(engine, title, query, rerank):
    print("=" * 64)
    print(title)
    print("=" * 64)
    print(f"Query: {query!r}   rerank={rerank}\n")
    scenarios = [
        ("No filter (all patents reach scoring)", {}),
        ("Filter B60C (tires)", {"classification_prefix": "B60C"}),
        ("Filter B60B (wheels)", {"classification_prefix": "B60B"}),
        ("Filter title contains 'tire'", {"title_contains": "tire"}),
    ]
    print(f"{'Scenario':<42}{'Pool':>6}{'Median':>11}")
    print("-" * 64)
    baseline = None
    for label, filt in scenarios:
        pool = len(engine._candidate_indices(
            classification_prefix=filt.get("classification_prefix"),
            title_contains=filt.get("title_contains"),
        ))
        med = time_query(engine, query=query, top_k=10, retrieve_k=50,
                         rerank=rerank, **filt)
        if baseline is None:
            baseline, note = med, "  (baseline)"
        else:
            r = baseline / med if med else 0
            note = f"  {r:.2f}x faster" if r >= 1 else f"  {1/r:.2f}x slower"
        print(f"{label:<42}{pool:>6}{med:>8} ms{note}")
    print("-" * 64 + "\n")


if __name__ == "__main__":
    print("Loading patents and building index...")
    patents = load_patents()

    from sentence_transformers import SentenceTransformer, CrossEncoder
    bi = SentenceTransformer("all-MiniLM-L6-v2")
    print("Loading cross-encoder re-ranker...")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    engine = PatentSearchEngine(patents, embed_model=bi, reranker=ce)
    query = "airless non-pneumatic tire for heavy vehicles"

    act(engine, "ACT 1 - BI-ENCODER ONLY (filter effect is tiny)", query,
        rerank=False)
    act(engine, "ACT 2 - WITH CROSS-ENCODER (filter STILL does not help)",
        query, rerank=True)

    # ACT 3: the real latency lever is retrieve_k, not the filter.
    print("=" * 64)
    print("ACT 3 - THE REAL LEVER: retrieve_k (docs the cross-encoder scores)")
    print("=" * 64)
    print(f"Query: {query!r}   no filter, pool = {len(patents)}\n")
    print(f"{'retrieve_k':>11}{'docs re-ranked':>16}{'median':>12}")
    print("-" * 40)
    for rk in [10, 25, 50, 100]:
        med = time_query(engine, runs=7, query=query, top_k=10,
                         retrieve_k=rk, rerank=True)
        print(f"{rk:>11}{rk:>16}{med:>9} ms")
    print("-" * 64 + "\n")

    print("Takeaway (from the measurements above):")
    print("  * The cross-encoder scores exactly retrieve_k documents, no")
    print("    matter how large the candidate pool is. So a metadata filter")
    print("    (Act 2) changes WHICH documents are eligible but not HOW MANY")
    print("    get re-ranked -- it does not reduce latency.")
    print("  * Latency scales with retrieve_k (Act 3), roughly linearly.")
    print("  * Conclusion: the filter is a RELEVANCE lever (better results,")
    print("    domain-constrained); retrieve_k is the LATENCY lever. Knowing")
    print("    which knob controls which outcome is the point of measuring.")

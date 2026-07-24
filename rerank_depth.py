"""
rerank_depth.py — isolate the REAL latency lever in two-phase search.

Act 2 of benchmark.py showed that filtering the candidate pool (640 -> 318
-> 298) did NOT reduce cross-encoder latency. This script shows why: the
cross-encoder scores exactly `retrieve_k` documents regardless of pool
size, so retrieve_k -- not the filter -- controls phase-2 cost.

Run:  python rerank_depth.py
"""

import time
import statistics
from loader import load_patents
from search import PatentSearchEngine


def median_ms(engine, runs=7, **kwargs):
    engine.search(**kwargs)
    ts = []
    for _ in range(runs):
        s = time.perf_counter()
        engine.search(**kwargs)
        ts.append((time.perf_counter() - s) * 1000)
    return round(statistics.median(ts), 1)


def main():
    print("Loading + building index...")
    patents = load_patents()
    from sentence_transformers import SentenceTransformer, CrossEncoder
    bi = SentenceTransformer("all-MiniLM-L6-v2")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    engine = PatentSearchEngine(patents, embed_model=bi, reranker=ce)
    q = "airless non-pneumatic tire for heavy vehicles"

    print("\nLEVER 1 - vary retrieve_k, NO filter (pool = 640 every time)")
    print(f"{'retrieve_k':>11}{'docs re-ranked':>16}{'median':>12}")
    print("-" * 40)
    for rk in [10, 25, 50, 100, 200]:
        med = median_ms(engine, query=q, top_k=10, retrieve_k=rk, rerank=True)
        print(f"{rk:>11}{rk:>16}{med:>9} ms")

    print("\nLEVER 2 - vary filter, retrieve_k fixed at 50")
    print(f"{'filter':>22}{'pool':>7}{'median':>12}")
    print("-" * 42)
    for label, filt in [
        ("none", {}),
        ("B60C", {"classification_prefix": "B60C"}),
        ("B60B", {"classification_prefix": "B60B"}),
    ]:
        med = median_ms(engine, query=q, top_k=10, retrieve_k=50,
                        rerank=True, **filt)
        pool = len(engine._candidate_indices(
            classification_prefix=filt.get("classification_prefix")))
        print(f"{label:>22}{pool:>7}{med:>9} ms")

    print("\nConclusion: latency tracks retrieve_k (docs actually re-ranked),")
    print("not pool size. A metadata filter changes WHICH docs are eligible")
    print("(relevance/precision), but not how many the cross-encoder scores,")
    print("so it does not reduce phase-2 latency. The latency lever is")
    print("retrieve_k; the filter is a relevance lever.")


if __name__ == "__main__":
    main()

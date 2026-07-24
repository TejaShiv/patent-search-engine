# Patent Search Engine

A search engine over vehicle patent applications (2024–present), built for the
ThinkStruct take-home challenge. It supports natural-language queries, metadata
filters, and a two-phase retrieve-then-rerank pipeline.

Corpus: **640 patent applications** across 64 weekly JSON batches.

---

## Problem statement (what I chose to build)

**Part 1 — core search.** Given a natural-language query (e.g. *"airless tire
for heavy vehicles"*), return the most relevant patents, ranked, each with its
title, document number, classification code, a matching snippet, and a flag if
its claims text is malformed. The user can also filter by classification code,
title/abstract keyword, or look up an exact title.

**Part 2 — enhancements.** I implemented two of the offered options:
- **(b) Hybrid searching** — metadata filters (classification prefix, title/
  abstract keyword, exact title) combined with scored retrieval, plus a timing
  analysis of filters.
- **(c) Two-phase search** — fast semantic retrieval followed by cross-encoder
  re-ranking for precision.

The interesting result is in the efficiency analysis below: the filters do
**not** speed up search the way one might assume, and the code + benchmarks
show exactly why.

---

## How it works

### Retrieval scoring
Each patent is scored two ways and the scores are blended:
- **BM25** (implemented directly in `search.py`, no library) — keyword
  relevance from exact term overlap.
- **Semantic** — sentence-transformer bi-encoder (`all-MiniLM-L6-v2`) embeddings,
  cosine similarity. Catches meaning the keywords miss (a query for *airless*
  matches *non-pneumatic*).

`alpha` controls the blend (1.0 = pure semantic, 0.0 = pure keyword,
default 0.5). Scores are min-max normalized so the two are comparable.

### Two-phase re-ranking
When re-ranking is on (default):
1. **Phase 1 (fast):** the hybrid score keeps the top `retrieve_k` candidates.
2. **Phase 2 (precise):** a cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores
   those `retrieve_k` pairs and returns the top `k`.

A bi-encoder embeds query and document separately; a cross-encoder reads the
query and document *together*, which is more accurate but much slower — hence
running it only on a shortlist.

### Filters (hybrid search)
Filters run first and shrink the candidate pool: classification prefix
(e.g. `B60B` → wheels), title/abstract substring, or exact-title match.

---

## Data quality: how malformed fields are handled

All 640 patents have complete fields (100% coverage of title, abstract, claims,
description, classification, etc.). One real issue: **some claims strings are
garbled by XML parsing** — orphaned reference numerals produce fragments like
`( the winding mechanism ( the baffle ...` with unbalanced parentheses.

I measured it precisely. The genuine corruption (a `( the` duplication artifact,
or two-or-more unclosed parentheses) affects **35 of 640 patents**. Note that
single reference numerals like `( 1 )` are *normal* patent notation, not
damage, so I deliberately do **not** flag those.

**Decision: flag, never alter.** The loader tags each patent with
`has_garbled_claims` and surfaces it in results (`[!] claims may contain
garble`). I do not attempt to repair claim text — silently rewriting legal
claim language risks fabricating content that was never in the source, which is
the wrong failure mode for patent data. Malformed claims are still indexed and
searchable; the flag lets a user judge the result. Empty paragraphs in
`detailed_description` are dropped (pure noise, no information lost).

---

## Efficiency analysis (the main finding)

I asked the obvious question — *do the metadata filters make search faster?* —
and measured it. The answer is **no**, and the reason is worth stating.

**Bi-encoder only (`rerank=False`):** a full search over all 640 patents takes
around 12 ms. A filter that cuts the pool to ~300 changes this by ~1 ms —
negligible, because vectorized scoring of 640 small vectors is already
trivially cheap.
`profile_breakdown.py` shows the actual bottleneck is **encoding the query
into an embedding** (~13 ms fixed cost, paid on every search regardless of
pool size), not scoring the documents.

**With cross-encoder (`rerank=True`):** re-ranking dominates — hundreds of ms.
But filtering the pool *still* does not reduce it. Why: the cross-encoder scores
exactly `retrieve_k` documents (default 50), no matter how big the candidate
pool is. Filtering changes *which* documents are eligible, not *how many* get
re-ranked. `rerank_depth.py` isolates the two levers:

- Varying **`retrieve_k`** (no filter, pool fixed at 640): latency scales
  roughly linearly — about 18 ms per document re-ranked (10 docs ≈ 180 ms,
  25 ≈ 455 ms, 50 ≈ 920 ms, 100 ≈ 1740 ms).
- Varying the **filter** (`retrieve_k` fixed at 50): latency stays essentially
  flat across pool sizes of 640 / 318 / 298. Small run-to-run wobble (roughly
  ±5%) is measurement noise, not a real effect — the cross-encoder still scores
  the same 50 documents in every case.

**Conclusion:** the metadata filter is a **relevance lever** (it constrains
results to a domain, e.g. only wheel patents), and `retrieve_k` is the
**latency lever**. They control different outcomes. The naive optimization
("filter to go faster") does not work in this architecture, and the benchmarks
show precisely why. At a much larger corpus, or if phase-1 retrieval itself
became the bottleneck, filtering *would* start to matter — which is the point
of measuring rather than assuming.

*(All numbers above are representative of runs on an M-series MacBook Air;
re-run `benchmark.py`, `profile_breakdown.py`, and `rerank_depth.py` to
reproduce on your machine.)*

---

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

First run downloads two models (~90 MB bi-encoder, ~90 MB cross-encoder).

```bash
python loader.py             # sanity check: loads 640 patents, reports flags
python cli.py                # interactive search
python benchmark.py          # filter timing analysis (Acts 1–3)
python profile_breakdown.py  # where a single search spends its time
python rerank_depth.py       # retrieve_k vs filter: the two levers
```

### Using the CLI
Type a query, optionally with filters:

```
search> airless tire --class B60C
search> wheel --class B60B --k 5
search> robot mobility --alpha 0.7
search> --exact "NON-PNEUMATIC TIRE"
```

Flags: `--class PREFIX`, `--title WORD`, `--abstract WORD`,
`--exact "TITLE"`, `--alpha 0.0–1.0`, `--k N`. Type `help` or `quit`.

---

## Files

| File | Purpose |
|------|---------|
| `loader.py` | Reads the 640 patents; flags garbled claims; drops empty paragraphs. Does not modify source data. |
| `search.py` | The engine: BM25 + semantic hybrid scoring, metadata filters, two-phase cross-encoder re-ranking. |
| `cli.py` | Interactive command-line interface. |
| `benchmark.py` | Filter timing analysis (Acts 1–3). |
| `profile_breakdown.py` | Decomposes single-search latency. |
| `rerank_depth.py` | Isolates `retrieve_k` vs filter as the latency lever. |
| `data/patent_data_small/` | The 64 JSON batches (committed so the repo runs out of the box). |

---

## Design choices, briefly

- **BM25 written by hand**, not imported — it is ~40 lines and I wanted every
  scoring decision to be explainable.
- **Flag-don't-fix** on malformed claims — conservative is correct for legal text.
- **Models injected** into the engine constructor — so the search logic is
  testable with lightweight stubs, without loading heavy models.
- **CLI-first** — the challenge allows a terminal interface; I kept it simple
  rather than gold-plating a web UI.
- **Measured before concluding** — the efficiency section reports what the
  benchmarks actually show, including the result that filtering does not speed
  up re-ranking.

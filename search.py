"""
search.py — the patent search engine.

Two scoring methods, combined into one hybrid score:
  1. BM25  — classic keyword relevance (exact word overlap).
  2. Semantic — sentence-transformer embeddings (meaning overlap;
     finds "airless tire" when the patent says "non-pneumatic wheel").

Plus metadata FILTERS (Part 2, "hybrid searching"):
  - classification prefix (e.g. "B60B" -> wheels)
  - keyword-in-title / keyword-in-abstract
  - exact-title lookup
Filters run FIRST to shrink the candidate pool, then we score only the
survivors. That is the efficiency idea we benchmark in benchmark.py.
"""

import re
import math
import numpy as np


# ----------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def patent_text(patent: dict) -> str:
    """The blob of text we search over for one patent."""
    parts = [
        patent.get("title", ""),
        patent.get("abstract", ""),
        " ".join(patent.get("claims", [])),
        " ".join(patent.get("detailed_description", [])),
    ]
    return " ".join(parts)


# ----------------------------------------------------------------------
# BM25 keyword scorer (implemented directly — no external dep, fully
# explainable line by line in an interview)
# ----------------------------------------------------------------------

class BM25:
    """Standard Okapi BM25. Scores documents by keyword relevance."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0

        # document frequency: how many docs contain each term
        df = {}
        for doc in corpus_tokens:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        # inverse document frequency
        self.idf = {
            term: math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

        # term frequency per document
        self.tf = []
        for doc in corpus_tokens:
            counts = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            self.tf.append(counts)

    def score(self, query_tokens: list[str], doc_indices: list[int]) -> np.ndarray:
        """Score a specific subset of documents against the query."""
        scores = np.zeros(len(doc_indices))
        for pos, idx in enumerate(doc_indices):
            counts = self.tf[idx]
            dl = self.doc_len[idx]
            s = 0.0
            for term in query_tokens:
                if term not in counts:
                    continue
                freq = counts[term]
                idf = self.idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * (freq * (self.k1 + 1)) / denom
            scores[pos] = s
        return scores


# ----------------------------------------------------------------------
# The engine
# ----------------------------------------------------------------------

def _normalize(x: np.ndarray) -> np.ndarray:
    """Min-max to [0,1] so keyword and semantic scores are comparable."""
    if len(x) == 0:
        return x
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class PatentSearchEngine:
    def __init__(self, patents: list[dict], embed_model=None, reranker=None):
        """
        patents:     list from loader.load_patents()
        embed_model: a sentence-transformers bi-encoder, or None for keyword-only.
        reranker:    a sentence-transformers CrossEncoder, or None to skip
                     re-ranking. When present, search runs two phases:
                     fast bi-encoder retrieval, then precise cross-encoder
                     re-ranking of the top candidates.
                     (All injectable so logic is testable without heavy models.)
        """
        self.patents = patents
        self.embed_model = embed_model
        self.reranker = reranker

        # Pre-tokenize corpus once for BM25.
        self.corpus_tokens = [tokenize(patent_text(p)) for p in patents]
        self.bm25 = BM25(self.corpus_tokens)

        # Pre-compute embeddings once (the expensive step, done at startup).
        self.doc_embeddings = None
        if embed_model is not None:
            texts = [
                (p.get("title", "") + ". " + p.get("abstract", ""))
                for p in patents
            ]
            self.doc_embeddings = embed_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )

    # ---- filters (Part 2) --------------------------------------------

    def _candidate_indices(
        self,
        classification_prefix: str = None,
        title_contains: str = None,
        abstract_contains: str = None,
        exact_title: str = None,
    ) -> list[int]:
        """Return indices of patents that survive the metadata filters."""
        indices = range(len(self.patents))
        result = []
        for i in indices:
            p = self.patents[i]
            if classification_prefix:
                if not p.get("classification", "").upper().startswith(
                    classification_prefix.upper()
                ):
                    continue
            if title_contains:
                if title_contains.lower() not in p.get("title", "").lower():
                    continue
            if abstract_contains:
                if abstract_contains.lower() not in p.get("abstract", "").lower():
                    continue
            if exact_title:
                if p.get("title", "").strip().lower() != exact_title.strip().lower():
                    continue
            result.append(i)
        return result

    # ---- search ------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        classification_prefix: str = None,
        title_contains: str = None,
        abstract_contains: str = None,
        exact_title: str = None,
        retrieve_k: int = 50,
        rerank: bool = True,
    ) -> list[dict]:
        """
        alpha = weight on semantic vs keyword.
          alpha=1.0 -> pure semantic, 0.0 -> pure keyword, 0.5 -> even blend.
        Filters (if any) shrink the candidate pool BEFORE scoring.

        Two-phase when a reranker is present and rerank=True:
          phase 1 (fast) keeps the top `retrieve_k` by hybrid bi-encoder score;
          phase 2 (precise) cross-encoder re-scores those, returns top_k.
        """
        candidates = self._candidate_indices(
            classification_prefix, title_contains, abstract_contains, exact_title
        )
        if not candidates:
            return []

        query_tokens = tokenize(query)

        # Filter-only lookup (no query text, e.g. exact-title): nothing to
        # rank, so just return the filtered patents in dataset order.
        if not query_tokens:
            results = []
            for idx in candidates[:top_k]:
                p = self.patents[idx]
                results.append({
                    "doc_number": p.get("doc_number"),
                    "title": p.get("title"),
                    "classification": p.get("classification"),
                    "score": None,
                    "snippet": (p.get("abstract", "")[:200]),
                    "has_garbled_claims": p.get("has_garbled_claims", False),
                })
            return results

        kw = _normalize(self.bm25.score(query_tokens, candidates))

        if self.embed_model is not None and self.doc_embeddings is not None:
            q_emb = self.embed_model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )[0]
            cand_emb = self.doc_embeddings[candidates]
            sem = cand_emb @ q_emb  # cosine (vectors already normalized)
            sem = _normalize(np.asarray(sem))
            combined = alpha * sem + (1 - alpha) * kw
        else:
            combined = kw  # keyword-only fallback

        order = np.argsort(combined)[::-1]

        use_rerank = (
            rerank and self.reranker is not None and len(order) > 0
        )
        if use_rerank:
            # Phase 1: keep the top `retrieve_k` candidates by hybrid score.
            phase1 = order[:retrieve_k]
            pairs = [
                [query, patent_text(self.patents[candidates[pos]])[:2000]]
                for pos in phase1
            ]
            ce_scores = self.reranker.predict(pairs, show_progress_bar=False)
            # Phase 2: order those by cross-encoder score, take top_k.
            rerank_order = np.argsort(ce_scores)[::-1][:top_k]
            final = [(phase1[i], float(ce_scores[i])) for i in rerank_order]
        else:
            final = [(pos, float(combined[pos])) for pos in order[:top_k]]

        results = []
        for pos, score in final:
            idx = candidates[pos]
            p = self.patents[idx]
            results.append({
                "doc_number": p.get("doc_number"),
                "title": p.get("title"),
                "classification": p.get("classification"),
                "score": round(score, 4),
                "snippet": self._snippet(p, query_tokens),
                "has_garbled_claims": p.get("has_garbled_claims", False),
            })
        return results

    def _snippet(self, patent: dict, query_tokens: set) -> str:
        """Return the abstract sentence with the most query-term overlap."""
        abstract = patent.get("abstract", "")
        sentences = re.split(r"(?<=[.!?])\s+", abstract)
        qset = set(query_tokens)
        best, best_overlap = "", -1
        for sent in sentences:
            overlap = len(qset & set(tokenize(sent)))
            if overlap > best_overlap:
                best, best_overlap = sent, overlap
        return (best[:200] + "...") if len(best) > 200 else best

"""Retrievers and the on-disk index bundle that the API serves.

Kinds
  bm25       sparse lexical baseline (bm25s + Snowball stemming)
  lsa        dense vectors *trained on this corpus*: TF-IDF -> truncated SVD (no external weights)
  fastembed  pretrained ONNX sentence embeddings (BAAI/bge-small-en-v1.5 by default)
  hybrid     reciprocal-rank fusion of member retrievers

A bundle is a folder: manifest.json + chunks.jsonl + one sub-folder per retriever.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from .text import normalize_ws

Hit = tuple[int, float]  # (chunk index, score)


# --------------------------------------------------------------------------- base
class Retriever:
    kind = "base"

    def __init__(self, name: str, config: dict):
        self.name, self.config = name, dict(config)

    def fit(self, texts: list[str]) -> Retriever:
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[Hit]:
        return self.search_many([query], k)[0]

    def search_many(self, queries: list[str], k: int) -> list[list[Hit]]:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, name: str, config: dict, path: Path) -> Retriever:
        raise NotImplementedError


def _topk_dense(scores: np.ndarray, k: int) -> list[list[Hit]]:
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    out = []
    for row, cand in zip(scores, idx):
        order = cand[np.argsort(-row[cand])]
        out.append([(int(i), float(row[i])) for i in order])
    return out


# --------------------------------------------------------------------------- BM25
class BM25Retriever(Retriever):
    kind = "bm25"

    def _tokenize(self, texts):
        import bm25s
        import Stemmer
        return bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"), show_progress=False)

    def fit(self, texts):
        import bm25s
        self.model = bm25s.BM25(k1=self.config.get("k1", 1.2), b=self.config.get("b", 0.75))
        self.model.index(self._tokenize(texts), show_progress=False)
        self.n_docs = len(texts)
        return self

    def search_many(self, queries, k):
        tokens = self._tokenize(queries)
        k = min(k, self.n_docs)
        docs, scores = self.model.retrieve(tokens, k=k, show_progress=False, n_threads=1)
        return [[(int(d), float(s)) for d, s in zip(dr, sr) if s > 0] for dr, sr in zip(docs, scores)]

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path / "bm25"))
        (path / "n_docs.txt").write_text(str(self.n_docs))

    @classmethod
    def load(cls, name, config, path):
        import bm25s
        r = cls(name, config)
        r.model = bm25s.BM25.load(str(path / "bm25"))
        r.n_docs = int((path / "n_docs.txt").read_text())
        return r


# --------------------------------------------------------------------------- LSA (trained here)
class LSARetriever(Retriever):
    kind = "lsa"

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer
        self.pipe = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=self.config.get("min_df", 2),
                            max_features=self.config.get("max_features", 60000), stop_words="english",
                            dtype=np.float32),
            TruncatedSVD(n_components=self.config.get("dim", 256), random_state=0),
            Normalizer(copy=False),
        )
        self.emb = self.pipe.fit_transform(texts).astype(np.float16)
        self.explained_variance = float(self.pipe[1].explained_variance_ratio_.sum())
        return self

    def encode(self, queries):
        return self.pipe.transform(queries).astype(np.float32)

    def search_many(self, queries, k):
        return _topk_dense(self.encode(queries) @ self.emb.T.astype(np.float32), k)

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "pipeline.pkl", "wb") as f:
            pickle.dump(self.pipe, f)
        np.save(path / "emb.npy", self.emb)

    @classmethod
    def load(cls, name, config, path):
        r = cls(name, config)
        with open(path / "pipeline.pkl", "rb") as f:
            r.pipe = pickle.load(f)  # noqa: S301 - produced by our own pipeline
        r.emb = np.load(path / "emb.npy")
        return r


# --------------------------------------------------------------------------- pretrained ONNX embeddings
class FastEmbedRetriever(Retriever):
    kind = "fastembed"

    def _model(self):
        if getattr(self, "_m", None) is None:
            from fastembed import TextEmbedding
            self._m = TextEmbedding(self.config.get("model", "BAAI/bge-small-en-v1.5"), threads=self.config.get("threads"))
        return self._m

    def fit(self, texts):
        """Embed every passage. Results are cached on disk by (model, corpus) hash, because
        embedding ~29k passages on a CPU runner is the slowest step of the whole pipeline."""
        import hashlib
        import os
        model = self.config.get("model", "BAAI/bge-small-en-v1.5")
        h = hashlib.sha1(model.encode())
        for t in texts:
            h.update(t.encode("utf-8"))
        cache_dir = Path(os.environ.get("MEDRAG_EMB_CACHE", Path.home() / ".cache" / "medrag-embeddings"))
        cache = cache_dir / f"{h.hexdigest()[:20]}.npy"
        if cache.exists():
            self.emb = np.load(cache)
            return self
        parallel = 0 if len(texts) > 2000 else None  # 0 = one worker process per CPU core
        vecs = np.array(list(self._model().passage_embed(texts, batch_size=64, parallel=parallel)), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        self.emb = vecs.astype(np.float16)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache, self.emb)
        return self

    def encode(self, queries):
        v = np.array(list(self._model().query_embed(queries)), dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def search_many(self, queries, k):
        return _topk_dense(self.encode(queries) @ self.emb.T.astype(np.float32), k)

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "emb.npy", self.emb)

    @classmethod
    def load(cls, name, config, path):
        r = cls(name, config)
        r.emb = np.load(path / "emb.npy")
        return r


# --------------------------------------------------------------------------- hybrid
class HybridRetriever(Retriever):
    """Reciprocal-rank fusion: score = sum over members of 1 / (rrf_k + rank)."""
    kind = "hybrid"

    def __init__(self, name, config, members: list[Retriever] | None = None):
        super().__init__(name, config)
        self.members = members or []

    def search_many(self, queries, k):
        depth = self.config.get("candidate_depth", 50)
        rrf_k = self.config.get("rrf_k", 60)
        fused = [dict() for _ in queries]
        for m in self.members:
            for acc, hits in zip(fused, m.search_many(queries, depth)):
                for rank, (i, _) in enumerate(hits):
                    acc[i] = acc.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        return [sorted(acc.items(), key=lambda kv: -kv[1])[:k] for acc in fused]


KINDS = {c.kind: c for c in (BM25Retriever, LSARetriever, FastEmbedRetriever)}


def build_retrievers(candidates: list[dict], texts: list[str], depth: int, log=print) -> dict[str, Retriever]:
    """Fit every non-hybrid candidate once, then wire hybrids to the fitted members."""
    built: dict[str, Retriever] = {}
    for cand in candidates:
        if cand["kind"] == "hybrid":
            continue
        t0 = time.time()
        built[cand["name"]] = KINDS[cand["kind"]](cand["name"], cand).fit(texts)
        built[cand["name"]].fit_seconds = time.time() - t0
        log(f"fitted {cand['name']} in {built[cand['name']].fit_seconds:.1f}s")
    for cand in candidates:
        if cand["kind"] == "hybrid":
            cfg = {**cand, "candidate_depth": depth}
            h = HybridRetriever(cand["name"], cfg, [built[m] for m in cand["members"]])
            h.fit_seconds = sum(built[m].fit_seconds for m in cand["members"])
            built[cand["name"]] = h
    return built


# --------------------------------------------------------------------------- bundle
CHUNK_FIELDS = ("chunk_id", "pair_id", "source", "url", "focus", "qtype", "question", "body")


class Bundle:
    """Everything the API needs to answer: chunk store + one (possibly hybrid) retriever."""

    def __init__(self, manifest: dict, chunks: list[dict], retriever: Retriever):
        self.manifest, self.chunks, self.retriever = manifest, chunks, retriever

    @property
    def version(self) -> str:
        return str(self.manifest.get("model_version") or self.manifest.get("run_id", "local"))[:40]

    def search(self, query: str, k: int) -> list[dict]:
        out = []
        for rank, (i, score) in enumerate(self.retriever.search(normalize_ws(query), k)):
            c = self.chunks[i]
            out.append({"rank": rank + 1, "score": round(score, 6), **c})
        return out

    # -------- persistence
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "chunks.jsonl", "w") as f:
            for c in self.chunks:
                f.write(json.dumps({k: c[k] for k in CHUNK_FIELDS}, ensure_ascii=False) + "\n")
        members = self.retriever.members if isinstance(self.retriever, HybridRetriever) else [self.retriever]
        for m in members:
            m.save(path / "retrievers" / m.name)
        manifest = {**self.manifest, "retriever": {
            "name": self.retriever.name, "kind": self.retriever.kind, "config": self.retriever.config,
            "members": [{"name": m.name, "kind": m.kind, "config": m.config} for m in members]}}
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, path: Path) -> Bundle:
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        with open(path / "chunks.jsonl") as f:
            chunks = [json.loads(line) for line in f]
        spec = manifest["retriever"]
        members = [KINDS[m["kind"]].load(m["name"], m["config"], path / "retrievers" / m["name"])
                   for m in spec["members"]]
        retriever = (HybridRetriever(spec["name"], spec["config"], members)
                     if spec["kind"] == "hybrid" else members[0])
        return cls(manifest, chunks, retriever)

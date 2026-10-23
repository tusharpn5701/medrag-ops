import math

import pytest

from medrag.evaluate import dedupe_pairs, evaluate, hit_at, mrr, ndcg_at, recall_at
from medrag.generate import Generator, Prompt, cited_indices, extractive_answer
from medrag.retrieval import BM25Retriever, Bundle, HybridRetriever, LSARetriever
from medrag.text import chunk_words, redact, sentences


def test_chunk_words_overlap():
    words = " ".join(str(i) for i in range(10))
    chunks = chunk_words(words, max_words=4, overlap=2)
    assert chunks[0] == "0 1 2 3" and chunks[1] == "2 3 4 5" and chunks[-1].endswith("9")


def test_redact_removes_contacts():
    out = redact("email me at jane.doe@mail.com or call +91 98765 43210 about my rash")
    assert "@" not in out and "98765" not in out and "rash" in out


def test_sentences_split_bullets():
    assert sentences("Symptoms include - Fever - Cough. See a doctor.") == ["Symptoms include", "Fever", "Cough.", "See a doctor."]


def test_metrics_known_values():
    qrels = {"a": 3, "b": 1}
    assert ndcg_at(["a", "b"], qrels, 10) == pytest.approx(1.0)
    assert ndcg_at(["b", "a"], qrels, 10) < 1.0
    assert recall_at(["x", "a"], qrels, 5, thr=2) == 1.0
    assert hit_at(["x", "y"], qrels, 2, thr=1) == 0.0
    assert mrr(["x", "b", "a"], qrels, thr=1) == pytest.approx(0.5)
    assert dedupe_pairs([(0, 1.0), (1, 0.9), (2, 0.8)], ["p", "p", "q"]) == ["p", "q"]
    assert math.isclose(ndcg_at([], qrels, 10), 0.0)


@pytest.mark.parametrize("cls,cfg", [(BM25Retriever, {"k1": 1.2, "b": 0.75}), (LSARetriever, {"dim": 4, "min_df": 1})])
def test_retrievers_rank_relevant_first(chunks, cls, cfg):
    r = cls("r", cfg).fit([c["text"] for c in chunks])
    top = r.search("how do you treat glaucoma eye pressure", 3)
    assert chunks[top[0][0]]["focus"] == "Glaucoma"


def test_hybrid_rrf_and_eval(chunks):
    texts = [c["text"] for c in chunks]
    h = HybridRetriever("h", {"rrf_k": 60, "candidate_depth": 5},
                        [BM25Retriever("b", {}).fit(texts), LSARetriever("l", {"dim": 4, "min_df": 1}).fit(texts)])
    items = [{"qid": "1", "query": "is gaucher disease inherited", "qrels": {"GARD_1_1": 3}}]
    agg, rows = evaluate(h, items, [c["pair_id"] for c in chunks], "t")
    assert agg["t/hit@5"] == 1.0 and rows[0]["top_pairs"][0].startswith("GARD_1")


def test_bundle_roundtrip(bundle_dir):
    b = Bundle.load(bundle_dir)
    hits = b.search("kidney stones prevention water", 2)
    assert hits[0]["pair_id"] == "NIDDK_3_2" and b.version == "medrag-retriever/v1"


def test_extractive_answer_cites(chunks):
    hits = [{**c, "rank": i + 1, "score": 1.0} for i, c in enumerate(chunks[3:5])]
    a = extractive_answer("what causes kidney stones", hits)
    assert a.mode == "extractive" and a.citations and "[1]" in a.text
    assert cited_indices("x [1] y [7] z [2]", 3) == [1, 2]


class _Boom:
    class chat:  # noqa: N801
        class completions:  # noqa: N801
            @staticmethod
            def create(**kw):
                raise TimeoutError("provider down")


class _Echo:
    class chat:  # noqa: N801
        class completions:  # noqa: N801
            @staticmethod
            def create(**kw):
                class R:
                    usage = None
                    choices = [type("C", (), {"message": type("M", (), {"content": "Drink water [2]."})()})()]
                return R()


def test_generator_llm_and_fallback(chunks, tmp_path):
    from medrag.config import ROOT
    prompt = Prompt.load(ROOT / "prompts" / "answer_v1.yaml")
    hits = [{**c, "rank": i + 1, "score": 1.0} for i, c in enumerate(chunks[3:5])]
    g = Generator(prompt, None, "http://x", "m")
    assert g.mode == "extractive"
    g.client = _Echo()
    a = g.answer("how to prevent kidney stones", hits)
    assert a.mode == "llm" and a.citations == [2]
    g.client = _Boom()
    a = g.answer("how to prevent kidney stones", hits)
    assert a.mode == "extractive" and "TimeoutError" in a.error

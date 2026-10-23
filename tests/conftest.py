import json
import os

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

CHUNKS = [
    ("GARD_1_1", "Gaucher disease", "inheritance",
     "Gaucher disease is inherited in an autosomal recessive pattern. Both copies of the GBA gene carry mutations."),
    ("GARD_1_2", "Gaucher disease", "symptoms",
     "Signs of Gaucher disease include an enlarged liver and spleen, anemia, bone pain and easy bruising."),
    ("NEI_2_1", "Glaucoma", "treatment",
     "Glaucoma is treated with prescription eye drops, laser trabeculoplasty or conventional surgery to lower eye pressure."),
    ("NIDDK_3_1", "Kidney stones", "causes",
     "Kidney stones form when urine contains more crystal-forming substances such as calcium and oxalate than the fluid can dilute."),
    ("NIDDK_3_2", "Kidney stones", "prevention",
     "Drinking enough water each day is the best way to help prevent most types of kidney stones."),
    ("NHLBI_4_1", "High blood pressure", "treatment",
     "Lifestyle changes such as the DASH eating plan, less salt and regular physical activity help lower high blood pressure."),
]


def make_chunks():
    out = []
    for i, (pid, focus, qtype, body) in enumerate(CHUNKS):
        out.append({"chunk_id": f"{pid}#0", "pair_id": pid, "source": pid.split("_")[0], "url": f"https://example.org/{i}",
                    "focus": focus, "qtype": qtype, "question": f"What about {focus} {qtype}?",
                    "text": f"{focus} | {qtype}. {body}", "body": body})
    return out


@pytest.fixture
def chunks():
    return make_chunks()


@pytest.fixture
def bundle_dir(tmp_path, chunks):
    from medrag.retrieval import BM25Retriever, Bundle
    r = BM25Retriever("bm25", {"kind": "bm25", "k1": 1.2, "b": 0.75}).fit([c["text"] for c in chunks])
    manifest = {"name": "bm25", "run_id": "test", "corpus_sha1": "abc", "model_version": "medrag-retriever/v1",
                "monitoring": {"low_confidence_threshold": 0.5}}
    d = tmp_path / "bundle"
    Bundle(manifest, chunks, r).save(d)
    with open(d / "reference_queries.jsonl", "w") as f:
        for q in ["What are the symptoms of Gaucher disease ?", "How to prevent Kidney stones ?",
                  "What are the treatments for Glaucoma ?"] * 20:
            f.write(json.dumps({"query": q, "set": "medquad", "top_score": 2.0}) + "\n")
    return d

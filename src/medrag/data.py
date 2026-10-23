"""Download MedQuAD and the TREC 2017 LiveQA-Med test collection, then build the corpus and eval sets.

Outputs (data/processed/):
  chunks.jsonl           retrieval units: one or more chunks per MedQuAD answer
  eval_medquad.jsonl     templated MedQuAD questions with their source answer(s) as gold
  eval_liveqa_val.jsonl  real consumer questions with graded NIH judgements (tuning split)
  eval_liveqa_test.jsonl same, held out for champion/challenger promotion
  stats.json             counts, written as a DVC metric
"""
from __future__ import annotations

import csv
import io
import json
import random
import re
import shutil
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

from .config import DATA_PROCESSED, DATA_RAW
from .text import chunk_words, norm_key, normalize_ws, sha1

MEDQUAD_GIT = "https://github.com/abachaa/MedQuAD"
MEDQUAD_ZIP = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
LIVEQA_TEST_XML = ("https://raw.githubusercontent.com/abachaa/LiveQA_MedicalTask_TREC2017/master/"
                   "TestDataset/TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml")

SOURCE_DIRS = {
    "CancerGov": "1_CancerGov_QA", "GARD": "2_GARD_QA", "GHR": "3_GHR_QA",
    "MPlusHealthTopics": "4_MPlus_Health_Topics_QA", "NIDDK": "5_NIDDK_QA", "NINDS": "6_NINDS_QA",
    "NIHSeniorHealth": "7_SeniorHealth_QA", "NHLBI": "8_NHLBI_QA_XML", "CDC": "9_CDC_QA",
}
# LiveQA judgement labels -> graded relevance gain
GAIN = {"1-Incorrect": 0, "2-Related": 1, "3-Incomplete": 2, "4-Excellent": 3}


# --------------------------------------------------------------------------- download
def download(raw_dir: Path = DATA_RAW) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    mq = raw_dir / "MedQuAD"
    if not (mq / "readme.txt").exists():
        shutil.rmtree(mq, ignore_errors=True)
        try:
            subprocess.run(["git", "clone", "--depth", "1", MEDQUAD_GIT, str(mq)], check=True)
            shutil.rmtree(mq / ".git", ignore_errors=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            with urllib.request.urlopen(MEDQUAD_ZIP, timeout=120) as r:
                zipfile.ZipFile(io.BytesIO(r.read())).extractall(raw_dir)
            (raw_dir / "MedQuAD-master").rename(mq)
    qrels_dir = raw_dir / "liveqa"
    if not (qrels_dir / "qrels.txt").exists():
        qrels_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(mq / "QA-TestSet-LiveQA-Med-Qrels-2479-Answers.zip") as z:
            for name in z.namelist():
                if name.startswith("__MACOSX") or name.endswith("/"):
                    continue
                target = "qrels.txt" if name.endswith(".txt") else "judged_answers.csv"
                (qrels_dir / target).write_bytes(z.read(name))
    if not (qrels_dir / "test_questions.xml").exists():
        with urllib.request.urlopen(LIVEQA_TEST_XML, timeout=120) as r:
            (qrels_dir / "test_questions.xml").write_bytes(r.read())


# --------------------------------------------------------------------------- parsing
def parse_medquad(raw_dir: Path, sources: list[str]) -> list[dict]:
    """One record per question-answer pair that still has its answer text."""
    pairs = []
    for src in sources:
        for f in sorted((raw_dir / "MedQuAD" / SOURCE_DIRS[src]).glob("*.xml")):
            root = ET.parse(f).getroot()
            focus = normalize_ws(root.findtext("Focus") or "")
            for qa in root.iter("QAPair"):
                answer = normalize_ws(qa.findtext("Answer") or "")
                q = qa.find("Question")
                if not answer or q is None:
                    continue
                pairs.append({
                    "pair_id": f"{src}_{f.stem}_{qa.get('pid')}",
                    "source": src, "doc_key": f.stem, "url": root.get("url", ""),
                    "focus": focus, "qtype": q.get("qtype", ""),
                    "question": normalize_ws(q.text or ""), "answer": answer,
                })
    return pairs


def make_chunks(pairs: list[dict], max_words: int, overlap: int, header: str) -> list[dict]:
    chunks = []
    for p in pairs:
        head = f"{p['focus']} | {p['qtype']}" if header == "focus_qtype" else ""
        for i, body in enumerate(chunk_words(p["answer"], max_words, overlap)):
            chunks.append({
                "chunk_id": f"{p['pair_id']}#{i}", "pair_id": p["pair_id"], "source": p["source"],
                "url": p["url"], "focus": p["focus"], "qtype": p["qtype"], "question": p["question"],
                "text": f"{head}. {body}" if head else body, "body": body,
            })
    return chunks


def medquad_eval(pairs: list[dict], n: int, seed: int) -> list[dict]:
    """Known-item questions. Every pair asking the same question counts as relevant."""
    by_q = defaultdict(list)
    for p in pairs:
        by_q[norm_key(p["question"])].append(p["pair_id"])
    rng = random.Random(seed)
    picked = rng.sample(pairs, min(n, len(pairs)))
    return [{"qid": f"mq-{p['pair_id']}", "query": p["question"],
             "qrels": {pid: 1 for pid in by_q[norm_key(p["question"])]}} for p in picked]


def _map_answer_ids(raw_dir: Path, pairs: list[dict]) -> dict[str, str]:
    """Map LiveQA judged answer ids (e.g. GHR_0000123_Sec2.txt) onto our pair ids.

    The 'SecN' number does not always equal the XML pid, so match on question text
    within the same document and fall back to the answer text.
    """
    csv.field_size_limit(10**9)
    with open(raw_dir / "liveqa" / "judged_answers.csv", encoding="utf-8", errors="replace") as f:
        judged = {row[0]: row[1] for row in csv.reader(f)}
    by_doc = defaultdict(list)
    for p in pairs:
        by_doc[(p["source"], p["doc_key"])].append(p)
    mapping = {}
    for aid, text in judged.items():
        m = re.match(r"([A-Za-z]+)_(.+)_Sec\d+\.txt$", aid)
        if not m or m.group(1) not in SOURCE_DIRS:
            continue
        cands = by_doc.get((m.group(1), m.group(2)), [])
        q = re.search(r"Question:\s*(.*?)(?:\(Also called.*)?\n", text)
        a = re.search(r"Answer:\s*(.*)", text, re.S)
        qk = norm_key(q.group(1)) if q else ""
        ak = norm_key(a.group(1))[:200] if a else ""
        hits = [p for p in cands if norm_key(p["question"]) == qk] or \
               [p for p in cands if ak and norm_key(p["answer"])[:200] == ak]
        if len(hits) > 1 and ak:
            hits = [p for p in hits if norm_key(p["answer"])[:200] == ak] or hits
        if hits:
            mapping[aid] = hits[0]["pair_id"]
    return mapping


def liveqa_eval(raw_dir: Path, pairs: list[dict], val_fraction: float, seed: int):
    root = ET.parse(raw_dir / "liveqa" / "test_questions.xml").getroot()
    questions = {}
    for q in root.iter("NLM-QUESTION"):
        num = q.get("qid", "").replace("TQ", "")
        orig = q.find("Original-Question")
        subject = normalize_ws(orig.findtext("SUBJECT") or "") if orig is not None else ""
        message = normalize_ws(orig.findtext("MESSAGE") or "") if orig is not None else ""
        questions[num] = {"query": normalize_ws(f"{subject}. {message}" if subject else message),
                          "paraphrase": normalize_ws(q.findtext("NIST-PARAPHRASE") or "")}
    mapping = _map_answer_ids(raw_dir, pairs)
    qrels = defaultdict(dict)
    n_judged = n_mapped = 0
    for line in (raw_dir / "liveqa" / "qrels.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        num, label, aid = parts
        if aid.split("_")[0] not in SOURCE_DIRS:
            continue
        n_judged += 1
        pid = mapping.get(aid)
        if pid is None:
            continue
        n_mapped += 1
        qrels[num][pid] = max(qrels[num].get(pid, 0), GAIN[label])
    items = [{"qid": f"liveqa-{num}", "query": questions[num]["query"],
              "paraphrase": questions[num]["paraphrase"], "qrels": rel}
             for num, rel in sorted(qrels.items(), key=lambda kv: int(kv[0]))
             if num in questions and max(rel.values()) >= 2]
    rng = random.Random(seed)
    rng.shuffle(items)
    cut = int(round(len(items) * val_fraction))
    stats = {"liveqa_judged_in_corpus_sources": n_judged, "liveqa_judged_mapped": n_mapped,
             "liveqa_questions_usable": len(items)}
    return items[:cut], items[cut:], stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare(params: dict, raw_dir: Path = DATA_RAW, out_dir: Path = DATA_PROCESSED) -> dict:
    d, c = params["data"], params["chunking"]
    pairs = parse_medquad(raw_dir, d["sources"])
    chunks = make_chunks(pairs, c["max_words"], c["overlap_words"], c["header"])
    mq = medquad_eval(pairs, d["medquad_eval_size"], d["seed"])
    val, test, lstats = liveqa_eval(raw_dir, pairs, d["liveqa_val_fraction"], d["seed"])
    write_jsonl(out_dir / "chunks.jsonl", chunks)
    write_jsonl(out_dir / "eval_medquad.jsonl", mq)
    write_jsonl(out_dir / "eval_liveqa_val.jsonl", val)
    write_jsonl(out_dir / "eval_liveqa_test.jsonl", test)
    stats = {"pairs": len(pairs), "chunks": len(chunks), "medquad_eval": len(mq),
             "liveqa_val": len(val), "liveqa_test": len(test), **lstats,
             "corpus_sha1": sha1("".join(ch["chunk_id"] + ch["text"] for ch in chunks))[:12]}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats

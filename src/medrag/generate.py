"""Answer generation: an OpenAI-compatible LLM (Groq by default) with an extractive fallback.

The fallback keeps the service useful (and CI deterministic) when no API key is set
or the provider fails: it quotes the most query-relevant sentences from the top passages.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .text import sentences, sha1

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an and are as at be by can do does for from has have how i if in is it its me my of on or "
            "the their there this to was what when where which who why will with you your".split())


@dataclass
class Prompt:
    version: str
    system: str
    user: str
    disclaimer: str

    @property
    def sha(self) -> str:
        return sha1(self.system + self.user)[:10]

    @classmethod
    def load(cls, path: str | Path) -> Prompt:
        d = yaml.safe_load(Path(path).read_text())
        return cls(d["version"], d["system"], d["user"], d["disclaimer"])


@dataclass
class Answer:
    text: str
    citations: list[int]
    mode: str                       # "llm" | "extractive" | "no_context"
    model: str | None = None
    latency_ms: float = 0.0
    error: str | None = None
    usage: dict = field(default_factory=dict)


def format_passages(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i + 1}] ({h['source']}: {h['focus']}) {h['body']}" for i, h in enumerate(hits))


def cited_indices(text: str, n: int) -> list[int]:
    found = sorted({int(x) for x in re.findall(r"\[(\d+)\]", text)})
    return [i for i in found if 1 <= i <= n]


def extractive_answer(question: str, hits: list[dict], max_sentences: int = 4) -> Answer:
    if not hits:
        return Answer("I could not find anything relevant in the NIH collection for that question.", [], "no_context")
    q = {w for w in _WORD.findall(question.lower()) if w not in _STOP}
    scored = []
    for rank, h in enumerate(hits[:3]):
        for pos, s in enumerate(sentences(h["body"])):
            n_words = len(s.split())
            if s.endswith("?") or n_words < 5 or n_words > 70:
                continue  # skip headings, questions and run-on lists
            words = set(_WORD.findall(s.lower()))
            overlap = len(q & words) / (len(q) or 1)
            scored.append((overlap - 0.15 * rank - 0.02 * pos, rank, pos, s if s[-1] in ".!" else s + ".",
                           overlap))
    if not scored or max(t[4] for t in scored) < 0.25:
        # Nothing matches the question well: quote the opening of the best passage instead
        lead = [t for t in scored if t[1] == 0][:3] or []
        if not lead:
            h = hits[0]
            return Answer(f"{' '.join(h['body'].split()[:60])}... [1]", [1], "extractive")
        return Answer(" ".join(t[3] for t in sorted(lead, key=lambda t: t[2])) + " [1]", [1], "extractive")
    best = sorted(scored, reverse=True)[:max_sentences]
    best.sort(key=lambda t: (t[1], t[2]))  # keep reading order
    text = " ".join(f"{t[3]} [{t[1] + 1}]" for t in best)
    return Answer(text, sorted({t[1] + 1 for t in best}), "extractive")


def reasoning_extra(model: str, effort: str | None) -> dict:
    """Groq's gpt-oss models think before answering; keep that short and out of the reply."""
    if effort and "gpt-oss" in model:
        return {"reasoning_effort": effort, "include_reasoning": False}
    return {}


class Generator:
    def __init__(self, prompt: Prompt, api_key: str | None, base_url: str, model: str,
                 timeout_s: float = 30.0, max_tokens: int = 1024, temperature: float = 0.1,
                 reasoning_effort: str | None = "low"):
        self.prompt, self.model = prompt, model
        self.max_tokens, self.temperature = max_tokens, temperature
        self.extra_body = reasoning_extra(model, reasoning_effort)
        self.client = None
        if api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=1)

    @property
    def mode(self) -> str:
        return "llm" if self.client else "extractive"

    def answer(self, question: str, hits: list[dict]) -> Answer:
        t0 = time.perf_counter()
        if not hits:
            a = extractive_answer(question, hits)
        elif self.client is None:
            a = extractive_answer(question, hits)
        else:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                    extra_body=self.extra_body or None,
                    messages=[{"role": "system", "content": self.prompt.system},
                              {"role": "user", "content": self.prompt.user.format(
                                  question=question, passages=format_passages(hits))}])
                text = (resp.choices[0].message.content or "").strip()
                usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
                a = Answer(text, cited_indices(text, len(hits)), "llm", self.model, usage=usage)
                if not text:
                    raise ValueError("empty completion")
            except Exception as e:  # provider down, rate limit, bad key: degrade, don't fail
                a = extractive_answer(question, hits)
                a.error = f"{type(e).__name__}: {str(e)[:200]}"
        a.latency_ms = 1000 * (time.perf_counter() - t0)
        return a

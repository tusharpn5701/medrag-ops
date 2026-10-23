"""MLflow helpers: tracking URI defaults, the pyfunc model wrapper, and the model registry.

Registry conventions (registered model "medrag-retriever"):
  alias "challenger"  newest candidate picked by the sweep
  alias "champion"    what production serves; moved only by promote()
Older MLflow servers without alias support fall back to a "role" tag on versions.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

from .config import ROOT

MODEL_NAME = os.environ.get("MEDRAG_MODEL_NAME", "medrag-retriever")


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v if isinstance(v, (int, float, str, bool)) else str(v)
    return out


@contextlib.contextmanager
def mlflow_ctx():
    import mlflow
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    yield mlflow


# --------------------------------------------------------------------------- pyfunc model
def _pyfunc_class():
    import mlflow.pyfunc

    class MedRAGRetriever(mlflow.pyfunc.PythonModel):
        """predict(DataFrame[query, k?]) -> one row per hit."""

        def load_context(self, context):
            from medrag.retrieval import Bundle
            self.bundle = Bundle.load(Path(context.artifacts["bundle"]))

        def predict(self, context, model_input, params=None):
            import pandas as pd
            rows = []
            for i, rec in enumerate(model_input.to_dict("records")):
                k = int(rec.get("k") or 5)
                for h in self.bundle.search(str(rec["query"]), k):
                    rows.append({"query_index": i, "rank": h["rank"], "score": h["score"],
                                 "chunk_id": h["chunk_id"], "pair_id": h["pair_id"], "url": h["url"]})
            return pd.DataFrame(rows)

    return MedRAGRetriever


REQUIREMENTS = ["numpy", "scipy", "scikit-learn", "bm25s", "PyStemmer", "fastembed", "pandas", "pyyaml"]
CODE_PATH = Path(__file__).resolve().parent  # ship the medrag package inside the model


# --------------------------------------------------------------------------- registry helpers
def get_version_by_role(client, role: str):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, role)
    except Exception:
        pass
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception:
        return None
    tagged = [v for v in versions if (v.tags or {}).get("role") == role]
    return max(tagged, key=lambda v: int(v.version)) if tagged else None


def set_role(client, role: str, version: str) -> None:
    try:
        client.set_registered_model_alias(MODEL_NAME, role, version)
    except Exception:
        for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
            if (v.tags or {}).get("role") == role and v.version != version:
                client.delete_model_version_tag(MODEL_NAME, v.version, "role")
    client.set_model_version_tag(MODEL_NAME, version, "role", role)


def download_bundle(client, version, dest: Path) -> Path:
    """Fetch the bundle folder stored inside a registered pyfunc model version."""
    import shutil

    import mlflow
    dest = Path(dest)
    shutil.rmtree(dest, ignore_errors=True)
    tmp = dest.parent / f".dl-{version.version}"
    shutil.rmtree(tmp, ignore_errors=True)
    local = Path(mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{MODEL_NAME}/{version.version}",
                                                     dst_path=str(tmp)))
    found = next(local.rglob("manifest.json")).parent
    shutil.move(str(found), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)
    return dest

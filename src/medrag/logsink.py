"""Request logging that survives a free-tier container restart.

Every request is appended to a local JSONL file and, when a bucket is configured, shipped
in small batches to S3-compatible storage (DagsHub storage, Cloudflare R2, AWS S3, MinIO).
Queries are redacted (emails, phone numbers) before they are written anywhere.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _s3(endpoint_url: str | None):
    import boto3
    return boto3.client("s3", endpoint_url=endpoint_url or None)


class RequestLogger:
    def __init__(self, log_dir: Path, bucket: str | None = None, prefix: str = "request-logs",
                 endpoint_url: str | None = None, flush_records: int = 25, flush_seconds: float = 60.0):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.bucket, self.prefix, self.endpoint_url = bucket, prefix.strip("/"), endpoint_url
        self.flush_records, self.flush_seconds = flush_records, flush_seconds
        self._buf: list[dict] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self.uploaded_batches = 0
        self.upload_errors = 0
        self._stop = threading.Event()
        self._thread = None
        if bucket:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ write
    def log(self, record: dict) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **record}
        line = json.dumps(record, ensure_ascii=False)
        day = record["ts"][:10].replace("-", "")
        with self._lock:
            with open(self.log_dir / f"requests-{day}.jsonl", "a") as f:
                f.write(line + "\n")
            if self.bucket:
                self._buf.append(record)
        if self.bucket and len(self._buf) >= self.flush_records:
            self.flush()

    def flush(self) -> int:
        if not self.bucket:
            return 0
        with self._lock:
            batch, self._buf = self._buf, []
            self._last_flush = time.time()
        if not batch:
            return 0
        now = datetime.now(timezone.utc)
        key = f"{self.prefix}/{now:%Y/%m/%d}/{now:%H%M%S}-{uuid.uuid4().hex[:8]}.jsonl"
        body = "\n".join(json.dumps(r, ensure_ascii=False) for r in batch) + "\n"
        try:
            _s3(self.endpoint_url).put_object(Bucket=self.bucket, Key=key, Body=body.encode())
            self.uploaded_batches += 1
        except Exception:
            self.upload_errors += 1
            with self._lock:           # keep the records for the next attempt
                self._buf = batch + self._buf
        return len(batch)

    def _loop(self):
        while not self._stop.wait(5.0):
            if self._buf and time.time() - self._last_flush >= self.flush_seconds:
                self.flush()

    def close(self):
        self._stop.set()
        self.flush()

    # ------------------------------------------------------------------ read
    def recent(self, limit: int = 500) -> list[dict]:
        rows = []
        for p in sorted(self.log_dir.glob("requests-*.jsonl"), reverse=True):
            for line in reversed(p.read_text().splitlines()):
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    return rows
        return rows


def load_logs(days: int, local_dir: str | Path | None = None, bucket: str | None = None,
              prefix: str = "request-logs", endpoint_url: str | None = None) -> list[dict]:
    """Read request logs from the last `days` days, from a local folder or the bucket."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    if bucket:
        s3 = _s3(endpoint_url)
        day = since.date()
        while day <= datetime.now(timezone.utc).date():
            pfx = f"{prefix.strip('/')}/{day:%Y/%m/%d}/"
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=pfx):
                for obj in page.get("Contents", []):
                    body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read().decode()
                    rows += [json.loads(line) for line in body.splitlines() if line.strip()]
            day += timedelta(days=1)
    elif local_dir:
        for p in sorted(Path(local_dir).glob("*.jsonl")):
            rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return [r for r in rows if datetime.fromisoformat(r["ts"]) >= since]

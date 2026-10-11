"""
빌드 스크립트가 함께 쓰는 것: DashScope 키 적재, 임베딩 캐시, 문서 임베딩 호출.

키는 운영 검색과 같은 키다. 호출은 순차로, 한 번에 10건씩만 보낸다(한도를 넘기면 운영 검색이 실패한다).
키를 찾는 순서: 환경변수 QWEN_API_KEY → back/.env 의 `QWEN_API_KEY=...` 줄(.env 는 커밋되지 않는다).
"""
import asyncio
import hashlib
import json
import os
import sys

import httpx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACK = os.path.dirname(HERE)
ENV = os.path.join(BACK, ".env")


def load_key():
    if os.environ.get("QWEN_API_KEY"):
        return
    if os.path.exists(ENV):
        with open(ENV, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("QWEN_API_KEY="):
                    os.environ["QWEN_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                    return
    sys.exit("QWEN_API_KEY 없음 — 환경변수로 주거나 back/.env 에 적는다")


load_key()
sys.path.insert(0, BACK)
import app  # noqa: E402  (키를 넣은 뒤에 불러야 한다)


class Cache:
    """글 단위 임베딩 캐시(JSON 파일). 같은 글은 다시 부르지 않는다."""

    def __init__(self, path):
        self.path = path
        self.d = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.d = json.load(f)
        self.dirty = 0

    @staticmethod
    def key(kind, text):
        return kind + ":" + hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get(self, kind, text):
        return self.d.get(self.key(kind, text))

    def put(self, kind, text, val):
        self.d[self.key(kind, text)] = val
        self.dirty += 1
        if self.dirty >= 50:
            self.save()

    def save(self):
        if self.path:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.d, f)
            self.dirty = 0


async def post(client, url, body):
    for attempt in range(6):
        r = await client.post(url, headers={"Authorization": f"Bearer {app.QWEN_API_KEY}"}, json=body)
        if r.status_code == 429 or r.status_code >= 500:
            await asyncio.sleep(3 * (attempt + 1))  # 한도에 걸리면 물러선다
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"DashScope 실패: {r.status_code}")


def missing(texts, cache):
    """캐시에 없어 실제로 호출해야 하는 글(중복 제거)."""
    return [t for t in dict.fromkeys(texts) if cache.get("d", t) is None]


async def embed_docs(texts, cache):
    """문서 임베딩(정규화). 캐시에 없는 글만 10건씩 순차로 부른다."""
    need = missing(texts, cache)
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(0, len(need), 10):
            chunk = need[i:i + 10]
            d = await post(client, f"{app.DASHSCOPE_BASE}/embeddings",
                           {"model": app.EMBED_MODEL, "input": chunk, "dimensions": app.DIM, "encoding_format": "float"})
            for t, e in zip(chunk, sorted(d["data"], key=lambda x: x["index"])):
                cache.put("d", t, e["embedding"])
            await asyncio.sleep(0.2)
    cache.save()
    out = {}
    for t in texts:
        v = np.asarray(cache.get("d", t), dtype=np.float32)
        n = np.linalg.norm(v)
        out[t] = v / n if n > 0 else v
    return out

"""
pinkbean-customize-shop-back — 코디 아이템 벡터 검색 API (1차)

- 사전연산 벡터(vectors_256.f16.npy, 정규화) + 메타(meta.json)를 기동 시 메모리에 적재.
- POST /search : 질의문을 text-embedding-v4(DashScope)로 임베딩 → 코사인(내적) topK → 아이템 리스트.
- 코사인 = 내적(양쪽 정규화). 15,672 x 256 브루트포스 matmul 이라 수 ms 수준(매우 빠름).
- cpu=1 + uvicorn 워커 다중화로 동시요청/임베딩 I/O 병렬.

품질 주의: 캡션 QC 11%, Qwen3 VL Flash/Plus + Sonnet 정정. 아직 개선 중(프론트에 '베타' 표기).
"""
import os
import json
import time
import numpy as np
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DIM = int(os.environ.get("EMBED_DIM", "256"))

QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
# DashScope OpenAI-호환 임베딩 엔드포인트. 계정 리전에 따라 intl / 본토(cn) 로 바꿀 수 있게 env 로.
DASHSCOPE_BASE = os.environ.get(
    "DASHSCOPE_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-v4")

# ── 데이터 적재(모듈 로드 = 워커별 1회) ─────────────────────────────
_t0 = time.time()
MAT = np.load(os.path.join(DATA, f"vectors_{DIM}.f16.npy")).astype(np.float32)  # (N, DIM)
with open(os.path.join(DATA, "meta.json"), "r", encoding="utf-8") as f:
    META = json.load(f)
ITEMS = META["items"]  # 벡터 행과 동일 순서
N = MAT.shape[0]
# 슬롯 → 행 인덱스(필터 검색 가속)
SLOT_ROWS: dict[str, np.ndarray] = {}
for i, it in enumerate(ITEMS):
    SLOT_ROWS.setdefault(it["slot"], []).append(i)
SLOT_ROWS = {k: np.asarray(v, dtype=np.int64) for k, v in SLOT_ROWS.items()}
print(f"[app] loaded {N} vectors dim={DIM} in {time.time()-_t0:.2f}s · slots={list(SLOT_ROWS)}")

app = FastAPI(title="pinkbean-customize-shop-back", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 무자격 POST 검색 API → 전체 허용(자격증명 없음)
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchReq(BaseModel):
    query: str
    slot: str | None = None      # 특정 슬롯(hair/cap/weapon...)만 검색. None=전체
    topK: int = 60


async def embed_query(text: str) -> np.ndarray:
    """질의문을 문서와 동일 모델·차원으로 임베딩(정규화 반환)."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            f"{DASHSCOPE_BASE}/embeddings",
            headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
            json={"model": EMBED_MODEL, "input": text, "dimensions": DIM,
                  "encoding_format": "float"},
        )
        r.raise_for_status()
        v = np.asarray(r.json()["data"][0]["embedding"], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


@app.get("/health")
def health():
    return {"ok": True, "count": N, "dim": DIM, "model": EMBED_MODEL,
            "slots": {k: int(len(v)) for k, v in SLOT_ROWS.items()}}


@app.post("/search")
async def search(req: SearchReq):
    q = (req.query or "").strip()
    if not q:
        return {"query": q, "count": 0, "results": []}
    t0 = time.time()
    qv = await embed_query(q)
    t_embed = time.time() - t0

    if req.slot and req.slot in SLOT_ROWS:
        rows = SLOT_ROWS[req.slot]
        scores = MAT[rows] @ qv
        k = min(req.topK, len(rows))
        top = rows[np.argsort(-scores)[:k]]
        top_scores = MAT[top] @ qv
    else:
        scores = MAT @ qv
        k = min(req.topK, N)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        top, top_scores = idx, scores[idx]

    results = []
    for row, sc in zip(top.tolist(), np.asarray(top_scores).tolist()):
        it = ITEMS[row]
        results.append({
            "id": it["id"], "slot": it["slot"], "name": it["name"],
            "grade": it["grade"], "isCash": it["isCash"], "gender": it["gender"],
            "image_source": it["image_source"], "score": round(float(sc), 4),
        })
    return {"query": q, "slot": req.slot, "count": len(results),
            "ms": {"embed": round(t_embed * 1000), "total": round((time.time() - t0) * 1000)},
            "results": results}

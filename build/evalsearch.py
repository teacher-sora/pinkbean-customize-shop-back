"""
캡션 방식의 검색 성능을 오프라인으로 잰다. 질의 쪽 처리는 운영과 같다(app.refine_query → 색 낱말 분리 → 정렬 → 임베딩).

  python build/evalsearch.py --docs DOCS.json --queries QUERIES.json --cache CACHE.json [--bonus 0.1] [--out RESULT.json]

DOCS    {"<방식>": {"<id>": {"slot": "hair", "texts": ["낱말 낱말 …", …]}}}
        texts 가 여럿이면 아이템당 벡터 여러 개로 보고 점수는 최댓값을 쓴다.
QUERIES [{"q": "발끝까지 오는 양갈래", "target": "00071570", "slot": "hair"}]
순위는 같은 부위·같은 방식의 문서들 안에서 매긴다. MRR · recall@1/5 를 방식별·부위별로 낸다.

DashScope 키는 운영 검색과 같은 키다. 호출은 순차로, 문서는 한 번에 10건씩만 보낸다(한도를 넘기면 운영 검색이 실패한다).
임베딩은 --cache 파일에 글 단위로 저장해 다시 부르지 않는다.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time

import httpx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACK = os.path.dirname(HERE)
ENV = os.path.abspath(os.path.join(BACK, "..", "..", "maple test", ".env.copy"))


def load_key():
    if os.environ.get("QWEN_API_KEY"):
        return
    with open(ENV, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("QWEN_API_KEY="):
                os.environ["QWEN_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                return
    sys.exit("QWEN_API_KEY 없음")


load_key()
sys.path.insert(0, BACK)
import app  # noqa: E402  (키를 넣은 뒤에 불러야 한다)


class Cache:
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


async def embed_docs(texts, cache):
    """문서 임베딩(정규화). 10건씩 순차."""
    need = [t for t in dict.fromkeys(texts) if cache.get("d", t) is None]
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


async def refine(q, cache):
    """운영 /search 와 같은 질의 처리 → 임베딩에 들어가는 글."""
    hit = cache.get("r", q)
    if hit is not None:
        return hit
    ref = await app.refine_query(q)
    if ref:
        words = ref[0]
    else:
        words = app.detect_gender(q)[1].split()
    concept = [w for w in words if not app.canon_colors(w)]
    cleaned = " ".join(sorted(concept or words))
    cache.put("r", q, cleaned)
    await asyncio.sleep(0.15)
    return cleaned


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--bonus", type=float, default=0.0, help="질의 낱말이 캡션에 그대로 있으면 주는 가산(낱말 일치 비율 × 값)")
    ap.add_argument("--pool", default="docs", choices=["docs", "back"], help="back: 운영 데이터의 같은 부위 전체(대상 제외)를 경쟁자로 둔다")
    ap.add_argument("--out")
    a = ap.parse_args()

    docs = json.load(open(a.docs, encoding="utf-8"))
    queries = json.load(open(a.queries, encoding="utf-8"))
    cache = Cache(a.cache)

    cleaned = {}
    for q in dict.fromkeys(x["q"] for x in queries):
        cleaned[q] = await refine(q, cache)
    cache.save()
    qvec = await embed_docs(list(cleaned.values()), cache)

    report = {}
    for variant, items in docs.items():
        texts = [t for it in items.values() for t in it["texts"]]
        dvec = await embed_docs(texts, cache)
        by_slot = {}
        for iid, it in items.items():
            by_slot.setdefault(it["slot"], []).append(iid)
        ranks, per_slot, detail = [], {}, []
        for x in queries:
            pool = by_slot.get(x["slot"], [])
            if x["target"] not in pool:
                continue
            c = cleaned[x["q"]]
            qv = qvec[c]
            qwords = c.split()
            def score_of(ts):
                v = max(float(dvec[t] @ qv) for t in ts)
                if a.bonus and qwords:
                    joined = " ".join(ts)
                    v += a.bonus * sum(1 for w in qwords if w in joined) / len(qwords)
                return v
            if a.pool == "back":
                # 운영 벡터(옛 캡션)의 같은 부위 전체와 겨룬다. 대상 아이템의 옛 행은 뺀다.
                rows = app.SLOT_ROWS[x["slot"]]
                rows = rows[[app.ITEMS[r]["id"] != x["target"] for r in rows.tolist()]]
                others = app.MAT[rows] @ qv
                if a.bonus and qwords:  # 경쟁자에게도 같은 가산을 준다(옛 캡션 낱말 기준)
                    others = others + np.asarray([a.bonus * sum(1 for w in qwords if w in " ".join(app.ITEMS[r].get("words") or [])) / len(qwords)
                                                  for r in rows.tolist()], dtype=np.float32)
                ts_ = score_of(items[x["target"]]["texts"])
                rank = 1 + int((others > ts_).sum())
                j = int(others.argmax())
                scores = [(float(others[j]), app.ITEMS[int(rows[j])]["id"]), (ts_, x["target"])]
                pool = list(rows) + [x["target"]]
            else:
                scores = [(score_of(items[iid]["texts"]), iid) for iid in pool]
                scores.sort(reverse=True)
                rank = [i for _, i in scores].index(x["target"]) + 1
            ranks.append(rank)
            per_slot.setdefault(x["slot"], []).append(rank)
            detail.append({"q": x["q"], "cleaned": c, "target": x["target"], "slot": x["slot"], "rank": rank, "pool": len(pool),
                           "top": scores[0][1], "top_score": round(scores[0][0], 3),
                           "target_score": round([s for s, i in scores if i == x["target"]][0], 3)})

        def summ(r):
            r = np.asarray(r, dtype=np.float32)
            return {"n": int(len(r)), "mrr": round(float((1 / r).mean()), 3) if len(r) else None,
                    "r1": round(float((r <= 1).mean()), 3) if len(r) else None,
                    "r5": round(float((r <= 5).mean()), 3) if len(r) else None,
                    "r10": round(float((r <= 10).mean()), 3) if len(r) else None, "median": float(np.median(r)) if len(r) else None}
        report[variant] = {"all": summ(ranks), "slots": {s: summ(r) for s, r in sorted(per_slot.items())}, "detail": detail}
        print(f"{variant:12} n={len(ranks):4} MRR {report[variant]['all']['mrr']}  R@1 {report[variant]['all']['r1']}  R@5 {report[variant]['all']['r5']}  R@10 {report[variant]['all']['r10']}  중앙 {report[variant]['all']['median']}")
    if a.out:
        json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(main())
    print(f"({time.time() - t0:.0f}s)")

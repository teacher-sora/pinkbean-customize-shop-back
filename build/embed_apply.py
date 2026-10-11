"""
캡션 정본(back/captions/<부위>.jsonl)을 검색 데이터(back/data/)에 반영한다.

  python build/embed_apply.py --slots hair,face [--cache CACHE.json] [--dry]

하는 일
  · meta.json: 그 부위 아이템의 words(상세 캡션)·short(짧은 캡션)를 바꾸고 tier 를 "v2" 로 적는다.
    정본에만 있는 아이템은 뒤에 덧붙인다(행 순서는 바꾸지 않는다 — 벡터 행과 1:1).
  · vectors_256.f16.npy: 바뀐 행만 상세 캡션으로 다시 임베딩한다.
  · vectors_short_256.f16.npy + meta.short_rows: 짧은 캡션 벡터와, 그 벡터가 속한 아이템 행 번호.
    왜 둘인가: 질의는 6낱말 이하라 상세 캡션 하나만 임베딩하면 희석돼 순위가 떨어진다(짧은 것과 상세한 것 중 높은 쪽을 쓴다).
  · 다른 부위의 행(words·벡터)은 건드리지 않고, 끝에 바이트 단위로 같음을 검사한다.
  · 행을 지우지 않는다. 정본에 없는 아이템의 행은 그대로 남는다(tier 도 그대로).
  · 덧붙인 행에는 recency 가 없다 — 헤어·성형·무기를 넣었으면 recency.py 를 이어서 돌린다.
  · --dry 는 아무것도 부르지도 저장하지도 않고, 바뀔 행 수와 임베딩을 새로 불러야 하는 글 수만 센다.

임베딩 규격: DashScope intl · text-embedding-v4 · 256차원 · 낱말을 공백으로 이은 글 · L2 정규화 · float16.
키는 운영 검색과 같은 키라 순차로 10건씩만 보낸다. 부른 글은 캐시(build/.embcache.json, 커밋 안 함)에 남겨 다시 부르지 않는다 —
캐시가 없는 새 컴퓨터에서는 짧은 캡션 전체(아이템 수만큼)와 그 부위의 상세 캡션을 한 번 다시 부른다.
"""
import argparse
import asyncio
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from dashscope import Cache, embed_docs, missing  # noqa: E402  (키 적재와 임베딩 호출)

DATA = os.path.join(BACK, "data")
CAPS = os.path.join(BACK, "captions")


def gender_of(name):
    n = name or ""
    return 1 if "(여)" in n else 0 if "(남)" in n else 2


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="", help="캡션 전체를 정본으로 바꿀 부위")
    ap.add_argument("--cache", default=os.path.join(BACK, "build", ".embcache.json"))
    ap.add_argument("--dry", action="store_true", help="부르지도 저장하지도 않고 바뀔 내용만 센다")
    a = ap.parse_args()
    slots = [s for s in a.slots.split(",") if s]
    touched = set(slots)

    meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
    items = meta["items"]
    dim = meta["dim"]
    V = np.load(os.path.join(DATA, f"vectors_{dim}.f16.npy"))
    assert V.shape[0] == len(items), "meta 와 벡터 행 수가 다르다"
    before_items = json.dumps([it for it in items if it["slot"] not in touched], ensure_ascii=False)
    before_rows = V[[i for i, it in enumerate(items) if it["slot"] not in touched]].tobytes()

    row_of = {it["id"]: i for i, it in enumerate(items)}

    caps = {}
    for s in slots:
        f = os.path.join(CAPS, f"{s}.jsonl")
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                caps[r["id"]] = r
    changed, added = [], []
    for cid, r in caps.items():
        if cid in row_of:
            it = items[row_of[cid]]
            it["words"], it["short"], it["tier"] = r["caption"], r["short"], "v2"
            changed.append(row_of[cid])
            # 모자의 갈래(kind)·귀 유무는 검색의 귀 규칙이 쓴다 — 낱말이 아니라 관찰 필드로 가른다.
            for k in ("kind", "has_ears", "animal", "type"):
                if r.get(k) is not None:
                    it[k] = r[k]
        else:
            items.append({"id": cid, "slot": r["slot"], "name": r["name"], "words": r["caption"], "short": r["short"],
                          "gender": gender_of(r["name"]), "label": r.get("label"), "isCash": r.get("isCash", True), "tier": "v2"})
            for k in ("kind", "has_ears", "animal", "type"):
                if r.get(k) is not None:
                    items[-1][k] = r[k]
            added.append(len(items) - 1)
    print(f"바뀐 행 {len(changed)} · 덧붙인 행 {len(added)} · 전체 {len(items)}")

    cache = Cache(a.cache)
    full_text = {i: " ".join(items[i]["words"]) for i in changed + added}
    short_rows = [i for i, it in enumerate(items) if it.get("short")]
    short_text = {i: " ".join(items[i]["short"]) for i in short_rows}
    texts = list(full_text.values()) + list(short_text.values())
    if a.dry:
        print(f"임베딩을 새로 불러야 하는 글 {len(missing(texts, cache))}개(10건씩 순차) · 저장하지 않음")
        return
    vec = await embed_docs(texts, cache)

    V2 = np.zeros((len(items), dim), dtype=np.float16)
    V2[: V.shape[0]] = V
    for i, t in full_text.items():
        V2[i] = vec[t].astype(np.float16)
    S = np.stack([vec[short_text[i]] for i in short_rows]).astype(np.float16) if short_rows else np.zeros((0, dim), dtype=np.float16)

    after_items = json.dumps([it for it in items if it["slot"] not in touched], ensure_ascii=False)
    after_rows = V2[[i for i, it in enumerate(items) if it["slot"] not in touched]].tobytes()
    assert before_items == after_items and before_rows == after_rows, "다른 부위 데이터가 바뀌었다 — 저장하지 않는다"
    print("다른 부위 데이터: 바이트 단위로 동일")

    meta.update({"count": len(items), "items": items, "short_rows": short_rows})
    np.save(os.path.join(DATA, f"vectors_{dim}.f16.npy"), V2)
    np.save(os.path.join(DATA, f"vectors_short_{dim}.f16.npy"), S)
    json.dump(meta, open(os.path.join(DATA, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"저장: vectors {V2.shape} · short {S.shape} · meta {len(items)}")


if __name__ == "__main__":
    asyncio.run(main())

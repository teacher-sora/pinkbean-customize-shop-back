"""
캡션 정본(back/captions/<부위>.jsonl)을 검색 데이터(back/data/)에 반영한다.

  python build/embed_apply.py --slots hair,face [--new cap,weapon] [--legs longcoat,pants] [--cache CACHE.json] [--dry]

하는 일
  · meta.json: 그 부위 아이템의 words(상세 캡션)·short(짧은 캡션)를 바꾸고 tier 를 "v2" 로 적는다.
    정본에만 있는 아이템은 뒤에 덧붙인다(행 순서는 바꾸지 않는다 — 벡터 행과 1:1).
  · vectors_256.f16.npy: 바뀐 행만 상세 캡션으로 다시 임베딩한다.
  · vectors_short_256.f16.npy + meta.short_rows: 짧은 캡션 벡터와, 그 벡터가 속한 아이템 행 번호.
    왜 둘인가: 질의는 6낱말 이하라 상세 캡션 하나만 임베딩하면 희석돼 순위가 떨어진다(짧은 것과 상세한 것 중 높은 쪽을 쓴다).
  · 다른 부위의 행(words·벡터)은 건드리지 않고, 끝에 바이트 단위로 같음을 검사한다.

임베딩 규격: DashScope intl · text-embedding-v4 · 256차원 · 낱말을 공백으로 이은 글 · L2 정규화 · float16.
키는 운영 검색과 같은 키라 순차로 10건씩만 보낸다.
"""
import argparse
import asyncio
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evalsearch import Cache, embed_docs, app  # noqa: E402  (키 적재와 임베딩 호출을 같이 쓴다)

DATA = os.path.join(BACK, "data")
CAPS = os.path.join(BACK, "captions")


LEG_OLD = re.compile("스타킹|니삭스|양말|타이츠|삭스|레깅스|신발 포함")
LEG_NEW = re.compile("스타킹|니삭스|양말|신발 포함")


def gender_of(name):
    n = name or ""
    return 1 if "(여)" in n else 0 if "(남)" in n else 2


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="", help="캡션 전체를 정본으로 바꿀 부위")
    ap.add_argument("--new", default="", help="정본에만 있는(검색 데이터에 아직 없는) 아이템만 덧붙일 부위 — 전량 재캡션 전인 부위의 패치 신규분")
    ap.add_argument("--legs", default="", help="다리 낱말만 정본으로 바꿀 부위(longcoat,pants)")
    ap.add_argument("--cache", default=os.path.join(BACK, "build", ".embcache.json"))
    ap.add_argument("--dry", action="store_true", help="저장하지 않고 바뀔 내용만 센다")
    a = ap.parse_args()
    slots = [s for s in a.slots.split(",") if s]
    legs = [s for s in a.legs.split(",") if s]
    new_only = [s for s in a.new.split(",") if s]
    touched = set(slots) | set(legs) | set(new_only)

    meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
    items = meta["items"]
    dim = meta["dim"]
    V = np.load(os.path.join(DATA, f"vectors_{dim}.f16.npy"))
    assert V.shape[0] == len(items), "meta 와 벡터 행 수가 다르다"
    before_items = json.dumps([it for it in items if it["slot"] not in touched], ensure_ascii=False)
    before_rows = V[[i for i, it in enumerate(items) if it["slot"] not in touched]].tobytes()

    row_of = {it["id"]: i for i, it in enumerate(items)}

    # --legs: 그 부위는 캡션 전체를 바꾸지 않고 **다리 낱말만** 정본의 것으로 바꾼다(스타킹·니삭스·양말·신발 포함).
    # 왜: 한벌옷·하의는 아직 전량 재캡션 전이라, 옛 캡션에 새 캡션을 섞으면 순위가 한쪽으로 쏠린다. 스타킹 판정만 먼저 바로잡는다.
    leg_changed = []
    for s in legs:
        f = os.path.join(CAPS, f"{s}.jsonl")
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] not in row_of or r.get("legs") is None:
                continue
            it = items[row_of[r["id"]]]
            new_leg = [w for w in r["caption"] if LEG_NEW.search(w)]
            words = [w for w in it["words"] if not LEG_OLD.search(w)] + new_leg
            if words != it["words"]:
                it["words"] = words
                leg_changed.append(row_of[r["id"]])
    if legs:
        print(f"다리 낱말만 바꾼 행 {len(leg_changed)}")

    caps = {}
    for s in slots + new_only:
        f = os.path.join(CAPS, f"{s}.jsonl")
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if s in new_only and r["id"] in row_of:
                    continue   # 이미 있는 아이템은 건드리지 않는다(옛 캡션과 섞지 않는다)
                caps[r["id"]] = r
    changed, added = [], []
    for cid, r in caps.items():
        if cid in row_of:
            it = items[row_of[cid]]
            it["words"], it["short"], it["tier"] = r["caption"], r["short"], "v2"
            changed.append(row_of[cid])
            # 모자의 갈래(kind)·귀 유무는 검색의 귀 규칙이 쓴다 — 낱말이 아니라 관찰 필드로 가른다.
            for k in ("kind", "has_ears", "animal"):
                if r.get(k) is not None:
                    it[k] = r[k]
        else:
            items.append({"id": cid, "slot": r["slot"], "name": r["name"], "words": r["caption"], "short": r["short"],
                          "gender": gender_of(r["name"]), "label": r.get("label"), "isCash": r.get("isCash", True), "tier": "v2"})
            for k in ("kind", "has_ears", "animal"):
                if r.get(k) is not None:
                    items[-1][k] = r[k]
            added.append(len(items) - 1)
    print(f"바뀐 행 {len(changed)} · 덧붙인 행 {len(added)} · 전체 {len(items)}")

    cache = Cache(a.cache)
    full_text = {i: " ".join(items[i]["words"]) for i in changed + added + leg_changed}
    short_rows = [i for i, it in enumerate(items) if it.get("short")]
    short_text = {i: " ".join(items[i]["short"]) for i in short_rows}
    vec = await embed_docs(list(full_text.values()) + list(short_text.values()), cache)

    V2 = np.zeros((len(items), dim), dtype=np.float16)
    V2[: V.shape[0]] = V
    for i, t in full_text.items():
        V2[i] = vec[t].astype(np.float16)
    S = np.stack([vec[short_text[i]] for i in short_rows]).astype(np.float16) if short_rows else np.zeros((0, dim), dtype=np.float16)

    after_items = json.dumps([it for it in items if it["slot"] not in touched], ensure_ascii=False)
    after_rows = V2[[i for i, it in enumerate(items) if it["slot"] not in touched]].tobytes()
    assert before_items == after_items and before_rows == after_rows, "다른 부위 데이터가 바뀌었다 — 저장하지 않는다"
    print("다른 부위 데이터: 바이트 단위로 동일")

    if a.dry:
        return
    meta.update({"count": len(items), "items": items, "short_rows": short_rows})
    np.save(os.path.join(DATA, f"vectors_{dim}.f16.npy"), V2)
    np.save(os.path.join(DATA, f"vectors_short_{dim}.f16.npy"), S)
    json.dump(meta, open(os.path.join(DATA, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"저장: vectors {V2.shape} · short {S.shape} · meta {len(items)}")


if __name__ == "__main__":
    asyncio.run(main())

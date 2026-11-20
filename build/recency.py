"""
아이템별 기본 최신도(recency, 0~1)를 back/data/meta.json 에 넣는다. app.py 가 이것으로 최신 가산을 준다.

  python build/recency.py --slots-dir DIR

최신도 = max( 부위 안 id 백분위 , 출시 순번(sn) 백분위 )
  · sn: CDN slots/*.json 에 있는 부위만(무기 등). 헤어·성형에는 없다. DIR 에 CDN 의 slots/{부위}.json 을 받아 둔다.
  · 이번 신규 · 지난 시즌 신규(CDN catalog/new.json · released.json)는 여기서 넣지 않는다 — app.py 가 CDN 에서 받아 이 값보다 위에 둔다.
왜 R2 meta 수정 시각을 쓰지 않나: meta 를 다시 올린 날(헤어 재추출, 염색 불가 수정 등)을 패치 반영 시각으로 잘못 읽어 옛 아이템이 최신으로 떴다.
대상 부위는 app.py 가 최신 가산을 주는 hair · face · weapon.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
SLOTS = ("hair", "face", "weapon")

ap = argparse.ArgumentParser()
ap.add_argument("--slots-dir", required=True, help="CDN slots/*.json 을 받아 둔 폴더(sn 용)")
a = ap.parse_args()

sn = {}
for s in SLOTS:
    f = os.path.join(a.slots_dir, f"{s}.json")
    if os.path.exists(f):
        for e in json.load(open(f, encoding="utf-8")):
            if e.get("sn"):
                sn[e["id"]] = e["sn"]

meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
items = meta["items"]
changed = 0
for s in SLOTS:
    rows = [i for i, it in enumerate(items) if it["slot"] == s]
    if len(rows) < 2:
        continue
    by_id = sorted(rows, key=lambda i: int(items[i]["id"]))
    id_pct = {i: r / (len(rows) - 1) for r, i in enumerate(by_id)}
    with_sn = sorted([i for i in rows if items[i]["id"] in sn], key=lambda i: sn[items[i]["id"]])
    sn_pct = {i: r / max(1, len(with_sn) - 1) for r, i in enumerate(with_sn)}
    lifted = 0
    for i in rows:
        v = max(id_pct[i], sn_pct.get(i, 0.0))
        if v > id_pct[i] + 0.05:
            lifted += 1
        items[i]["recency"] = round(v, 4)
        changed += 1
    print(f"{s}: {len(rows)}개 · id 백분위보다 올라간 것 {lifted}개 (sn 있는 것 {len(with_sn)})")
json.dump(meta, open(os.path.join(DATA, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(f"recency 기록 {changed}개")

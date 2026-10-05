"""
아이템별 최신도(recency, 0~1)를 back/data/meta.json 에 넣는다. app.py 가 이것으로 최신 가산을 준다.

  python build/recency.py --times META_TIMES.txt [--slots-dir DIR]

왜: id 가 높다고 최신이 아니다. 낮은 id 로 나온 신상품이 적지 않다(사용자 지적, 2026-10-04).
최신도 = max( 부위 안 id 백분위 , 패치 반영 시각 점수 , 출시 순번(sn) 백분위 )
  · 패치 반영 시각: R2 `meta/{id}.json` 의 수정 시각. 이관 기준일(BASELINE) 뒤에 들어온 것은 id 와 무관하게 최신으로 본다.
    META_TIMES.txt 는 `rclone lsf r2:<버킷>/meta --files-only --format "tp" --separator ";"` 의 출력이다.
  · sn: CDN slots/*.json 에 있는 부위만(무기·한벌옷 등). 헤어·성형에는 없다.
한계: 이관 이전에 들어온 '저 id 신상품'은 가릴 자료가 없다 — 그래서 app.py 의 가산 상한을 낮게(0.03) 둔다.
대상 부위는 app.py 가 최신 가산을 주는 hair · face · weapon.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
BASELINE = "2026-07-20"   # 이 날짜 뒤에 반영된 meta 는 패치 신규분이다
SLOTS = ("hair", "face", "weapon")

ap = argparse.ArgumentParser()
ap.add_argument("--times", required=True)
ap.add_argument("--slots-dir", default=None, help="CDN slots/*.json 을 받아 둔 폴더(sn 용). 없으면 sn 은 쓰지 않는다")
a = ap.parse_args()

when = {}
for line in open(a.times, encoding="utf-8"):
    if ";" in line:
        t, p = line.strip().split(";", 1)
        when[p.replace(".json", "")] = t[:10]
patch_dates = sorted({d for d in when.values() if d > BASELINE})
# 가장 최근 패치 1.0, 그 앞 패치일수록 조금씩 낮게(가장 오래된 패치도 0.9 — 기준선 아이템 대부분보다 위).
patch_score = {d: 0.9 + 0.1 * (i + 1) / len(patch_dates) for i, d in enumerate(patch_dates)}

sn = {}
if a.slots_dir:
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
        p = patch_score.get(when.get(items[i]["id"], ""), 0.0)
        v = max(id_pct[i], p, sn_pct.get(i, 0.0))
        if v > id_pct[i] + 0.05:
            lifted += 1
        items[i]["recency"] = round(v, 4)
        changed += 1
    print(f"{s}: {len(rows)}개 · id 백분위보다 올라간 것 {lifted}개 (sn 있는 것 {len(with_sn)})")
json.dump(meta, open(os.path.join(DATA, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(f"recency 기록 {changed}개 · 패치일 {patch_dates}")

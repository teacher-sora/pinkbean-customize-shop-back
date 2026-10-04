"""
고정 질의 세트(build/eval_queries.json)를 검색 서버에 던져 결과를 저장하거나, 두 저장본을 비교한다.

  python build/snapshot.py take  --base https://pinkbean-customize-shop-back.fly.dev --out before.json
  python build/snapshot.py take  --base http://127.0.0.1:8080 --out after.json
  python build/snapshot.py diff  before.json after.json [--changed hair,face]

diff 가 보는 것
  · checks: expect(상위 N 안에 있어야 함) · reject(있으면 안 됨) · expect_slot(전체 검색에서 그 부위가 다수인가)
  · stable: --changed 에 없는 부위를 지정한 질의는 상위 10 이 전과 같아야 한다(의도하지 않은 변화 감지)
  · 응답 필드 집합과 응답 시간(ms.total 중앙값)
호출은 순차로 한다(운영 검색과 같은 DashScope 키를 쓰는 서버다).
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
QS = json.load(open(os.path.join(HERE, "eval_queries.json"), encoding="utf-8"))


def search(base, q, slot, topk=300):
    body = json.dumps({"query": q, "slot": slot, "topK": topk}).encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + "/search", data=body, headers={"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # 깨어나는 중이거나 일시 오류
            last = e
            time.sleep(3 * (attempt + 1))
    raise last


def take(base, out):
    res = {"base": base, "health": json.loads(urllib.request.urlopen(base.rstrip("/") + "/health", timeout=40).read()), "runs": []}
    for kind in ("checks", "stable"):
        for c in QS[kind]:
            d = search(base, c["q"], c.get("slot"))
            res["runs"].append({"kind": kind, "q": c["q"], "slot": c.get("slot"), "resolved_slot": d.get("slot"), "refined": d.get("refined"),
                                "count": d.get("count"), "ms": (d.get("ms") or {}).get("total"), "keys": sorted(d.keys()),
                                "item_keys": sorted(d["results"][0].keys()) if d.get("results") else [],
                                "top": [{"id": r["id"], "slot": r["slot"], "name": r["name"], "score": r["score"]} for r in d.get("results", [])[:300]]})
            time.sleep(0.3)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"저장 {out}: 질의 {len(res['runs'])} · count {res['health'].get('count')}")


def diff(a, b, changed):
    A = json.load(open(a, encoding="utf-8"))
    B = json.load(open(b, encoding="utf-8"))
    ra = {(r["q"], r["slot"]): r for r in A["runs"]}
    ok = True
    lines = [f"count {A['health'].get('count')} → {B['health'].get('count')}"]
    ms_a = statistics.median([r["ms"] for r in A["runs"] if r["ms"]])
    ms_b = statistics.median([r["ms"] for r in B["runs"] if r["ms"]])
    lines.append(f"응답 시간 중앙값 {ms_a}ms → {ms_b}ms")
    spec = {(c["q"], c.get("slot")): c for c in QS["checks"]}
    for r in B["runs"]:
        key = (r["q"], r["slot"])
        old = ra.get(key)
        if old and (old["keys"] != r["keys"] or (old["item_keys"] and r["item_keys"] and not set(old["item_keys"]) <= set(r["item_keys"]))):
            ok = False
            lines.append(f"✗ 응답 필드가 바뀜: {r['q']}")
        if r["kind"] == "checks":
            c = spec[key]
            n = c.get("top", 10)
            names = [t["name"] or "" for t in r["top"][:n]]
            miss = [e for e in c.get("expect", []) if not any(e in x for x in names)]
            bad = [x for x in names if any(j in x for j in c.get("reject", []))]
            slot_share = None
            if c.get("expect_slot"):
                slot_share = sum(1 for t in r["top"][:n] if t["slot"] == c["expect_slot"]) / max(1, len(r["top"][:n]))
                if slot_share < 0.6:
                    miss.append(f"{c['expect_slot']} 비율 {slot_share:.0%}")
            good = not miss and not bad
            ok = ok and good
            was = ""
            if old:
                on = [t["name"] or "" for t in old["top"][:n]]
                was = f" (전: 기대 {sum(1 for e in c.get('expect', []) if any(e in x for x in on))}/{len(c.get('expect', []))}, 결과 {old['count']})"
            lines.append(f"{'✓' if good else '✗'} [{r['slot'] or '전체'}] {r['q']} → 결과 {r['count']} · 상위 {n}{was}"
                         + (f" · 빠짐 {miss}" if miss else "") + (f" · 나오면 안 되는 것 {bad[:5]}" if bad else "")
                         + (f" · {c['expect_slot']} {slot_share:.0%}" if slot_share is not None else ""))
            lines.append("     상위 5: " + " / ".join(names[:5]))
        elif old:
            # 동점끼리 자리만 바뀐 것은 같은 결과로 본다(상위 10 의 구성으로 비교).
            same = {t["id"] for t in old["top"][:10]} == {t["id"] for t in r["top"][:10]}
            touched = r["slot"] in changed or (r["slot"] is None and any(t["slot"] in changed for t in (old["top"][:10] + r["top"][:10])))
            if not same and not touched:
                ok = False
                lines.append(f"✗ 바뀌면 안 되는데 바뀜: [{r['slot'] or '전체'}] {r['q']}")
            elif not same:
                lines.append(f"· 바뀜(의도한 부위): [{r['slot'] or '전체'}] {r['q']} → " + " / ".join((t['name'] or '') for t in r['top'][:4]))
    print("\n".join(lines))
    print("통과" if ok else "미통과")
    return ok


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["take", "diff"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--base")
    ap.add_argument("--out")
    ap.add_argument("--changed", default="hair,face", help="이번에 의도적으로 바꾼 부위(최신 가산을 바꿨다면 weapon 도)")
    a = ap.parse_args()
    if a.mode == "take":
        take(a.base, a.out)
    else:
        sys.exit(0 if diff(a.files[0], a.files[1], set(a.changed.split(","))) else 1)

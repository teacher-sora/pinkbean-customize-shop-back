"""캡션 정본(captions/*.jsonl)의 관찰 서술을 코디 평가(/rate)가 읽는 data/describe.json 으로 옮긴다.

    python build/describe.py          # data/describe.json 을 다시 쓴다

왜 따로 두나: 평가는 핑크빈이 아이템의 '모습'을 보고 말하는 것이라, 검색용 낱말(words)보다 문장으로 된 관찰 서술이
말이 되는 감상을 만든다. 정본은 배포 이미지에 들어가지 않으므로(크다) 평가에 쓸 것만 추려 data/ 에 둔다.
캡션을 다시 찍었으면(embed_apply.py 뒤에) 이것도 다시 돌린다.

옮기는 것: 생김새를 말하는 축만. 갈래 판정, 범위, 좌우 비교, 뒷모습처럼 보는 사람이 말할 거리가 아닌 축은 뺀다.
'→ 판정' 꼬리와 칸 수(도트 측정값)는 지운다. 아이템 하나에 MAX 자까지.
"""
import glob
import io
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DROP = {
    "형태 분류", "갈래", "덮는 범위", "범위", "좌우가 같은가", "좌우", "한쪽인가 양쪽인가", "다리 덮개", "발", "뒷모습",
    "위치", "올라오는 높이", "드는 방식", "상태", "눈의 모양", "눈동자의 크기", "눈 밑", "눈꺼풀",
}
MAX = 300
CELLS = re.compile(r"(가로|세로|폭|높이)?\s*\d+\s*칸(\s*(이상|이하))?")


def clean(text: str) -> str:
    t = str(text).split("→")[0]                 # 판정 꼬리(→ 모자)는 뺀다
    t = CELLS.sub("", t)
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"\s+", " ", t).strip(" ,.;")
    return t


def describe(axes: dict) -> str:
    parts = []
    for k, v in axes.items():
        if k in DROP:
            continue
        c = clean(v)
        if not c or "적지 않음" in c:
            continue
        parts.append(f"{k}: {c}")
    out = ""
    for p in parts:                               # 축 단위로 끊는다(문장 중간에서 자르지 않는다)
        if out and len(out) + len(p) + 3 > MAX:
            break
        out = f"{out} / {p}" if out else p
    return out[:MAX]


def main():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "captions", "*.jsonl"))):
        for line in io.open(f, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            text = describe(d.get("axes") or {})
            if text:
                out[d["id"]] = text
    path = os.path.join(ROOT, "data", "describe.json")
    with io.open(path, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(out, fp, ensure_ascii=False, separators=(",", ":"))
    print(f"describe.json: {len(out)} items, {os.path.getsize(path) // 1024} KB")


if __name__ == "__main__":
    main()

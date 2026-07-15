"""
result/ 산출물(embeddings-256.json + search-docs.json)을 백엔드가 즉시 로드할 수 있는
컴팩트 아티팩트로 변환한다.

출력:
  back/data/vectors_256.f16.npy   (N x 256, float16, 정규화)  ← 코사인=내적
  back/data/meta.json             ({dim, model, count, items:[{id,slot,name,gender,grade,isCash,image_source}]})
                                   items 순서 == 벡터 행 순서 (id 로 정렬 정합 보장)

사용:
  python build/precompute.py  [RESULT_DIR]
  기본 RESULT_DIR = ../../../maple test/result  (없으면 인자로 지정)
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACK = os.path.dirname(HERE)
DEFAULT_RESULT = os.path.abspath(os.path.join(BACK, "..", "..", "maple test", "result"))

result_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULT
if not os.path.isdir(result_dir):
    sys.exit(f"[precompute] result 폴더 없음: {result_dir}")

emb_path = os.path.join(result_dir, "embeddings-256.json")
docs_path = os.path.join(result_dir, "search-docs.json")
print(f"[precompute] result = {result_dir}")

with open(emb_path, "r", encoding="utf-8") as f:
    emb = json.load(f)
with open(docs_path, "r", encoding="utf-8") as f:
    docs = json.load(f)

dim = emb.get("dim", 256)
model = emb.get("model", "text-embedding-v4")
vectors = emb["vectors"]
docs_by_id = {d["id"]: d for d in docs}
print(f"[precompute] embeddings={len(vectors)} (dim={dim}, model={model}) docs={len(docs)}")

ids, rows, items = [], [], []
missing = 0
for rec in vectors:
    _id = rec["id"]
    d = docs_by_id.get(_id)
    if d is None:
        missing += 1
        continue
    v = np.asarray(rec["v"], dtype=np.float32)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n  # 방어적 재정규화(코사인=내적 보장)
    rows.append(v)
    ids.append(_id)
    items.append({
        "id": _id,
        "slot": d.get("slot"),
        "name": d.get("name"),
        "gender": d.get("gender"),
        "grade": d.get("grade"),
        "isCash": d.get("isCash"),
        "image_source": d.get("image_source"),
    })

if missing:
    print(f"[precompute] ⚠ search-docs 에 없는 벡터 {missing}건 스킵")

mat = np.asarray(rows, dtype=np.float16)  # 저장은 f16(용량↓), 로드 시 f32 로 업캐스트
os.makedirs(os.path.join(BACK, "data"), exist_ok=True)
np.save(os.path.join(BACK, "data", f"vectors_{dim}.f16.npy"), mat)
with open(os.path.join(BACK, "data", "meta.json"), "w", encoding="utf-8") as f:
    json.dump({"dim": dim, "model": model, "count": len(items), "items": items},
              f, ensure_ascii=False)

by_slot = {}
for it in items:
    by_slot[it["slot"]] = by_slot.get(it["slot"], 0) + 1
print(f"[precompute] 저장 완료: {mat.shape} float16  ({mat.nbytes/1e6:.1f} MB)")
print(f"[precompute] 슬롯별 건수: {json.dumps(by_slot, ensure_ascii=False)}")

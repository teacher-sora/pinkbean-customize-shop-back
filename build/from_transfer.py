"""
맥 골드 캡션 산출물(transfer/) → 백엔드 아티팩트(back/data/).

입력(transfer/):
  vectors_256.f16.npy   (N,256) float16, L2 정규화. meta.json 과 행 순서 일치.
  meta.json             list[{id,slot,name,words,label,isCash,tier,model}]
  (captions.jsonl / expand_map.json 은 참고용 — 백엔드는 쓰지 않는다.
   색 변형은 프론트가 colorGroup 으로 폴딩해 대표를 보여주므로, 검색도 대표 단위로 낸다.)

하는 일:
  1) 스코프 밖 슬롯 제거 — earring/shield/skin (착용해도 거의 안 보여 검색 방해. 2026-07-17 확정)
  2) 아이템 이름의 성별 접미 (여)/(남) → gender 파생 (1=여 / 0=남 / 2=공용)
     ⚠️ 검색 성별 필터의 유일한 근거. 이름에 접미가 없으면 공용.
  3) back/data/vectors_{dim}.f16.npy + meta.json 저장(행 순서 일치 보장)

사용:  python build/from_transfer.py [TRANSFER_DIR]
기본 TRANSFER_DIR = ../../maple test/transfer
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACK = os.path.dirname(HERE)
DEFAULT_TRANSFER = os.path.abspath(os.path.join(BACK, "..", "..", "maple test", "transfer"))

src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRANSFER
if not os.path.isdir(src):
    sys.exit(f"[from_transfer] transfer 폴더 없음: {src}")

# 캡션/임베딩/검색 스코프(2026-07-17 확정): 이 12개 슬롯만.
CAPTION_SLOTS = {"hair", "face", "cap", "faceAcc", "eyeAcc",
                 "coat", "longcoat", "pants", "shoes", "glove", "cape", "weapon"}


def gender_of(name: str) -> int:
    """이름 접미로 성별 판정. 1=여, 0=남, 2=공용."""
    n = name or ""
    if "(여)" in n:
        return 1
    if "(남)" in n:
        return 0
    return 2


meta_path = os.path.join(src, "meta.json")
with open(meta_path, "r", encoding="utf-8") as f:
    items_in = json.load(f)
if isinstance(items_in, dict) and "items" in items_in:
    items_in = items_in["items"]

vec_files = [f for f in os.listdir(src) if f.startswith("vectors_") and f.endswith(".f16.npy")]
if not vec_files:
    sys.exit(f"[from_transfer] vectors_*.f16.npy 없음: {src}")
vec_path = os.path.join(src, vec_files[0])
dim = int(vec_files[0].split("_")[1].split(".")[0])

V = np.load(vec_path)
if V.shape[0] != len(items_in):
    sys.exit(f"[from_transfer] ✗ 행 수 불일치: vectors {V.shape[0]} vs meta {len(items_in)}")
print(f"[from_transfer] 입력: vectors {V.shape} ({V.dtype}) · meta {len(items_in)}")

keep_rows, items = [], []
dropped = {}
for i, d in enumerate(items_in):
    slot = d.get("slot")
    if slot not in CAPTION_SLOTS:                 # earring/shield/skin 등 스코프 밖 제거
        dropped[slot] = dropped.get(slot, 0) + 1
        continue
    keep_rows.append(i)
    items.append({
        "id": d.get("id"),
        "slot": slot,
        "name": d.get("name"),
        "words": d.get("words") or [],           # 캡션(단어 배열) — 검색 텍스트의 원천
        "gender": gender_of(d.get("name")),      # 이름 접미 기반
        "label": d.get("label"),
        "isCash": d.get("isCash"),
        "tier": d.get("tier"),
    })

mat = V[keep_rows].astype(np.float16)
# 방어적 재정규화(코사인=내적 보장)
f32 = mat.astype(np.float32)
norms = np.linalg.norm(f32, axis=1, keepdims=True)
norms[norms == 0] = 1.0
mat = (f32 / norms).astype(np.float16)

os.makedirs(os.path.join(BACK, "data"), exist_ok=True)
np.save(os.path.join(BACK, "data", f"vectors_{dim}.f16.npy"), mat)
with open(os.path.join(BACK, "data", "meta.json"), "w", encoding="utf-8") as f:
    json.dump({"dim": dim, "model": "text-embedding-v4", "count": len(items), "items": items},
              f, ensure_ascii=False)

by_slot, by_gender = {}, {0: 0, 1: 0, 2: 0}
for it in items:
    by_slot[it["slot"]] = by_slot.get(it["slot"], 0) + 1
    by_gender[it["gender"]] += 1
print(f"[from_transfer] 스코프 밖 제거: {dropped or '없음'}")
print(f"[from_transfer] 저장: {mat.shape} float16 ({mat.nbytes/1e6:.1f} MB) · items {len(items)}")
print(f"[from_transfer] 슬롯별: {json.dumps(by_slot, ensure_ascii=False)}")
print(f"[from_transfer] 성별(이름 접미): 여={by_gender[1]} 남={by_gender[0]} 공용={by_gender[2]}")

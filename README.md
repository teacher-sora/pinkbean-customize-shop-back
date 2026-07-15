# pinkbean-customize-shop-back

메이플스토리 코디샵 **AI 코디 검색** 백엔드 (1차). 코디 아이템 스프라이트를 Vision LLM으로
캡션 → `text-embedding-v4` 임베딩한 벡터를 브루트포스 코사인 검색한다.

> ⚠️ **베타/개선 중**: 캡션 QC는 전체의 ~11%만 완료(한벌옷 37%, 그 외 5~10%). Qwen3 VL
> Flash/Plus + Claude Sonnet 정정. 더 좋은 임베딩으로 점진 업그레이드 예정.

## 구조
- `data/vectors_256.f16.npy` — 15,672 × 256 float16(정규화). 코사인 = 내적.
- `data/meta.json` — 벡터 행과 동일 순서의 `{id, slot, name, gender, grade, isCash, image_source}`.
- `app.py` — FastAPI. 기동 시 벡터/메타 적재, `/search` 에서 질의 임베딩 + topK.
- `build/precompute.py` — `result/`(맥 산출물) → 위 아티팩트 생성.

## API
```
GET  /health
POST /search   { "query": "세일러복 절대영역", "slot": "longcoat"|null, "topK": 60 }
     → { results: [{ id, slot, name, grade, isCash, gender, image_source, score }], ms:{embed,total} }
```
`slot` 은 코디탭 부위(hair/face/cap/faceAcc/eyeAcc/earring/coat/longcoat/pants/shoes/glove/cape/weapon/shield).

## 로컬 실행
```bash
pip install -r requirements.txt
QWEN_API_KEY=... uvicorn app:app --port 8080 --workers 2
```

## 배포 (Fly.io)
```bash
fly deploy
fly secrets set QWEN_API_KEY=...        # DashScope 키
# 계정 리전이 본토면: fly secrets set DASHSCOPE_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```
`cpu=1` + uvicorn 워커 2개(병렬). nrt(도쿄) 리전, 512MB, 1대 예열(콜드스타트 제거).

## 벡터 재빌드 (골드 확장·재임베딩 후)
```bash
python build/precompute.py  "/path/to/result"
git add data && git commit -m "rebuild vectors" && fly deploy
```

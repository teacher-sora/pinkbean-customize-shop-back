# pinkbean-customize-shop-back

핑크빈 커마샵의 **AI 코디 검색**과 **코디 평가(핑크빈 말풍선)** 백엔드. FastAPI, Fly.io.

아이템마다 미리 써 둔 캡션(보이는 특징 단어)을 `text-embedding-v4` 로 임베딩해 두고,
질의문을 같은 공간에 임베딩해 브루트포스 코사인으로 찾는다.

## 구조
- `app.py` — 기동 시 벡터·메타를 메모리에 올린다. 엔드포인트 3개.
- `data/vectors_256.f16.npy` — 10,319 × 256 float16(L2 정규화). 코사인 = 내적.
- `data/meta.json` — `{dim, model, count, items}`. `items` 는 벡터 행과 **같은 순서**의
  `{id, slot, name, words, label, isCash, gender, tier}`.
- `data/item_colors.json` — 아이템 대표색(색이 든 질의의 2차 재정렬용).
- `build/from_transfer.py` — 캡션·임베딩 산출물을 `data/` 로 옮긴다.
- `build/localtest.py` — 임베딩 리전 확인 + 검색 스모크 테스트.

캡션 대상은 캐시 아이템 12부위다(hair, face, cap, faceAcc, eyeAcc, coat, longcoat, pants, shoes, glove, cape, weapon).

## API
```
GET  /health   → { ok, count, dim, model, slots }
POST /search   { "query": "세일러복 절대영역", "slot": "longcoat"|null, "topK": 100 }
               → { query, slot, refined, count, results: [{ id, slot, name, ... , score }], ms }
POST /rate     { "items": [{ "slot", "name", "id" }], "tone": 12, "history": ["직전 말풍선"] }
               → { bubbles: ["..."], model, dropped }
```
- `/search` 는 질의를 LLM 으로 정제(단어 나열 + 부위·성별 분리)한 뒤 임베딩한다. "리본 없는" 같은 부정어는
  코드로 파싱해 그 특징을 가진 아이템을 후보에서 뺀다.
- `/rate` 의 말풍선은 모델이 짓는다. 나온 문장을 코드로 검사해 부적절하면 버린다(프런트도 같은 규칙으로 한 번 더 본다).

## 로컬 실행
```bash
pip install -r requirements.txt
QWEN_API_KEY=... uvicorn app:app --port 8080
```

## 배포 (Fly.io)
```bash
fly deploy
fly secrets set QWEN_API_KEY=...        # DashScope 키(intl 엔드포인트)
```
도쿄(nrt), shared-cpu 1개 · 512MB · 워커 1개. 유휴 시 머신이 멈추고 요청이 오면 다시 켜진다.
배포 뒤 `/health` 의 `count` 로 벡터 수를 확인한다.

## 벡터를 바꿀 때
`data/meta.json` 의 `items` 순서와 `vectors_256.f16.npy` 의 행 순서가 반드시 같아야 한다.
신규 아이템은 뒤에 덧붙이고, 기존 아이템의 캡션을 고치면 그 행의 벡터도 다시 임베딩한다.

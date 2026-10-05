# pinkbean-customize-shop-back

핑크빈 커마샵의 **AI 코디 검색**과 **코디 평가(핑크빈 말풍선)** 백엔드. FastAPI, Fly.io.

아이템마다 미리 써 둔 캡션(보이는 특징 낱말)을 `text-embedding-v4` 로 임베딩해 두고,
질의문을 같은 공간에 임베딩해 브루트포스 코사인으로 찾는다. 캡션을 만드는 방법은 `maple test/CAPTION-METHOD.md`.

## 구조
- `app.py` — 기동 시 벡터·메타를 메모리에 올린다. 엔드포인트 3개.
- `data/vectors_256.f16.npy` — 아이템당 한 행(256차원 float16, L2 정규화). 코사인 = 내적.
- `data/vectors_short_256.f16.npy` — 짧은 캡션 벡터. `meta.short_rows` 가 각 벡터가 속한 아이템 행 번호다.
  점수는 상세 벡터와 짧은 벡터 중 높은 쪽. 왜: 질의는 6낱말 이하라 상세 캡션 하나만으로는 희석된다.
- `data/meta.json` — `{dim, model, count, items, short_rows}`. `items` 는 벡터 행과 **같은 순서**의
  `{id, slot, name, words, short?, label, isCash, gender, tier, recency?}`. `tier: "v2"` 가 새 방식 캡션이다.
- `data/vocab.json` — 낱말 스키마(이표기 → 대표어). 캡션을 정규화한 것과 같은 사전으로 질의 낱말을 바꾼다.
- `data/item_colors.json` — 아이템 대표색(색이 든 질의의 2차 재정렬용).
- `captions/{부위}.jsonl` — 캡션 정본(관찰 서술 + 측정값 + 캡션). 배포 이미지에는 들어가지 않는다.
- `build/embed_apply.py` — 정본을 `data/` 에 반영(바뀐 행만 재임베딩, 신규는 뒤에 덧붙임).
- `build/recency.py` — 아이템별 최신도를 `meta.json` 에 넣는다(id 만으로 최신을 판단하지 않는다).
- `build/snapshot.py` + `build/eval_queries.json` — 배포 전후에 같은 질의로 결과를 비교한다.
- `build/evalsearch.py` — 캡션 방식의 검색 성능을 오프라인으로 잰다.
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
`/search` 의 처리 순서
1. 질의를 LLM 으로 정제(낱말 나열 + 부위·성별 분리). 길이·위치·개수와 **인상 낱말**(차가운, 날카로운 …)은 남긴다.
2. 낱말을 `vocab.json` 으로 캡션 낱말로 바꾼다(트윈테일 → 양갈래, 쎈 → 사나운 · 날카로운 · 올라간 눈꼬리). 원문의 구를 먼저 맞추고(`웃는 눈`, `일자 눈썹`), 정제가 떨어뜨린 낱말은 되살린다.
3. 색 낱말을 뺀 나머지를 임베딩해 코사인을 낸다. 새 방식 캡션에는 **낱말 일치 가산**(질의 낱말이 캡션에 어절 경계로 그대로 있는 비율 × 0.30)을 더한다.
4. 1위 점수의 72% 에 못 미치는 것은 버린다. **낱말 일치의 단**으로도 자른다 — 가장 많이 맞은 아이템의 60% 에 못 미치게 맞은 것은 버린다(두 낱말 질의면 둘 다 맞아야 남는다). 낱말이 하나도 맞지 않으면 12개만. 색을 말했으면 그 색이 캡션에 적힌 것만. 개념 질의는 **남은 것을 전부** 돌려준다(상한 300). `topK` 는 순수 색 질의에만 쓰인다.
5. 재정렬: 이름 일치, 색(말했을 때만, 형태보다 약하게), 무기 타입, 귀의 갈래(`ear_adjust` — "동물 귀"=귀 장식 / "동물 귀 모자"=귀 달린 모자·후드), 스타킹류의 갈래(`hose_adjust` — 부위를 말하지 않으면 갈래가 양말·스타킹인 신발 먼저), 무기 종류(관찰한 `kind`), 최신도(`recency`, 상한 0.03).

규칙
- "리본 없는" 같은 부정어는 코드로 파싱해 그 특징을 가진 아이템을 후보에서 뺀다.
- 사용자가 부위 이름을 직접 말하면("한벌옷 스타킹") 그 부위로 가둔다.
- 스타킹·니삭스 질의에서 성별을 말하지 않았으면 남성 한벌옷·하의는 뺀다. `신발 포함` 이 붙은 아이템은 뒤로 보낸다.
- `/rate` 의 말풍선은 모델이 짓는다. 나온 문장을 코드로 검사해 부적절하면 버린다(프런트도 같은 규칙으로 한 번 더 본다).

## 로컬 실행
```bash
pip install -r requirements.txt
QWEN_API_KEY=... uvicorn app:app --port 8080
```

## 데이터를 바꿀 때
```bash
python build/snapshot.py take --base https://pinkbean-customize-shop-back.fly.dev --out before.json   # 운영의 지금 결과
python build/embed_apply.py --slots <바뀐 부위,…>                        # 정본 → data/ (다른 부위가 바이트 단위로 같은지 검사한다). kind·has_ears·type 도 meta 로 옮긴다. 12부위 모두 새 방식이다
python build/recency.py --times META_TIMES.txt --slots-dir <CDN slots 폴더>
QWEN_API_KEY=... uvicorn app:app --port 8080                            # 로컬에서 띄우고
python build/snapshot.py take --base http://127.0.0.1:8080 --out after.json
python build/snapshot.py diff before.json after.json --changed <바뀐 부위,…>  # "통과"가 나와야 배포한다
```
`meta.json` 의 `items` 순서와 벡터 행 순서는 반드시 같아야 한다. 신규 아이템은 뒤에 덧붙인다.
DashScope 키는 운영 검색과 같은 키다 — 임베딩 호출은 순차로, 한 번에 10건씩만 보낸다.

## 배포 (Fly.io)
```bash
fly deploy
fly secrets set QWEN_API_KEY=...        # DashScope 키(intl 엔드포인트)
```
도쿄(nrt), shared-cpu 1개 · 512MB · 워커 1개. 유휴 시 머신이 멈추고 요청이 오면 다시 켜진다.
배포 뒤 `/health` 의 `count` 를 확인하고, `snapshot.py` 로 운영 주소의 결과가 로컬과 같은지 본다.
브랜치를 병합하는 것만으로는 반영되지 않는다 — `fly deploy` 를 해야 바뀐다.

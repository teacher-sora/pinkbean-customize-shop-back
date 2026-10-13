# pinkbean-customize-shop-back

핑크빈 커마샵의 **AI 코디 검색**과 **코디 평가(핑크빈 말풍선)** 서버. FastAPI, Fly.io.

아이템마다 미리 써 둔 캡션(보이는 특징의 낱말)을 `text-embedding-v4` 로 임베딩해 두고, 질의를 같은 공간에 임베딩해 브루트포스 코사인으로 찾는다.

| 문서 | 내용 |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 구성 · API · 검색 규칙과 그 이유 · 알려진 상태 · 배포와 환경변수 |
| [CLAUDE.md](CLAUDE.md) | Claude Code 세션 지침 |
| parser 레포의 `RUNBOOK.md` §3 | 패치로 아이템이 추가됐을 때의 순서(캡션 → 임베딩 → 배포) |
| parser 레포의 `caption/CAPTION-METHOD/` | 캡션을 만드는 방법 |

세 레포([front](https://github.com/teacher-sora/pinkbean-customize-shop-front) · [parser](https://github.com/teacher-sora/pinkbean-customize-shop-parser) · back)를 **같은 폴더에 나란히** 받는다. parser 의 캡션 도구가 이 레포의 `captions/` 와 `data/vocab.json` 에 쓴다.

## 로컬 실행

Python 3.10 이상.

```bash
pip install -r requirements.txt
QWEN_API_KEY=... uvicorn app:app --port 8080        # Git Bash. PowerShell 은  $env:QWEN_API_KEY='...'; uvicorn app:app --port 8080
```

확인이 끝나면 끈다. 키 없이도 서버는 뜨지만 `/search`·`/rate` 는 실패한다.
빌드 스크립트(`build/*.py`)는 키를 환경변수 `QWEN_API_KEY` 에서, 없으면 `back/.env` 의 `QWEN_API_KEY=...` 줄에서 읽는다(`.env` 는 커밋되지 않는다).

## 데이터를 바꿀 때

```bash
python build/snapshot.py take --base https://pinkbean-customize-shop-back.fly.dev --out before.json   # 운영의 지금 결과
python build/embed_apply.py --slots <바뀐 부위,…> --dry      # 아무것도 부르지 않고 바뀔 행 수·새로 부를 글 수만 센다
python build/embed_apply.py --slots <바뀐 부위,…>            # 정본(captions/) → data/
python build/recency.py --times META_TIMES.txt --slots-dir ../parser/web/public/data/slots   # 헤어·성형·무기를 건드렸을 때
QWEN_API_KEY=... uvicorn app:app --port 8080
python build/snapshot.py take --base http://127.0.0.1:8080 --out after.json
python build/snapshot.py diff before.json after.json --changed <바뀐 부위,…>   # "통과"가 나와야 배포한다
```

`META_TIMES.txt` 를 만드는 법과 앞뒤 순서는 parser 의 `RUNBOOK.md` §3. 임베딩 호출은 운영 검색과 같은 키를 쓰므로 스크립트가 순차로 10건씩만 보낸다 — 병렬로 바꾸지 않는다.

## 배포

브랜치에 올리면 GitHub 워크플로가 배포한다(`.github/workflows/`). 손으로 `fly deploy` 하지 않는다.

| 브랜치 | 서버 | 쓰는 곳 |
|---|---|---|
| `dev` | `pinkbean-customize-shop-back-dev.fly.dev` | dev 사이트, 로컬 |
| `main` | `pinkbean-customize-shop-back.fly.dev` | 운영 사이트 |

```bash
fly secrets set QWEN_API_KEY=... -a <앱>   # 키를 바꿀 때(두 앱 모두)
```

도쿄(nrt), shared-cpu 1개, 512MB, 워커 1개. 유휴 시 머신이 멈추고 요청이 오면 다시 켜진다.
배포 뒤 `/health` 의 `count` 를 보고 검색을 한두 번 실제로 던져 본다. `main` 병합이 곧 운영 배포다.

## 브랜치

`dev` 에 커밋하고 푸시한다. `main` 병합은 지시가 있을 때만 한다.

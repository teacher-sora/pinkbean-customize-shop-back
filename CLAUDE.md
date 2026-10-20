# 핑크빈 커마샵 back — 세션 지침

AI 코디 검색·코디 평가 서버(FastAPI, Fly.io). **운영 중인 서비스다.** 운영 주소 `https://pinkbean-customize-shop-back.fly.dev`.

## 먼저 읽을 것

- [ARCHITECTURE.md](ARCHITECTURE.md) — 구성, 검색 규칙과 그 이유, **알려진 상태**(§6), 배포. 검색을 고치기 전에 §4 를 읽는다.
- 캡션은 parser 레포에서 만든다(`caption/CAPTION-METHOD/`, `ARCHITECTURE.md` §3). 패치 때의 순서는 parser 의 `RUNBOOK.md` §3.
- 세 레포(front · parser · back)는 같은 폴더에 나란히 둔다.

## 일하는 방식

- **답은 한국어로.** 진행 문구와 보고 모두. 주석·커밋 메시지도 한국어.
- **`dev` 에서 일한다.** `main` 병합은 지시가 있을 때만 한다.
- **버그는 발견하는 즉시 핫픽스로 `main`(운영)까지 올린다**(사용자 지시 2026-10-20 — 어떤 작업 중이든, 지시를 기다리지 않는다). `origin/main` 에서 브랜치를 떼어 그 수정만 커밋하고 `git merge --no-ff` 병합 커밋으로 올린 뒤, 같은 수정을 `dev` 에도 넣는다. dev 에만 있는 기능 · 디자인 작업은 섞지 않는다(그건 지금처럼 지시가 있을 때만). 올리기 전에 dev 데이터를 지우고(아래 '시험 데이터'), 보고에 '버그라서 바로 올렸다'고 적는다. 운영 서버는 main 에 올리면 자동 배포된다(아래).
- **dev 서버는 `dev` 브랜치에 올리면 자동으로 배포된다**(`.github/workflows/deploy-dev.yml`). dev 사이트가 그 서버를 쓰므로, 고친 것은 dev 사이트에서 확인한다.
- **운영 서버는 `main` 에 올리면 자동으로 배포된다**(`.github/workflows/deploy-main.yml`). `main` 병합이 곧 운영 배포이므로 병합은 지시가 있을 때만 한다. 손으로 `fly deploy` 하지 않는다(작업 폴더가 그대로 운영에 올라가 저장소와 어긋난다).
- **검색 규칙·숫자를 바꾸면 전후를 비교한다.** `build/snapshot.py` 로 운영과 로컬에 같은 질의를 던져 "통과"가 나와야 배포한다. 바꾼 숫자는 ARCHITECTURE §4 에도 고쳐 적는다.
- **고쳤다는 보고는 실제 결과로.** `/health` 의 count 만 보지 말고 검색을 실제로 던져 본다.
- 로컬 서버는 8080 으로 띄우고 끝나면 끈다.
- 임시 산출물(`before.json`·`after.json`·`META_TIMES.txt`)은 끝나면 지운다.
- **저장소를 깨끗하게 둔다(사용자 지시).** 바꾸거나 없앤 것의 흔적을 남기지 않는다. 쓰지 않게 된 함수 · 타입 · 파일 · 설정 · 스크립트는 그 작업 안에서 지운다. 같은 역할을 하는 것이 둘이 되면 최신 하나로 합치고 옛것을 지운다. 끝내기 전에 이번에 건드린 이름이 아직 쓰이는지 찾아본다. 왜: 남은 옛것이 다음 작업에서 정본처럼 다시 쓰인다.

## 시험 데이터 — Supabase 에 남기지 않는다

이 레포는 Supabase 를 쓰지 않지만, 작업을 확인하느라 **사이트(운영 · dev)를 여는 순간 Supabase 에 기록이 생긴다.** Supabase(프로젝트 `pinkbean-customize`)에는 운영 데이터만 남긴다.

- **사이트를 열기만 해도 익명 계정이 하나 생긴다**(코디 광장이 첫 화면 뒤에 세션을 만든다). 자동화 브라우저나 스크립트로 열 때는 **`*.supabase.co/auth/` 요청을 막고 연다**:
  ```js
  await page.route((u) => u.host.endsWith('supabase.co') && u.pathname.startsWith('/auth/'), (r) => r.abort())
  ```
  막아도 목록 · 미리보기 · 검색 · 광장 읽기는 그대로 된다. front 레포의 `pinkbean-screencheck`(`shots.mjs`)는 기본으로 막는다.
- **글 · 댓글 · 좋아요를 만드는 시험은 dev 사이트에서만 한다.** 운영 사이트에서는 읽기만 한다.
- **`main` 에 병합할 때마다 dev 데이터를 지운다**: front 레포에서 `node scripts/clean-test-data.mjs` 로 세어 보고 `--apply` 로 지운다(dev 광장의 모든 행 · dev 그림 · 자동화로 생긴 익명 계정. 운영 데이터는 그대로. 키는 `SUPABASE_TOKEN`). 병합 없이 끝나는 작업도 계정이나 글을 만들었으면 끝에 돌린다.
- front 레포나 키가 없어 지우지 못했으면 **무엇이 남았는지 보고한다** — 남긴 채로 끝났다고 하지 않는다.

## 하지 말 것

- **임베딩 호출을 병렬로 바꾸지 않는다.** 키가 운영 검색과 같다 — 한도를 넘기면 운영 검색이 실패한다(순차, 10건씩).
- **`data/meta.json` 의 `items` 순서를 바꾸지 않는다.** 벡터 행과 1:1 이다. 신규는 뒤에 덧붙인다.
- **한 부위 안에 다른 방식의 캡션을 섞지 않는다.** 다시 찍을 때는 그 부위 전량을 한 번에.
- **비밀값을 출력하지 않는다**(`QWEN_API_KEY`, `.env`, `fly secrets` 의 값).
- Fly 계정이 `sora05153@gmail.com` 이 아니면(`fly auth whoami`) 우회하지 말고 사용자에게 알린다.
- 질의 정제 지시문(`REFINE_SYSTEM`)에 캡션 예시나 낱말 목록을 넣지 않는다(모델이 베껴 넣는다).
- 말풍선 금지 낱말을 고칠 때는 front 의 `src/lib/safeText.ts` 와 함께 고친다.
- ARCHITECTURE §9 의 "하지 않기로 한 것"을 묻지 않고 되살리지 않는다.

## 받은 뒤 한 번

```bash
pip install -r requirements.txt
git config user.name teacher-sora
git config user.email sora05153@gmail.com
```

**GitHub 작업은 `teacher-sora` 계정으로 한다.** 이 PC 에는 GitHub 계정이 여럿이다. push 전에 `gh auth status` 로 지금 활성 계정을 적어 두고, `teacher-sora` 가 아니면 `gh auth switch -u teacher-sora` 로 바꿔 작업한 뒤, **끝나면 `gh auth switch -u <적어 둔 계정>` 으로 원래대로 되돌린다**(사용자 지시). `git push` 가 403 / Repository not found 로 실패하는 것도 대개 활성 계정이 다른 탓이다.

## 문서를 고칠 때

규칙이 바뀌면 `ARCHITECTURE.md` 의 **그 자리를 고쳐 쓴다.** 이력·날짜·옛 규칙을 쌓지 않는다(이력은 git 이 맡는다). 새 규칙에는 왜를 한 줄 붙인다.

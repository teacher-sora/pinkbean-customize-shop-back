"""
pinkbean-customize-shop-back — 코디 아이템 벡터 검색 API (1차)

- 사전연산 벡터(vectors_256.f16.npy, 정규화) + 메타(meta.json)를 기동 시 메모리에 적재.
- POST /search : 질의문을 text-embedding-v4(DashScope)로 임베딩 → 코사인(내적) topK → 아이템 리스트.
- 코사인 = 내적(양쪽 정규화). 15,672 x 256 브루트포스 matmul 이라 수 ms 수준(매우 빠름).
- cpu=1 + uvicorn 워커 다중화로 동시요청/임베딩 I/O 병렬.

품질 주의: 캡션 QC 11%, Qwen3 VL Flash/Plus + Sonnet 정정. 아직 개선 중(프론트에 '베타' 표기).
"""
import os
import re
import json
import time
import random
import numpy as np
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DIM = int(os.environ.get("EMBED_DIM", "256"))

QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
# DashScope OpenAI-호환 임베딩 엔드포인트. 계정 리전에 따라 intl / 본토(cn) 로 바꿀 수 있게 env 로.
DASHSCOPE_BASE = os.environ.get(
    "DASHSCOPE_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-v4")
RATE_MODEL = os.environ.get("RATE_MODEL", "qwen-flash")  # 코디 평가용 저가·고속 텍스트 모델

# 핑크빈 페르소나 — 코디를 보고 그 자리에서 툭 던지는 혼잣말/감상(점수·평가 아님).
PINKBEAN_SYSTEM = (
    "너는 메이플스토리의 인기 마스코트 '핑크빈'이야. 천진난만하고 장난스럽고, 조금 떼쓰고 자기중심적이지만 "
    "사랑스러운 어린아이 같은 성격이야. 호기심 많고, 가끔은 먹을 것에도 관심을 보여.\n"
    "말투 느낌 예시: \"심심해! 새로운 게 하나도 없잖아!\" / \"꼭 분홍색이어야 해? 내 맘대로 바꿀 거야!\" / "
    "\"흥, 이름값 같은 게 뭐가 중요해? 재밌으면 그만이지!\" / \"우와, 이 옷 어디서 났어? 나도 입어보고 싶다!\"\n"
    "사용자가 꾸민 캐릭터의 코디(착용 아이템)를 보고, 핑크빈이 그 자리에서 문득 떠오른 대로 '툭 던지는 혼잣말·감상'을 해줘. "
    "점수를 매기거나 심사하듯 평가하는 게 아니라, 코디를 구경하다가 자연스럽게 튀어나오는 반응이야.\n"
    "반드시 JSON 으로만 답해: {\"bubbles\": [\"...\", \"...\"]}\n"
    "- bubbles 개수는 사용자 메시지에 적힌 수를 **정확히** 지켜라.\n"
    "- 각 bubble 은 뜻이 자연스럽게 통하는 '완성된 한 문장'. 앞뒤 안 맞는 말이나 어색한 비유(예: 맛있겠다 했다가 맛없겠다) 금지.\n"
    "- 먹는 것에 억지로 갖다붙이지 말 것. 정말 어울릴 때만 가볍게.\n"
    "- 너무 길지 않게(공백 포함 35자 이내), 서로 다른 내용으로.\n"
    "- ⚠️ '뀨', '부농부농' 같은 의성어·말버릇은 남발 금지. 보통은 안 쓰고, 아주 가끔 최대 1개만.\n"
    "- ⚠️ **반말로만** 말해라(핑크빈은 어린아이다). '~요', '~니다' 같은 존댓말 금지.\n"
    "- ★ 아이템마다 '생김새'가 함께 적혀 있으면 **그 생김새를 보고** 반응해라(이름만 보고 넘겨짚지 말 것).\n"
    "- ★ 매번 **다른 곳에 눈이 가야 한다.** 머리·옷·신발·무기·장식 등 여러 부위 중 이번엔 무엇이 눈에 띄었는지 골라서 말해라.\n"
    "- ★ '방금 한 말'이 주어지면 **그것과 겹치는 소재·표현은 피하고** 새로운 얘기를 해라."
)

# ── 데이터 적재(모듈 로드 = 워커별 1회) ─────────────────────────────
_t0 = time.time()
MAT = np.load(os.path.join(DATA, f"vectors_{DIM}.f16.npy")).astype(np.float32)  # (N, DIM)
with open(os.path.join(DATA, "meta.json"), "r", encoding="utf-8") as f:
    META = json.load(f)
ITEMS = META["items"]  # 벡터 행과 동일 순서
N = MAT.shape[0]
# 슬롯 → 행 인덱스(필터 검색 가속)
SLOT_ROWS: dict[str, np.ndarray] = {}
for i, it in enumerate(ITEMS):
    SLOT_ROWS.setdefault(it["slot"], []).append(i)
SLOT_ROWS = {k: np.asarray(v, dtype=np.int64) for k, v in SLOT_ROWS.items()}
# 성별 = 아이템 "이름"의 접미 (여)/(남) 가 유일한 근거(build/from_transfer.py 에서 파생). 0남/1여/2공용.
GENDERS = np.asarray([it.get("gender") if it.get("gender") is not None else 2 for it in ITEMS], dtype=np.int64)
# 헤어·성형 저품질(저ID=구형 디자인) 완만 억제: 슬롯 내 ID를 0~1로 정규화해 높은 ID(신형)에 소폭 가산점.
# 캡션 유사도가 우선이라 상한을 낮게 둔다(동점~근접 구간에서만 신형이 위로).
ID_BONUS_MAX = 0.05
ID_BONUS = np.zeros(len(ITEMS), dtype=np.float32)
# 신형(높은 ID) 완만 우대. **순위(백분위)** 정규화 → ID 간격이 들쭉날쭉하거나 이상치(무기: 대부분
# 170xxxx인데 투명블레이드 134xxxx)가 있어도 왜곡 없이 오래된(낮은 ID) 아이템을 확실히 아래로.
# hair/face/weapon 적용(구형 절판 무기 '장난감 총'·'모형 라이플' 후순위화).
for _slot in ("hair", "face", "weapon"):
    _rows = SLOT_ROWS.get(_slot)
    if _rows is None or len(_rows) < 2:
        continue
    try:
        _ids = [int(ITEMS[i]["id"]) for i in _rows.tolist()]
    except (TypeError, ValueError):
        continue
    _order = sorted(range(len(_rows)), key=lambda k: _ids[k])
    _n = len(_rows)
    for _rank, _k in enumerate(_order):
        ID_BONUS[_rows[_k]] = np.float32(ID_BONUS_MAX * _rank / (_n - 1))

# ── 색-매칭 점수 ────────────────────────────────────────────────────
# 질의에 색이 있으면 아이템 "대표색"(아이콘 픽셀 면적최다색; 없으면 캡션 첫 색토큰)이
# 질의색 계열이면 가산, 아니면 감점. 임베딩이 색을 잘 못 가르는 걸 보정.
COLOR_CANON = {
    '빨강': '빨강', '빨간': '빨강', '붉은': '빨강', '레드': '빨강',
    '주황': '주황', '오렌지': '주황', '갈색': '갈색', '브라운': '갈색',
    '노랑': '노랑', '노란': '노랑', '옐로': '노랑',
    '연두': '연두', '초록': '초록', '녹색': '초록', '그린': '초록',
    '청록': '청록', '민트': '청록',
    '파랑': '파랑', '파란': '파랑', '푸른': '파랑', '블루': '파랑',
    '하늘': '하늘색', '하늘색': '하늘색', '하늘빛': '하늘색',
    '남색': '남색', '네이비': '남색',
    '보라': '보라', '보라색': '보라', '퍼플': '보라', '자주': '자주색', '자주색': '자주색',
    '분홍': '분홍', '분홍색': '분홍', '핑크': '분홍',
    '검정': '검정', '검은': '검정', '까만': '검정', '블랙': '검정',
    '하양': '흰색', '하얀': '흰색', '흰': '흰색', '흰색': '흰색', '화이트': '흰색',
    '회색': '회색', '그레이': '회색', '베이지': '베이지색', '베이지색': '베이지색',
}
_FAMILY = {
    '빨강': {'빨강', '분홍', '자주색'}, '분홍': {'분홍', '빨강', '자주색'}, '자주색': {'자주색', '보라', '빨강'},
    '주황': {'주황', '갈색', '노랑'}, '갈색': {'갈색', '주황'}, '노랑': {'노랑', '주황', '연두'},
    '연두': {'연두', '초록', '노랑'}, '초록': {'초록', '연두', '청록'}, '청록': {'청록', '초록', '파랑', '하늘색'},
    '파랑': {'파랑', '하늘색', '남색', '청록'}, '하늘색': {'하늘색', '파랑', '청록'}, '남색': {'남색', '파랑', '보라'},
    '보라': {'보라', '자주색', '남색'},
    '검정': {'검정', '회색'}, '흰색': {'흰색', '회색', '베이지색'}, '회색': {'회색', '검정', '흰색'},
    '베이지색': {'베이지색', '흰색', '갈색'},
}


def canon_colors(text: str) -> set:
    return {c for tok, c in COLOR_CANON.items() if tok in (text or "")}


# 아이템 대표색: item_colors.json(아이콘 면적최다색, ratio>=0.28) 우선, 없으면 캡션 첫 색토큰.
try:
    with open(os.path.join(DATA, "item_colors.json"), "r", encoding="utf-8") as f:
        _ITEM_COLORS = json.load(f)
except (OSError, ValueError):
    _ITEM_COLORS = {}


def _primary_color(it):
    c = _ITEM_COLORS.get(it["id"]) or {}
    if c.get("ratio", 0) >= 0.28:
        return c.get("dom")
    for w in (it.get("words") or []):
        cc = canon_colors(w)
        if cc:
            return next(iter(cc))
    return None


ITEM_PRIMARY = [_primary_color(it) for it in ITEMS]
COLOR_BONUS, COLOR_PEN = 0.15, 0.12
# 거리 임계: 개념 질의에서 top1 코사인의 이 비율 미만인 후보는 "무관"으로 컷(개수 채우기 방지).
# 쿼리마다 코사인 절대값이 달라 상대비율이 안전. 0.72 = 최상위와 비슷하게 가까운 것만 남김.
DIST_RATIO = 0.72
# 전체 검색 슬롯-관련도 게이트: 슬롯 최상위가 전역 최상위의 이 비율 이상일 때만 그 슬롯을 포함.
# 교차슬롯 개념(스타킹: 바지/한벌옷/신발 모두)은 여러 슬롯이 통과, 단일슬롯 개념(총: 무기만)은 하나만.
SLOT_GATE = 0.62

# ── 무기 타입-매칭 점수 ─────────────────────────────────────────────
# 무기의 정체성 = 타입(총/검/창/활...). 임베딩이 타입을 잘 못 가르므로, 캡션 형태토큰으로
# 계열을 뽑아 타입 질의 시 일치 가산/불일치 감점. (색-매칭과 동일 패턴)
_WTYPE = {
    '총': '총', '엽총': '총', '런처': '총', '새총': '총', '물총': '총', '라이플': '총', '레이저건': '총',
    '작살총': '총', '슈터': '총', '건': '총', '대포': '총', '머스킷': '총', '권총': '총', '개틀링': '총',
    '검': '검', '대검': '검', '쌍검': '검', '단검': '검', '세이버': '검', '블레이드': '검', '소드': '검',
    '도': '검', '태도': '검', '장검': '검', '칼': '검', '레이피어': '검', '에너지소드': '검', '광선검': '검',
    '창': '창', '폴암': '창', '랜스': '창', '삼지창': '창', '표창': '창', '스피어': '창',
    '활': '활', '석궁': '활', '보우': '활', '쇠뇌': '활', '크로스보우': '활',
    '도끼': '도끼', '낫': '낫',
    '둔기': '둔기', '망치': '둔기', '해머': '둔기', '몽둥이': '둔기', '방망이': '둔기', '메이스': '둔기',
    '지팡이': '지팡이', '스태프': '지팡이', '완드': '지팡이', '봉': '지팡이', '로드': '지팡이', '고서': '책', '책': '책',
    '부채': '부채', '오브': '오브', '너클': '너클', '건틀렛': '너클', '채찍': '채찍', '아대': '아대',
}
_WTYPE_QUERY = {'총': '총', '권총': '총', '엽총': '총', '검': '검', '칼': '검', '창': '창', '활': '활',
                '도끼': '도끼', '낫': '낫', '둔기': '둔기', '지팡이': '지팡이', '스태프': '지팡이',
                '너클': '너클', '부채': '부채', '석궁': '활'}


def _weapon_type(it):
    if it.get("slot") != "weapon":
        return None
    words = it.get("words") or []
    for w in ([words[0]] + words if words else []):  # 첫 토큰(형태) 우선
        for k, fam in _WTYPE.items():
            if w == k or w.endswith(k):
                return fam
    return None


def weapon_query_type(text: str):
    for k, fam in _WTYPE_QUERY.items():
        if k in (text or ""):
            return fam
    return None


ITEM_WTYPE = [_weapon_type(it) for it in ITEMS]
WTYPE_BONUS, WTYPE_PEN = 0.20, 0.15
# 검색 스코프(2026-07-17 확정): 아래 12개 슬롯만. earring(귀고리)·shield(방패)·skin(피부)는
# 착용해도 거의 안 보여 검색 방해라 제외. 스코프는 build/from_transfer.py 가 이미 강제하지만,
# 잘못된 데이터가 들어와도 스코프 밖이 새어나오지 않도록 방어적으로 한 번 더 건다.
CAPTION_SLOTS = {"hair", "face", "cap", "faceAcc", "eyeAcc",
                 "coat", "longcoat", "pants", "shoes", "glove", "cape", "weapon"}
SCOPE_MASK = np.asarray([it.get("slot") in CAPTION_SLOTS for it in ITEMS], dtype=bool)
SCOPE_ROWS = np.nonzero(SCOPE_MASK)[0]
# id → 아이템(캡션 words 조회용). 코디 평가가 "이름"이 아니라 "생김새"를 보고 말하게 한다.
BY_ID = {it["id"]: it for it in ITEMS}

# ── 이름 검색용 정규화 인덱스 ────────────────────────────────────────
# 검색 근거는 "캡션(=벡터)" + "아이템 이름(=아래 부분일치)" 딱 두 가지뿐이다.
_GENDER_SUFFIX = re.compile(r"\((?:남|여)\)")
_NON_WORD = re.compile(r"[^0-9a-z가-힣]+")


def norm_text(s: str) -> str:
    """공백·기호 제거 + 소문자화. 이름/질의 부분일치 비교용."""
    return _NON_WORD.sub("", (s or "").lower())


# 이름에서 성별 접미를 떼고 정규화(성별은 별도 필터로 처리하므로 이름 매칭에선 무시).
NAMES_NORM = [norm_text(_GENDER_SUFFIX.sub("", it.get("name") or "")) for it in ITEMS]
NAME_BONUS = 0.35  # 이름 부분일치 가산점(코사인 위에 더함). 이름으로 찾으면 확실히 위로 올라오게.

print(f"[app] loaded {N} vectors dim={DIM} in {time.time()-_t0:.2f}s · slots={list(SLOT_ROWS)} · scope={int(SCOPE_MASK.sum())}")

# 질의문 성별 의도 감지. 안전한 토큰만(‘여우/여름/남색’ 오탐 방지 위해 단일 ‘여/남’ 은 제외).
_FEMALE = ["여자", "여성", "여캐", "여아", "여자아이", "여캐릭", "걸즈", "소녀"]
_MALE = ["남자", "남성", "남캐", "남아", "남자아이", "남캐릭", "소년"]


def detect_gender(q: str):
    """(allowed_genders|None, 성별토큰 제거한 질의) 반환.

    여자 질의 → {1,2}: 이름에 (여) 인 것 + 성별 표기 없는 공용. **(남) 은 절대 안 나온다.**
    남자 질의 → {0,2}: 반대. 성별 의도가 없으면 None(전체).
    """
    allowed = None
    cleaned = q
    if any(w in q for w in _FEMALE):
        allowed = {1, 2}
        for w in _FEMALE:
            cleaned = cleaned.replace(w, " ")
    elif any(w in q for w in _MALE):
        allowed = {0, 2}
        for w in _MALE:
            cleaned = cleaned.replace(w, " ")
    return allowed, " ".join(cleaned.split()) or q


# ── 질의 정제(LLM) ───────────────────────────────────────────────────
# 왜 필요한가: 아이템은 "보이는 특징을 단어로만 나열한" 캡션으로 색인돼 있는데(예 ["단발","일자앞머리"]),
# 사용자는 문장으로 검색한다("한벌옷에 스타킹 달린거"). 형태가 다르면 임베딩이 어긋난다.
# 게다가 부위를 말해도("신발") 슬롯 필터로 쓰이지 않아 엉뚱한 슬롯이 1등으로 나왔다.
# → 질의를 캡션과 같은 형태(단어 나열)로 바꾸고, 부위/성별은 필터로 분리한다.
REFINE_MODEL = os.environ.get("REFINE_MODEL", "qwen-flash")  # 저가·고속
# ⚠️ 이 프롬프트에 "캡션 예시"를 넣지 말 것. 넣었더니 모델이 그 예시 단어를 그대로 베껴서
# 사용자가 말하지도 않은 특징을 질의에 주입했다(예: '세갈래' 질의에 '단발·끝플립'이 붙음).
# 캡션 규칙과 같은 함정 — 예시를 주면 그 틀에 갇힌다. 형식만 설명하고 예시는 주지 않는다.
REFINE_SYSTEM = (
    "너는 메이플스토리 아이템 검색의 질의 전처리기다.\n"
    "아이템은 '눈에 보이는 시각 특징을 짧은 단어로만 나열한' 캡션으로 색인돼 있다. "
    "사용자 문장을 그 캡션과 **같은 형태(시각 특징 단어 나열)**로 바꾸는 것이 네 일이다.\n"
    "반드시 JSON 으로만 답해: {\"words\":[...],\"slot\":\"<슬롯|null>\",\"gender\":\"f|m|null\"}\n"
    "규칙:\n"
    "- ★ **사용자가 말한 것만 남긴다. 말하지 않은 특징을 절대 추가하지 마라.**\n"
    "  (길이·색·모양 등을 사용자가 언급하지 않았다면 네가 상상해서 넣지 말 것.)\n"
    "- words = 사용자가 말한 '생김새' 단어만. 조사·어미·군더더기(달린거, 같은, 느낌, 찾아줘, 추천, 예쁜)는 버린다.\n"
    "- 주관·감상(청순한, 귀여운, 힙한)은 생김새가 아니므로 버린다.\n"
    "- 부위를 가리키는 말은 words 에 넣지 말고 slot 으로 뺀다.\n"
    "  slot 값: hair(헤어/머리) face(성형/눈) cap(모자/탈/투구) faceAcc(얼굴장식) eyeAcc(눈장식/안경)\n"
    "  coat(상의) longcoat(한벌옷/원피스/드레스) pants(하의/바지/치마) shoes(신발) glove(장갑)\n"
    "  cape(망토) weapon(무기). 부위 언급이 없으면 null.\n"
    "- 성별을 가리키는 말(여자/여성/남자/남성…)은 words 에 넣지 말고 gender(f/m)로 뺀다. 없으면 null.\n"
    "- ★ **동의어·유사어를 새로 만들어 넣지 마라.** 사용자가 쓴 낱말을 그대로 쓴다"
    "(억지 확장이 오히려 엉뚱한 결과를 부른다).\n"
    "- words 는 최대 6개. 부정 표현('~없는', '~아닌')은 통째로 버린다.\n"
    "- 아이템 이름으로 보이면 그 이름을 words 에 그대로 남긴다."
)


# 부위를 가리키는 말은 words 에 남으면 임베딩을 오염시킨다(캡션엔 부위명이 안 들어있으므로).
# LLM 이 가끔 slot 으로 빼놓고도 words 에 남겨서("스타킹 한벌옷") 결과가 망가졌다 → 코드로 확실히 제거.
# 유추된 그 슬롯의 표현만 지운다(예: slot=longcoat 일 때만 '한벌옷' 제거) → 특징어를 잘못 지우지 않는다.
SLOT_WORDS = {
    "hair": ("헤어", "머리"), "face": ("성형",), "cap": ("모자",), "faceAcc": ("얼굴장식",),
    "eyeAcc": ("눈장식", "안경"), "coat": ("상의",), "longcoat": ("한벌옷", "원피스", "드레스"),
    "pants": ("하의", "바지"), "shoes": ("신발",), "glove": ("장갑",), "cape": ("망토",),
    "weapon": ("무기",),
}


async def refine_query(q: str):
    """문장 → (words, slot|None, gender|None). 실패하면 None (호출부가 규칙 기반으로 폴백)."""
    try:
        q_sorted = " ".join(sorted(q.split()))  # 어순 불변: 정렬해 LLM 입력을 정규화("여자 단발 헤어"="헤어 여자 단발")
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.post(
                f"{DASHSCOPE_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
                json={
                    "model": REFINE_MODEL,
                    "messages": [{"role": "system", "content": REFINE_SYSTEM},
                                 {"role": "user", "content": q_sorted}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0,
                    "max_tokens": 200,
                },
            )
            r.raise_for_status()
            d = json.loads(r.json()["choices"][0]["message"]["content"])
        words = [str(w).strip() for w in (d.get("words") or []) if str(w).strip()][:6]
        slot = d.get("slot") if d.get("slot") in CAPTION_SLOTS else None
        g = d.get("gender")
        gender = {1, 2} if g == "f" else {0, 2} if g == "m" else None
        if slot:  # 유추된 슬롯의 부위어가 words 에 남아있으면 제거(임베딩 오염 방지)
            drop = SLOT_WORDS.get(slot, ())
            pruned = [w for w in words if w not in drop]
            if pruned:  # 전부 지워지면(부위만 말한 질의) 원래 words 유지
                words = pruned
        if not words:
            return None
        return words, slot, gender
    except Exception:
        return None


def name_bonus_for(rows: np.ndarray, cleaned_query: str) -> np.ndarray:
    """이름 부분일치 가산점. 정규화한 질의가 이름에 통째로 들어가면 가산.

    토큰 부분점수는 주지 않는다 — '헤어' 같은 흔한 토큰이 전부에 붙어 노이즈가 되기 때문.
    (예: '윈디 헤어' → '윈디헤어' ⊂ '윈디 헤어 (여)' → 가산 / '단발 헤어' → 이름에 없으면 캡션 벡터로만 판정)
    """
    q = norm_text(cleaned_query)
    out = np.zeros(len(rows), dtype=np.float32)
    if len(q) < 2:
        return out
    for k, r in enumerate(rows):
        nm = NAMES_NORM[r]
        if nm and q in nm:
            out[k] = 1.0
    return out

app = FastAPI(title="pinkbean-customize-shop-back", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 무자격 POST 검색 API → 전체 허용(자격증명 없음)
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchReq(BaseModel):
    query: str
    slot: str | None = None      # 특정 슬롯(hair/cap/weapon...)만 검색. None=전체
    topK: int = 100


async def embed_query(text: str) -> np.ndarray:
    """질의문을 문서와 동일 모델·차원으로 임베딩(정규화 반환)."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            f"{DASHSCOPE_BASE}/embeddings",
            headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
            json={"model": EMBED_MODEL, "input": text, "dimensions": DIM,
                  "encoding_format": "float"},
        )
        r.raise_for_status()
        v = np.asarray(r.json()["data"][0]["embedding"], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


@app.get("/health")
def health():
    return {"ok": True, "count": N, "dim": DIM, "model": EMBED_MODEL,
            "slots": {k: int(len(v)) for k, v in SLOT_ROWS.items()}}


@app.post("/search")
async def search(req: SearchReq):
    q = (req.query or "").strip()
    if not q:
        return {"query": q, "count": 0, "results": []}
    t0 = time.time()
    # ① LLM 정제: 문장 → 캡션과 같은 형태(단어 나열) + 부위/성별 분리. 실패하면 규칙 기반으로 폴백.
    ref = await refine_query(q)
    if ref:
        words, ref_slot, allowed_g = ref
    else:
        ref_slot = None
        allowed_g, _cq = detect_gender(q)  # 폴백: 성별 토큰만 제거
        words = _cq.split()
    # 정체성 = 개념(명사)이지 색이 아니다. 임베딩은 **색을 뺀 개념**만으로 → 색이 개념을 덮지 않게.
    # 색은 아래에서 개념 매치 안의 2차 재정렬로만 반영. 어순 불변을 위해 정렬해 임베딩.
    qcolors = canon_colors(" ".join(words)) | canon_colors(q)
    concept_words = [w for w in words if not canon_colors(w)]
    embed_words = concept_words or words           # 순수 색 질의면 색 자체로 임베딩
    cleaned = " ".join(sorted(embed_words))
    color_strong = not concept_words               # 개념 없이 색만 → 색 강하게
    t_refine = time.time() - t0

    t1 = time.time()
    qv = await embed_query(cleaned)
    t_embed = time.time() - t1

    # 후보는 항상 스코프 안에서. 부위는 **사용자가 고른 것(req.slot)만** 하드 제한한다.
    # 유추 슬롯(ref_slot)으로는 제한하지 않는다 — "전체"에서 '스타킹'을 치면 바지로 갇혀 한벌옷/신발
    # 스타킹이 사라지던 문제 때문. 대신 전체 검색은 아래 슬롯-관련도 게이트로 관련 슬롯만 남긴다.
    slot = req.slot if (req.slot and req.slot in SLOT_ROWS) else None
    has_slot = bool(slot)
    rows = SLOT_ROWS[slot][SCOPE_MASK[SLOT_ROWS[slot]]] if has_slot else SCOPE_ROWS
    if allowed_g is not None:
        rows = rows[np.isin(GENDERS[rows], list(allowed_g))]
    if len(rows) == 0:
        return {"query": q, "slot": slot, "count": 0, "results": []}
    # 점수 = 캡션 벡터 코사인 + 이름 부분일치 가산점 (검색 근거는 이 둘뿐)
    # 이름은 정제문·원문 **둘 다**로 본다 — 정제가 이름을 쪼개거나 부위를 떼어내도 이름 검색이 죽지 않도록.
    base = MAT[rows] @ qv  # 순수 개념 코사인(거리)
    nb = np.maximum(name_bonus_for(rows, cleaned), name_bonus_for(rows, detect_gender(q)[1]))
    wq = weapon_query_type(cleaned)  # 무기 타입 질의(총/검/창...) — 색토큰이 빠진 cleaned 기준
    # 무기 타입 가산을 "관련도"에 미리 반영: 총류는 임베딩 코사인이 낮아도 거리컷에 안 잘리게.
    rel = base + (np.fromiter(
        ((WTYPE_BONUS if ITEM_WTYPE[r] == wq else (-WTYPE_PEN if ITEM_WTYPE[r] else 0.0))
         for r in rows.tolist()), dtype=np.float32, count=len(rows)) if wq else 0.0)
    # ★ 거리 임계: 개수를 채우지 말고 "가까운 것만" 남긴다. 개념 질의는 top1 대비 상대임계로 무관한 꼬리를 컷.
    #   전체(슬롯 미지정)는 **슬롯별** top1 기준 → 한 슬롯이 전역을 독점해도 다른 슬롯 상위가 살아남음
    #   (예: 전체 '스타킹'에서 바지뿐 아니라 한벌옷/신발 스타킹도). 이름매칭 항상 포함. 순수 색 질의는 컷 안 함.
    if concept_words and len(rel):
        keep = (nb > 0)
        if has_slot:
            keep = keep | (rel >= float(rel.max()) * DIST_RATIO)
        else:
            gtop = float(rel.max())
            slot_arr = np.array([ITEMS[r]["slot"] for r in rows.tolist()])
            for s in set(slot_arr.tolist()):
                m = slot_arr == s
                smax = float(rel[m].max())
                if smax >= gtop * SLOT_GATE:   # 이 슬롯 최상위가 전역 최상위와 견줄 만할 때만 슬롯 포함
                    keep = keep | (m & (rel >= smax * DIST_RATIO))    # 포함된 슬롯 안에서 상대임계
        if keep.any():
            rows = rows[keep]
            base = MAT[rows] @ qv
            nb = np.maximum(name_bonus_for(rows, cleaned), name_bonus_for(rows, detect_gender(q)[1]))
    scores = base + NAME_BONUS * nb
    # 남성 스타킹 후순위: '스타킹'류는 여성 연상어. 성별 지정이 없으면 남성(gender==0) 아이템을 소폭 감점.
    if allowed_g is None and any(h in cleaned for h in ("스타킹", "타이츠", "팬티스타킹")):
        scores = scores - 0.05 * (GENDERS[rows] == 0)
    # 헤어·성형: 신형(높은 ID) 완만 우대.
    scores = scores + ID_BONUS[rows]
    # 무기 타입-매칭: 정체성=타입. 계열 일치 가산/불일치 감점(총→총류, 검→검류...).
    if wq:
        scores = scores + np.fromiter(
            ((WTYPE_BONUS if ITEM_WTYPE[r] == wq else (-WTYPE_PEN if ITEM_WTYPE[r] else 0.0))
             for r in rows.tolist()), dtype=np.float32, count=len(rows))
    # 색-매칭: 색은 2차. 개념+색이면 약하게, 순수 색이면 강하게. 대표색 계열 일치 가산/불일치 감점.
    if qcolors:
        qfam = set().union(*[_FAMILY.get(c, {c}) for c in qcolors])
        cb, cp = (COLOR_BONUS, COLOR_PEN) if color_strong else (0.08, 0.06)
        adj = np.fromiter(
            ((cb if (ITEM_PRIMARY[r] in qfam) else (-cp if ITEM_PRIMARY[r] else 0.0))
             for r in rows.tolist()),
            dtype=np.float32, count=len(rows),
        )
        scores = scores + adj
    k = min(req.topK, len(rows))
    order = np.argsort(-scores)[:k]
    top, top_scores = rows[order], scores[order]

    results = []
    for row, sc in zip(top.tolist(), np.asarray(top_scores).tolist()):
        it = ITEMS[row]
        results.append({
            "id": it["id"], "slot": it["slot"], "name": it["name"],
            "label": it.get("label"), "isCash": it.get("isCash"), "gender": it.get("gender"),
            "words": it.get("words") or [], "tier": it.get("tier"),
            "score": round(float(sc), 4),
        })
    return {"query": q, "slot": slot, "refined": cleaned, "count": len(results),
            "ms": {"refine": round(t_refine * 1000), "embed": round(t_embed * 1000),
                   "total": round((time.time() - t0) * 1000)},
            "results": results}


SLOT_KO = {
    "hair": "헤어", "face": "성형", "cap": "모자", "faceAcc": "얼굴장식", "eyeAcc": "눈장식",
    "earring": "귀고리", "coat": "상의", "longcoat": "한벌옷", "pants": "하의", "shoes": "신발",
    "glove": "장갑", "cape": "망토", "weapon": "무기", "shield": "방패",
}


class RateItem(BaseModel):
    slot: str
    name: str | None = None
    id: str | None = None  # 있으면 골드 캡션(생김새)을 붙여 보낸다 → 핑크빈이 이름이 아니라 "모습"을 보고 말한다


class RateReq(BaseModel):
    items: list[RateItem] = []
    tone: int | None = None
    history: list[str] = []  # 직전에 한 말(반복 방지)


def _extract_bubbles(text: str) -> list[str]:
    """모델 응답에서 bubbles 리스트 추출(코드펜스/여분 텍스트 방어)."""
    try:
        return [str(b).strip() for b in json.loads(text).get("bubbles", []) if str(b).strip()]
    except Exception:
        pass
    import re
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return [str(b).strip() for b in json.loads(m.group(0)).get("bubbles", []) if str(b).strip()]
        except Exception:
            pass
    return []


@app.post("/rate")
async def rate(req: RateReq):
    # 아이템 이름만 보내면 "윈디 헤어(여)" 처럼 생김새 정보가 없어 모델이 헤어를 무시하고
    # 이름이 튀는 옷/장식에만 반응했다(헤어를 바꿔도 같은 말이 나오던 원인).
    # → 골드 캡션(words)을 함께 붙여 "모습"을 보고 말하게 한다.
    worn = []
    for it in req.items:
        if not it.name:
            continue
        line = f"{SLOT_KO.get(it.slot, it.slot)}: {it.name}"
        meta = BY_ID.get(it.id or "")
        if meta and meta.get("words"):
            line += f" — {', '.join(meta['words'])}"
        worn.append(line)
    # 말풍선 개수: 2개 또는 3개를 50:50 으로. 모델에 맡기면 늘 2개만 나왔다.
    n = random.choice((2, 3))
    parts = ["착용 코디:\n" + ("\n".join(worn) if worn else "(아무것도 안 입은 맨몸이야!)")]
    hist = [h for h in (req.history or []) if h][:6]
    if hist:
        parts.append("방금 한 말(겹치지 마):\n" + "\n".join(f"- {h}" for h in hist))
    parts.append(f"말풍선 {n}개로 말해줘.")
    user = "\n\n".join(parts)
    try:
        async with httpx.AsyncClient(timeout=18.0) as client:
            r = await client.post(
                f"{DASHSCOPE_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
                json={
                    "model": RATE_MODEL,
                    "messages": [
                        {"role": "system", "content": PINKBEAN_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.9,
                    "max_tokens": 300,
                },
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return {"bubbles": ["뀨…? 지금은 좀 부끄러운걸!", "이따 다시 보여줘…"], "error": str(e)[:120]}
    bubbles = _extract_bubbles(content)
    if len(bubbles) > n:      # 모델이 개수를 어기면 코드로 맞춘다
        bubbles = bubbles[:n][:3]
    if not bubbles:
        bubbles = ["뀨? 뭔가 신기한 코디인데?", "부농부농! 마음에 들어!"]
    return {"bubbles": bubbles, "model": RATE_MODEL}

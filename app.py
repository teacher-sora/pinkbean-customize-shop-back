"""
pinkbean-customize-shop-back — AI 코디 검색 · 코디 평가 API

- 기동할 때 data/ 의 벡터(정규화, float16)와 메타를 메모리에 올린다.
- POST /search : 질의를 정제 → 캡션 낱말로 치환 → 임베딩 → 코사인 + 낱말 일치 → 거리 컷 → 재정렬.
- POST /rate   : 착용 코디를 보고 핑크빈 말풍선을 짓는다(나온 문장은 안전망으로 거른다).
- POST /guide  : 건의함의 기능 안내. front 가 보낸 기능 목록에서 질문에 맞는 항목을 고른다.
- 코사인 = 내적(양쪽 정규화). 아이템 수 × 256 브루트포스라 검색 자체는 수 ms 다.
규칙과 그 이유는 ARCHITECTURE.md 에 있다.
"""
import os
import re
import json
import time
import unicodedata
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
    "반드시 JSON 으로만 답해: {\"bubbles\": [{\"s\": \"부위\", \"t\": \"말\"}, {\"s\": \"부위\", \"t\": \"말\"}]}\n"
    "- s 는 그 말을 할 때 **보고 있는 부위**다. 착용 코디에 적힌 부위 이름(헤어, 모자, 신발 …) 가운데 하나를 그대로 적어라. "
    "코디 전체를 두고 하는 말이면 빈 문자열.\n"
    "- bubbles 개수는 사용자 메시지에 적힌 수를 **정확히** 지켜라.\n"
    "- 각 bubble 은 뜻이 자연스럽게 통하는 '완성된 한 문장'. 앞뒤 안 맞는 말이나 어색한 비유(예: 맛있겠다 했다가 맛없겠다) 금지.\n"
    "- 먹는 것에 억지로 갖다붙이지 말 것. 정말 어울릴 때만 가볍게.\n"
    "- 너무 길지 않게(공백 포함 35자 이내), 서로 다른 내용으로.\n"
    "- ⚠️ '뀨', '부농부농' 같은 의성어·말버릇은 남발 금지. 보통은 안 쓰고, 아주 가끔 최대 1개만.\n"
    "- ⚠️ **반말로만** 말해라(핑크빈은 어린아이다). '~요', '~니다' 같은 존댓말 금지.\n"
    "- ★ 아이템마다 '생김새'가 함께 적혀 있으면 **그 생김새를 보고** 반응해라(이름만 보고 넘겨짚지 말 것).\n"
    "- ★ 매번 **다른 곳에 눈이 가야 한다.** 머리·옷·신발·무기·장식 등 여러 부위 중 이번엔 무엇이 눈에 띄었는지 골라서 말해라.\n"
    "- ★ '방금 한 말'이 주어지면 **그것과 겹치는 소재·표현은 피하고** 새로운 얘기를 해라.\n"
    "- ⛔ 성적인 말, 성폭력(강간·추행 등)을 뜻하는 말, 신체 부위를 성적으로 가리키는 말, 욕설, 폭력·자해를 "
    "암시하는 말은 **절대 쓰지 마라.** 핑크빈은 어린아이이고, 코디 구경은 누구나 보는 자리야. "
    "옷이나 장식을 의인화할 때도 그런 비유를 쓰지 마라(예: 옷이 '당했다'는 식의 표현 금지)."
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
# 짧은 캡션 벡터(있을 때만). 질의는 6낱말 이하라 상세 캡션 하나만으로는 희석된다 →
# 아이템의 점수는 상세 벡터와 짧은 벡터 중 높은 쪽을 쓴다. 짧은 벡터가 없는 행은 상세 벡터만 쓴다.
_SHORT_PATH = os.path.join(DATA, f"vectors_short_{DIM}.f16.npy")
SHORT_ROWS = np.asarray(META.get("short_rows") or [], dtype=np.int64)
MAT_SHORT = np.load(_SHORT_PATH).astype(np.float32) if (len(SHORT_ROWS) and os.path.exists(_SHORT_PATH)) else None
SHORT_POS = np.full(N, -1, dtype=np.int64)
if MAT_SHORT is not None:
    SHORT_POS[SHORT_ROWS] = np.arange(len(SHORT_ROWS))


def cosine(rows: np.ndarray, qv: np.ndarray) -> np.ndarray:
    base = MAT[rows] @ qv
    if MAT_SHORT is not None:
        pos = SHORT_POS[rows]
        m = pos >= 0
        if m.any():
            base[m] = np.maximum(base[m], MAT_SHORT[pos[m]] @ qv)
    return base


# 정본 캡션에서 온 행(tier v2)만 낱말 일치 가산을 받는다. 왜: 정본에 없는 행은 낱말이 사전으로 통일돼 있지 않아
# 가산이 우연에 좌우된다(embed_apply.py 는 정본에 없는 행을 건드리지 않는다 — ARCHITECTURE.md §6).
IS_V2 = np.asarray([it.get("tier") == "v2" for it in ITEMS], dtype=bool)
WORDS_JOINED = [" ".join(it.get("words") or []) for it in ITEMS]


def _ngrams(words):
    """캡션 낱말마다 어절 단위로 이어진 조각 전부. 낱말 일치는 이 집합으로 본다.
    왜 글자 부분일치가 아닌가: '큰 눈' 이 '큰 눈동자' 에, '순한' 이 '단순한' 에, '웃는 입' 이 '비웃는 입' 에 맞아 버린다."""
    out = set()
    for w in words or []:
        t = w.split()
        for i in range(len(t)):
            for j in range(i + 1, len(t) + 1):
                out.add(" ".join(t[i:j]))
    return out


WORD_NGRAMS = [(_ngrams(it.get("words")) if it.get("tier") == "v2" else None) for it in ITEMS]
LEX_BONUS = 0.30   # 질의 낱말이 캡션에 그대로 있는 비율 × 이 값


def lex_bonus(rows: np.ndarray, qwords: list) -> np.ndarray:
    """qwords 의 원소는 낱말이거나, 한 낱말을 푼 묶음(' | ' 로 이은 것)."""
    out = np.zeros(len(rows), dtype=np.float32)
    groups = [[a.strip() for a in w.split("|") if a.strip()] for w in qwords]
    groups = [g for g in groups if g]
    if not groups:
        return out
    for k, r in enumerate(rows.tolist()):
        if IS_V2[r]:
            t = WORD_NGRAMS[r]
            # 묶음(한 낱말을 푼 것)은 맞은 개수에 비례해 준다 — 하나만 맞으면 조금, 여러 특징이 함께 맞는 것이 위로 온다.
            # 왜 하나짜리를 낮게 주는가: "여우상"을 푼 낱말 중 '가늘고 긴 눈' 하나만 맞은 졸린 눈이 상위에 섞였다.
            tot = 0.0
            for g in groups:
                if g[0].startswith("="):   # '=' 로 시작하는 묶음은 하나만 맞아도 다 맞은 것으로 친다(같은 갈래의 다른 이름)
                    if any(" ".join(a.lstrip("=").split()) in t for a in g):
                        tot += 1.0
                    continue
                hit = sum(1 for a in g if " ".join(a.split()) in t)
                if hit:
                    tot += 0.25 + 0.75 * hit / len(g)
            out[k] = LEX_BONUS * tot / len(groups)
    return out


# 낱말 스키마(이표기 → 대표어). 캡션을 정규화한 것과 **같은 사전**으로 질의 낱말도 바꾼다.
# 프롬프트에 낱말 목록을 넣지 않고 코드로 바꾸는 이유: 목록을 주면 모델이 사용자가 말하지 않은 낱말을 베껴 넣는다.
try:
    with open(os.path.join(DATA, "vocab.json"), "r", encoding="utf-8") as _f:
        VOCAB = json.load(_f)
except Exception:
    VOCAB = {}


_VOCAB_CACHE = {}
_EMPTY = frozenset()
# 낱말(어절) → 그 낱말이 캡션에 쓰인 부위들.
TOKEN_SLOTS = {}
for _i, _it in enumerate(ITEMS):
    if WORD_NGRAMS[_i]:
        for _x in WORD_NGRAMS[_i]:
            if " " not in _x:
                TOKEN_SLOTS.setdefault(_x, set()).add(_it.get("slot"))


def vocab_for(slot) -> dict:
    """그 부위의 사전. 부위를 모르는 전체 검색에서는 부위 사전을 모두 합쳐 쓴다(옷·모자 먼저).
    왜: 전체 검색이 기본인데 부위 사전을 못 쓰면 "오피스룩"·"스쿨룩" 같은 말이 풀리지 않는다."""
    if slot not in _VOCAB_CACHE:
        m = {}
        if slot:
            m.update(VOCAB.get("*") or {})
            m.update(VOCAB.get(slot) or {})
        else:
            # 앞에 온 부위가 이긴다(옷 → 신발 → 모자 → 망토 → 무기 → 장식 → 헤어 → 성형). 공통 사전은 맨 뒤 — 부위 사전이 더 새롭고 구체적이다.
            for s_ in ("longcoat", "coat", "pants", "shoes", "cap", "cape", "weapon", "glove", "eyeAcc", "faceAcc", "hair", "face", "*"):
                for k_, v_ in (VOCAB.get(s_) or {}).items():
                    if s_ != "cap" and "귀" in k_ and "귀여" not in k_:
                        continue   # 귀는 모자 부위의 갈래 규칙이 맡는다
                    # 다른 부위의 캡션에서도 쓰이는 낱말은 한 부위의 풀이로 바꾸지 않는다.
                    # 왜: 망토 사전의 `동물` → `작은 동물 · 곁에 있는 것` 이 전체 검색의 "동물 귀"를 가로챘다.
                    if s_ != "*" and (TOKEN_SLOTS.get(k_, _EMPTY) - {s_}):
                        continue
                    m.setdefault(k_, v_)
        _VOCAB_CACHE[slot] = m
    return _VOCAB_CACHE[slot]


_Q_TAIL = ("오는", "내려오는", "닿는", "되는", "있는", "달린", "보이는", "같은", "느낌", "느낌의")   # 뜻 없이 붙는 꼬리말


# 성형에서는 "순한 인상", "도도한 얼굴" 처럼 부위·인상을 가리키는 말이 뒤에 붙는다 — 떼고 앞 낱말로 찾는다.
_Q_TAIL_FACE = ("인상", "인상의", "얼굴", "표정", "분위기", "눈매", "눈빛", "상")


# 부위별로, 캡션에 실제로 있는 두 어절 이상의 구. 질의의 구가 여기 있으면 쪼개지 않고 그대로 쓴다.
SLOT_PHRASES = {}
for _i, _it in enumerate(ITEMS):
    if WORD_NGRAMS[_i]:
        SLOT_PHRASES.setdefault(_it.get("slot"), set()).update(x for x in WORD_NGRAMS[_i] if " " in x)


# 캡션에 낱말이나 어절로 실제 쓰인 것(두 글자 이상).
KNOWN_TOKENS = set()
for _i, _it in enumerate(ITEMS):
    if WORD_NGRAMS[_i]:
        KNOWN_TOKENS.update(x for x in WORD_NGRAMS[_i] if " " not in x and len(x) >= 2)


# 부위별로, 캡션에 한 낱말로 통째 쓰인 것.
SLOT_FULL = {}
for _it in ITEMS:
    if _it.get("tier") == "v2":
        SLOT_FULL.setdefault(_it.get("slot"), set()).update(_it.get("words") or [])


def _all_words(span, slot, m) -> bool:
    """구의 어절이 저마다 독립된 캡션 낱말(또는 사전의 키)이면 구로 묶지 않는다.
    왜: "양갈래 당고머리" 를 한 구로 잡으면, 두 낱말을 따로 가진 아이템이 빠진다."""
    full = SLOT_FULL.get(slot) or ()
    if any(t in full for t in span):   # 어절 하나가 그 자체로 캡션 낱말이면("긴 양갈래"의 양갈래) 구로 좁히지 않는다
        return True
    return all((t in full or t in m) for t in span)


def _keep_parts(c, span, slot):
    """여러 어절이 사전에서 한 낱말로 바뀔 때, 어절이 저마다 캡션 낱말이면 그것들도 같은 묶음에 남긴다.
    왜: "양갈래 당고머리" → `양쪽 당고머리` 로만 바꾸면 양갈래와 당고머리를 따로 가진 아이템이 빠진다."""
    if c is None or len(span) < 2:
        return c
    full = SLOT_FULL.get(slot) or ()
    if all(t in full for t in span):
        have = [a.strip() for a in c.split("|")]
        return " | ".join(have + [t for t in span if t not in have])
    return c


# 한 글자짜리 캡션 낱말(검·창·활·총·낫·봉·책 …). 정제가 떨어뜨리면 되살린다.
ONE_CHAR_WORDS = {w for _it in ITEMS if _it.get("tier") == "v2" for w in (_it.get("words") or []) if len(w) == 1}


def _with_bare(key, span, slot_words):
    """캡션 구로 맞은 것에 부위어가 들어 있으면("날카로운 눈"), 부위어를 뗀 낱말("날카로운")도 같은 묶음에 넣는다.
    왜: 그 구가 드물게만 쓰였을 때 구 하나로 좁히면 같은 뜻의 흔한 낱말을 가진 아이템이 다 빠진다."""
    bare = " ".join(t for t in span if t not in slot_words)
    return f"{key} | {bare}" if bare and bare != key else key


def canon_query(words: list, slot, slot_words=()) -> list:
    """질의 낱말을 캡션 낱말로 바꾼다. 긴 구부터 사전·캡션 구와 맞춰 본다.
    왜: 부위어를 먼저 떼면 "웃는 눈" 이 "웃는" 이 되어 웃는 입에 맞고, "큰 눈" 이 "큰" 이 되어 큰 입·큰 눈동자에 맞는다."""
    m = vocab_for(slot)
    phrases = SLOT_PHRASES.get(slot) or ()
    tail = _Q_TAIL + (_Q_TAIL_FACE if slot == "face" else ())
    toks = [t for w in words for t in w.split()]
    out = []

    def add(c):
        if c and c not in out:
            out.append(c)

    i = 0
    while i < len(toks):
        hit = False
        for n in range(min(4, len(toks) - i), 0, -1):
            span = toks[i:i + n]
            key = " ".join(span)
            c = _keep_parts(m.get(key) or m.get("".join(span)), span, slot)
            if c is None and n >= 2 and key in phrases and not any(t in _Q_TAIL for t in span) and not canon_colors(key) and not _all_words(span, slot, m):   # 꼬리말이 낀 구는 좁게 잡지 않는다. 색이 낀 구도 — 색은 따로 2차로 본다
                c = _with_bare(key, span, slot_words)
            if c is not None:
                add(c)
                i += n
                hit = True
                break
        if hit:
            continue
        t = toks[i]
        i += 1
        if t in slot_words or t in tail:
            continue
        add(t)
    return out


_CLOTH_Q = ("원피스", "드레스", "한벌옷", "치마", "스커트", "바지", "교복", "제복", "옷", "세트", "코디")
# 신발 부위 가운데 **양말·스타킹인 것**(관찰 필드 kind). kind 가 없는 행은 신발이면 다 넣는다.
_SOCK_KINDS = ("양말·스타킹", "양말 신은 신발")
IS_SHOES = np.asarray([it.get("slot") == "shoes" and (it.get("kind") is None or it.get("kind") in _SOCK_KINDS)
                       for it in ITEMS], dtype=bool)
# 가산이 낱말 일치 가산(0.30)보다 커야 한다: 신발 부위의 양말은 대부분 발목 높이라 `양말` 로 적히고,
# `스타킹`·`니삭스` 낱말을 가진 것은 몇 개뿐이다. 사용자는 셋을 한 갈래로 찾는다.
HOSE_SHOES_BONUS, HOSE_OTHER_PEN = 0.35, 0.05
HOSE_PANTS_PEN = 0.30   # 스타킹 검색에서 '이어진 다리 덮개'(바지로도 읽히는 다리)를 뒤로 보내는 폭


def hose_adjust(rows):
    return np.where(IS_SHOES[rows], HOSE_SHOES_BONUS, -HOSE_OTHER_PEN).astype(np.float32)


def phrase_hits(q: str, slot, slot_words=()):
    """원문에서 두 어절 이상의 구가 사전이나 캡션 구와 맞는 것을 찾는다 → (캡션 낱말들, 쓰인 원문 어절 집합)."""
    m = vocab_for(slot)
    phrases = SLOT_PHRASES.get(slot) or ()
    toks = (q or "").split()
    out, used = [], set()
    i = 0
    while i < len(toks):
        for n in range(min(4, len(toks) - i), 1, -1):
            span = toks[i:i + n]
            key = " ".join(span)
            c = _keep_parts(m.get(key) or m.get("".join(span)), span, slot)
            if c is None and key in phrases and not any(t in _Q_TAIL for t in span) and not canon_colors(key) and not _all_words(span, slot, m):
                c = _with_bare(key, span, slot_words)
            if c is not None:
                if c not in out:
                    out.append(c)
                used.update(span)
                i += n - 1
                break
        i += 1
    return out, used


# 성별 = 아이템 "이름"의 접미 (여)/(남) 가 유일한 근거(build/embed_apply.py 가 meta 에 적는다). 0남/1여/2공용.
GENDERS = np.asarray([it.get("gender") if it.get("gender") is not None else 2 for it in ITEMS], dtype=np.int64)
# 최신 가산(헤어·성형·무기): 새로 나온 아이템을 조금 위로. 캡션 유사도가 우선이라 상한을 낮게 둔다
# (동점~근접 구간에서만 순서가 바뀐다).
# 기본값은 부위 안 id 의 **순위(백분위)** — id 간격이 들쭉날쭉하거나 이상치가 있어도 왜곡되지 않는다.
# build/recency.py 가 넣은 recency 가 있는 행은 아래에서 그 값으로 덮는다(recency 가 없는 행만 이 값이 남는다).
ID_BONUS_MAX = 0.03
ID_BONUS = np.zeros(len(ITEMS), dtype=np.float32)
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

# 최신도는 id 만으로 정하지 않는다: 낮은 id 로 나온 신상품이 있다. 빌드가 넣은 recency(0~1 —
# id 백분위·패치 반영 시각·출시 순번 중 가장 높은 값)가 있으면 그것을 쓴다.
for _i, _it in enumerate(ITEMS):
    if _it.get("recency") is not None:
        ID_BONUS[_i] = np.float32(ID_BONUS_MAX * float(_it["recency"]))

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
# 캡션에 적힌 색 전부(tier v2 행). "검은 마스크"·"빨간 테 안경"처럼 색을 말한 질의는 대표색 하나가 아니라
# 캡션이 적은 색(부분의 색 포함)으로 맞춘다.
ITEM_CAPCOLORS = [(set().union(*[canon_colors(w) for w in (it.get("words") or [])]) if it.get("tier") == "v2" else None)
                  for it in ITEMS]
COLOR_BONUS, COLOR_PEN = 0.15, 0.12
# 거리 임계: 개념 질의에서 top1 코사인의 이 비율 미만인 후보는 "무관"으로 컷(개수 채우기 방지).
# 쿼리마다 코사인 절대값이 달라 상대비율이 안전. 0.72 = 최상위와 비슷하게 가까운 것만 남김.
DIST_RATIO = 0.72
NO_LEX_CAP = 12   # 낱말이 하나도 맞지 않은 질의에서 돌려주는 최대 개수
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


_WKIND = {"검·칼": "검", "창·긴 자루": "창", "활·석궁": "활", "총": "총", "둔기": "둔기", "지팡이·봉": "지팡이"}


def _weapon_type(it):
    if it.get("slot") != "weapon":
        return None
    if it.get("kind") is not None:   # 관찰한 갈래가 낱말 끝 글자 추측보다 정확하다('물총'은 총, '꽃다발 지팡이'는 소품)
        fam = _WKIND.get(it["kind"])
        if fam:
            return fam
        if it["kind"] in ("도끼·낫", "그 밖의 무기"):
            w0 = (it.get("words") or [""])[0]
            for k, f in _WTYPE.items():
                if w0 == k or w0.endswith(k):
                    return f
        return "기타"   # 무기가 아닌 것 — 무기 종류 질의에서는 감점된다
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

# ── 동물 귀 갈래 점수 ───────────────────────────────────────────────
# "동물 귀" 는 임베딩에서 '동물' 이 지배해 동물 옷·무기가 올라오고, 정작 귀(모자 부위)가 묻힌다.
# 귀 질의에서는 모자 부위의 귀 갈래(아래 _ear_class)에 가산하고 그 밖은 감점한다.
_EAR_ITEM_EXCLUDE = ("귀걸이", "귀고리", "귀여", "잎사귀")


def _is_ear_item(it):
    if it.get("slot") != "cap":  # 귀 형태 = 모자 슬롯 머리띠. 동물 옷(한벌옷)은 '귀'가 있어도 제외.
        return False
    for w in (it.get("words") or []):
        if "귀" in w and not any(x in w for x in _EAR_ITEM_EXCLUDE):
            return True
    return False


EAR_BONUS, EAR_PEN = 0.25, 0.12

# 귀는 두 갈래다 — 관찰 필드(kind·has_ears)로 가른다.
#   1 = 귀만 달린 장식: 머리카락과 어우러져 캐릭터 자체가 동물처럼 보인다("동물 귀").
#   2 = 귀 달린 모자·후드·탈·머리띠, 동물 모양 모자("동물 귀 모자").
# 왜: 낱말 '귀'만 보면 곰 후드가 "동물 귀" 맨 위에 온다. 사람이 찾는 것은 갈래가 다르다.
# kind 가 없는 행은 낱말에 '귀'가 있으면 1 로 둔다.
_EAR_HAT_KINDS = ("모자", "후드·탈", "머리띠")


def _ear_class(it):
    if it.get("slot") != "cap":
        return 0
    kind = it.get("kind")
    if kind is None:
        return 1 if _is_ear_item(it) else 0
    if kind == "귀·뿔 장식":
        return 1 if it.get("has_ears") else 0
    if kind in _EAR_HAT_KINDS and (it.get("has_ears") or (kind != "머리띠" and it.get("animal"))):
        return 2
    return 0


ITEM_EARC = np.asarray([_ear_class(it) for it in ITEMS], dtype=np.int8)
EAR_OTHER = 0.05   # 찾는 갈래가 아닌 쪽 귀 — 빼지는 않고 뒤로만 보낸다
_EAR_HAT_Q = ("모자", "후드", "후디", "탈", "머리띠", "비니", "캡", "두건", "인형탈")


def ear_adjust(rows, text: str):
    """귀 질의의 갈래에 맞춰 가산·감점. 질의에 모자 낱말이 있으면 귀 달린 모자 쪽, 없으면 귀 장식 쪽."""
    want = 2 if any(w in (text or "") for w in _EAR_HAT_Q) else 1
    c = ITEM_EARC[rows]
    return np.where(c == want, EAR_BONUS, np.where(c == 0, -EAR_PEN, -EAR_OTHER)).astype(np.float32)
# 귀 질의 감지: 홀로 선 '귀' 토큰(예: "동물 귀", "고양이 귀", "귀 머리띠"). 귀걸이/귀여운/귀신/까마귀 등 오탐 제외.
_EAR_Q_EXCLUDE = ("귀걸이", "귀고리", "귀여", "귀신", "귀족", "귀환", "귀가", "귀중", "잎사귀", "까마귀", "당나귀")


def ear_query(text: str) -> bool:
    for t in (text or "").split():
        if t == "귀":
            return True
        if t.endswith("귀") and len(t) <= 4 and not any(x in t for x in _EAR_Q_EXCLUDE):
            return True
    return False


# 검색 범위: 캐시 12부위 + 명찰 · 말풍선. 귀고리(earring)·방패(shield)·피부(skin)는 착용해도 거의 안 보여 넣지 않는다.
# 명찰 · 말풍선은 캐릭터와 함께 늘 보여 코디를 바꾸므로 넣는다(사용자 지시). 일루전 링 · 캐시 이펙트는 넣지 않는다.
# 정본 캡션도 이 부위뿐이지만, 잘못된 데이터가 들어와도 범위 밖이 새어 나오지 않게 한 번 더 건다.
CAPTION_SLOTS = {"hair", "face", "cap", "faceAcc", "eyeAcc",
                 "coat", "longcoat", "pants", "shoes", "glove", "cape", "weapon", "nameTag", "chatBalloon"}
SCOPE_MASK = np.asarray([it.get("slot") in CAPTION_SLOTS for it in ITEMS], dtype=bool)
SCOPE_ROWS = np.nonzero(SCOPE_MASK)[0]
# id → 아이템(캡션 words 조회용). 코디 평가가 "이름"이 아니라 "생김새"를 보고 말하게 한다.
BY_ID = {it["id"]: it for it in ITEMS}

# ── 이름 검색용 정규화 인덱스 ────────────────────────────────────────
# 아이템 이름으로 찾는 길. 캡션에는 이름을 적지 않으므로(캡션 규칙) 이름은 여기서 따로 맞춘다.
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
    "- 사용자가 쓴 **인상 낱말**(생김새에서 받는 느낌을 나타내는 말)은 캡션에도 색인돼 있으므로 words 에 그대로 남긴다.\n"
    "  '예쁜, 멋진, 좋은, 추천' 처럼 생김새를 가리지 못하는 평가어만 버린다.\n"
    "- **길이·높이·위치·개수를 나타내는 말은 생김새다. 버리지 않는다.** '발끝까지 오는', '허리까지', '한쪽', '양쪽', '두 개'\n"
    "  같은 표현은 군더더기가 아니므로 words 에 남긴다(조사·어미만 떼고).\n"
    "- 부위를 가리키는 말은 words 에 넣지 말고 slot 으로 뺀다.\n"
    "  slot 값: hair(헤어/머리) face(성형/눈/눈매/눈동자) cap(모자/탈/투구) faceAcc(얼굴장식) eyeAcc(눈장식/안경)\n"
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
# 임베딩에서 뺄 "순수 부위어"(캡션에 거의 안 쓰이는 카테고리 명칭)만. 원피스/드레스/바지/치마 같은
# **형태를 지칭하는 말은 캡션 토큰이라 빼지 않는다**(빼면 '검정 원피스'가 '검정'만 남아 개념이 사라짐).
SLOT_WORDS = {
    "hair": ("헤어", "머리"), "face": ("성형", "눈매", "눈", "얼굴"), "cap": (), "faceAcc": ("얼굴장식",),
    "eyeAcc": ("눈장식",), "coat": ("상의",), "longcoat": (),
    # 모자·안경·장갑·망토는 부위 이름이면서 그 부위 안의 **갈래 이름**이기도 하다(모자 부위에는 귀·핀·후광이,
    # 망토 부위에는 날개·꼬리·가방이 함께 있다) → 떼지 않는다. 떼면 "왕의 망토"에 날개가 나온다.
    "pants": ("하의",), "shoes": ("신발",), "glove": (), "cape": (),
    "weapon": ("무기",),
    # 명찰 · 말풍선 캡션에는 모두 그 부위 이름이 들어 있어(코드가 붙인다) 그 부위 안에서는 가르지 못한다 → 뗀다.
    "nameTag": ("명찰", "네임택"), "chatBalloon": ("말풍선",),
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
            "slots": {k: int(len(v)) for k, v in SLOT_ROWS.items()},
            "looks": len(LOOKS["lines"]), "looksReady": LOOKS["ready"]}  # 코디 조합 표본(아래 POST /looks)


# 사용자가 직접 말한 부위 이름 → 부위. (형태를 가리키는 말 '원피스·드레스·치마·바지'는 넣지 않는다 — 캡션 낱말이다.)
_EXPLICIT_SLOT = (("한벌옷", "longcoat"), ("상의", "coat"), ("하의", "pants"), ("신발", "shoes"), ("얼굴장식", "faceAcc"),
                  ("눈장식", "eyeAcc"), ("망토", "cape"), ("무기", "weapon"), ("성형", "face"), ("헤어", "hair"),
                  # 명찰 · 말풍선은 질의 정제의 부위 목록에 넣지 않고 여기서만 가른다(정제 지시문을 바꾸면 다른 질의의 정제가 흔들린다).
                  # '이름표'는 넣지 않는다 — 옷 캡션의 이름표(교복 · 제복에 단 것) 14곳과 부딪힌다.
                  ("명찰", "nameTag"), ("네임택", "nameTag"), ("말풍선", "chatBalloon"))
_CLOTH_ROWS = np.concatenate([SLOT_ROWS.get(s_, np.zeros(0, dtype=np.int64)) for s_ in ("longcoat", "pants")])


# ── 부정어(제외) 처리 ────────────────────────────────────────────────
# "묶음 없고", "리본 없는" 처럼 특정 특징을 **빼달라는** 질의. 임베딩은 부정을 표현 못 하고(오히려 그 단어가
# 들어간 걸 끌어온다), 캡션이 "장발 트윈테일 끝 컬"이면 '장발 컬' 질의에 트윈테일이 딸려온다.
# → 부정어를 코드로 파싱해 **그 특징을 가진 아이템을 후보에서 아예 제외**한다.
_NEG_MARKERS = ("없는", "없고", "없이", "없음", "않은", "않는", "아닌", "빼고", "제외", "말고")
# 묶음류는 캡션 토큰이 여러 형태(트윈테일/포니테일/번/당고…)라 계열로 확장한다. 나머지는 낱말 그대로 제외.
_NEG_TIE = ("묶음", "묶은", "포니테일", "포니", "트윈테일", "트윈", "양갈래", "당고", "올림머리", "올림", "땋", "반묶", "번", "꽁지")
_NEG_FAMILY = {"묶음": _NEG_TIE, "묶은": _NEG_TIE, "묶": _NEG_TIE, "트윈": _NEG_TIE, "포니": _NEG_TIE}
# 낱말 자체가 "~이 없다"는 뜻인 것. 캡션은 없는 것을 적지 않으므로, 그 특징을 가진 아이템을 빼는 것으로 찾는다.
# (무쌍·죽은 눈은 여기 두지 않는다 — 성형 캡션이 `홑꺼풀`·`빛 없는 눈동자` 를 긍정 낱말로 갖고 있어 사전이 푼다.)
_NEG_IMPLICIT = {"생얼": "화장", "민낯": "화장", "노메이크업": "화장"}


def parse_negatives(q: str):
    """원문에서 '<특징> 없고/없는/아닌…' 의 <특징> 들을 뽑는다."""
    out = []
    toks = (q or "").split()
    for i, t in enumerate(toks):
        if t in _NEG_IMPLICIT:
            out.append(_NEG_IMPLICIT[t])
        for m in _NEG_MARKERS:
            if t == m:
                if i > 0:
                    out.append(toks[i - 1])
            elif t.endswith(m) and len(t) > len(m):
                out.append(t[: -len(m)])
    return out


def neg_exclude_tokens(negs):
    ex = set()
    for term in negs:
        fam = None
        for k, f in _NEG_FAMILY.items():
            if k in term:
                fam = f
                break
        ex.update(fam if fam else (term,))
    return ex


def _item_negated(it, ex) -> bool:
    subs = set()
    for w in (it.get("words") or []):
        for s in w.split():
            subs.add(s)
    return any(any(tok in s for s in subs) for tok in ex)


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
    # 부정어(제외): 원문에서 '<특징> 없고' 파싱 → 긍정 임베딩에서 그 특징/마커를 빼고, 제외토큰 집합을 만든다.
    neg_terms = parse_negatives(q)
    neg_ex = neg_exclude_tokens(neg_terms) if neg_terms else set()
    if neg_terms:
        _drop = set(neg_terms) | set(_NEG_MARKERS) | set(_NEG_IMPLICIT)
        words = [w for w in words if w not in _drop and not any(w.endswith(m) for m in _NEG_MARKERS)]
    # 정체성 = 개념(명사)이지 색이 아니다. 임베딩은 **색을 뺀 개념**만으로 → 색이 개념을 덮지 않게.
    # 색은 아래에서 개념 매치 안의 2차 재정렬로만 반영. 어순 불변을 위해 정렬해 임베딩.
    # 부위를 직접 고른 검색에서도 그 부위를 가리키는 말("눈", "머리")은 임베딩·낱말 일치에서 뺀다.
    # 사용자가 부위 이름을 직접 말했으면("한벌옷 스타킹") 그 부위어도 낱말에서 뗀다 — 남겨 두면 임베딩이 부위어 쪽으로 쏠린다.
    _named0 = next(((w_, s_) for w_, s_ in _EXPLICIT_SLOT if w_ in q), None)
    _sw = SLOT_WORDS.get(req.slot or (_named0[1] if _named0 else None) or ref_slot or "", ()) + ((_named0[0],) if _named0 else ())
    # 낱말 스키마: 이표기 → 대표어. 부위어는 구("웃는 눈", "큰 눈")로 먼저 맞춰 본 뒤 남은 것만 뗀다.
    # 정제가 부위어를 떼어 버리기 전의 원문에서 구를 먼저 찾는다("웃는 눈" → 눈이 웃는 낱말, "일자 눈썹").
    # 사전을 찾을 부위. 옷(상의·한벌옷·하의)으로 추정된 질의는 여러 부위에 걸쳐 찾으므로 합친 사전을 쓴다.
    _lk = req.slot or (_named0[1] if _named0 else None) or (ref_slot if ref_slot not in ("coat", "longcoat", "pants") else None)
    # 정제가 떨어뜨린 낱말을 되살린다: 원문의 어절이 사전이나 캡션에 있는 낱말인데 정제 결과에 없으면 다시 넣는다.
    # 왜: 정제 모델이 "교복"·"드레스" 같은 옷 종류를 부위어로 보고 지우는 일이 있다("빨간 드레스" → "빨간").
    _have = " ".join(words)
    _vk = vocab_for(_lk)
    for _t in detect_gender(q)[1].split():
        if _t in _have or _t in _sw or _t in _NEG_MARKERS or _t in _NEG_IMPLICIT or canon_colors(_t):
            continue
        if any(_t.endswith(m_) for m_ in _NEG_MARKERS) or _t in neg_terms:
            continue
        if _t in _vk or _t in KNOWN_TOKENS or _t in ONE_CHAR_WORDS:
            words.append(_t)
    _pre, _used = phrase_hits(q, _lk, _sw)
    if _pre:
        words = [w for w in words if not all(t in _used for t in w.split())]
    words = (_pre + [c for c in canon_query(words, _lk, _sw) if c not in _pre]) or words
    # 사전이 통째로 푼 구 안의 "없는"은 부정이 아니다("손가락 없는 장갑" → 반장갑, "굽 없는 신발" → 납작한 신발).
    if neg_terms and _used:
        neg_terms = [t for t in neg_terms if t not in _used]
        neg_ex = neg_exclude_tokens(neg_terms) if neg_terms else set()
    qcolors = canon_colors(" ".join(words)) | canon_colors(q)
    concept_words = [w for w in words if not canon_colors(w)]
    embed_words = concept_words or words           # 순수 색 질의면 색 자체로 임베딩
    # 임베딩에는 묶음을 풀어 낱말로 넣는다(낱말 일치는 묶음 단위로 센다).
    cleaned = " ".join(sorted(a.strip() for w in embed_words for a in w.split("|") if a.strip()))
    if not cleaned:   # 부정어만 있는 질의("하이라이트 없는 눈")는 낱말이 다 빠진다 → 원문에서 부정 표지만 떼고 쓴다
        cleaned = " ".join(w for w in q.split() if w not in _NEG_MARKERS) or q
    color_strong = not concept_words               # 개념 없이 색만 → 색 강하게
    # 부위를 말하지 않은 스타킹·니삭스·양말은 한 갈래로 본다(낱말 일치에서만). 왜: 신발 부위의 양말은 대부분
    # 발목 높이라 `양말` 로만 적혀 있어, "스타킹" 으로 찾으면 낱말이 맞지 않아 거리 컷에서 잘린다.
    _HOSE = ("스타킹", "니삭스", "양말", "타이츠", "팬티스타킹")
    lex_words = embed_words
    if (not req.slot and not _named0 and not any(c in q for c in _CLOTH_Q)
            and any(h in q for h in _HOSE)):
        lex_words = [("=스타킹 | 니삭스 | 양말" if any(h in w for h in _HOSE) and "|" not in w else w) for w in embed_words]
    t_refine = time.time() - t0

    t1 = time.time()
    qv = await embed_query(cleaned)
    t_embed = time.time() - t1

    # 후보 부위: 사용자가 고른 것(req.slot) 우선, 없으면 질의에서 유추한 슬롯(ref_slot)으로 제한.
    # ★ 단, 교차슬롯 개념(스타킹/타이츠/양말/니삭스)은 여러 슬롯에 걸쳐 있으므로 ref_slot 으로 가두지
    #   않는다("전체 스타킹"이 바지로 갇혀 한벌옷/신발 스타킹이 사라지던 문제). 그 경우 전체+슬롯게이트.
    _cross = any(c in cleaned or c in q for c in ("스타킹", "타이츠", "양말", "니삭스", "팬티스타킹"))
    # 옷(상의/한벌옷/하의)은 슬롯으로 가두지 않는다 → 전체+슬롯게이트+한벌옷 가산으로 **한벌옷을 우선** 노출.
    _clothing = ref_slot in ("coat", "longcoat", "pants")
    # 사용자가 부위 이름을 **직접 말했으면** 그 부위로 가둔다("한벌옷 스타킹" → 한벌옷만).
    # 교차 슬롯·옷 완화는 부위를 말하지 않았을 때만 쓴다.
    _named = next((s_ for w_, s_ in _EXPLICIT_SLOT if w_ in q), None)
    slot = req.slot if (req.slot and req.slot in SLOT_ROWS) else (
        _named if _named else (ref_slot if (ref_slot and not _cross and not _clothing) else None))
    has_slot = bool(slot)
    rows = SLOT_ROWS[slot][SCOPE_MASK[SLOT_ROWS[slot]]] if has_slot else SCOPE_ROWS
    if allowed_g is not None:
        rows = rows[np.isin(GENDERS[rows], list(allowed_g))]
    # 스타킹·니삭스는 남성 한벌옷·하의에 없는 요소다(긴 바지를 스타킹으로 보지 않는다) → 성별을 말하지 않았으면 남성 옷은 뺀다.
    _hosiery_q = any(h in q or h in " ".join(words) for h in ("스타킹", "타이츠", "팬티스타킹", "니삭스"))
    if _hosiery_q and allowed_g is None and len(rows):
        rows = rows[~((GENDERS[rows] == 0) & np.isin(rows, _CLOTH_ROWS))]
    # 부정어 제외: 해당 특징(계열 포함)을 가진 아이템은 후보에서 제거(트윈테일 등 묶음류 아웃).
    if neg_ex and len(rows):
        rows = rows[np.fromiter((not _item_negated(ITEMS[r], neg_ex) for r in rows.tolist()), dtype=bool, count=len(rows))]
    if len(rows) == 0:
        return {"query": q, "slot": slot, "count": 0, "results": []}
    # 관련도 = 캡션 벡터 코사인 + 낱말 일치 가산. 이름 부분일치는 따로 본다(nb).
    # 이름은 정제문·원문 **둘 다**로 본다 — 정제가 이름을 쪼개거나 부위를 떼어내도 이름 검색이 죽지 않도록.
    base = cosine(rows, qv) + lex_bonus(rows, lex_words)  # 개념 코사인 + 낱말 일치(tier v2 행만)
    nb = np.maximum(name_bonus_for(rows, cleaned), name_bonus_for(rows, detect_gender(q)[1]))
    wq = weapon_query_type(cleaned)  # 무기 타입 질의(총/검/창...) — 색토큰이 빠진 cleaned 기준
    # 무기 타입 가산을 "관련도"에 미리 반영: 총류는 임베딩 코사인이 낮아도 거리컷에 안 잘리게.
    rel = base + (np.fromiter(
        ((WTYPE_BONUS if ITEM_WTYPE[r] == wq else (-WTYPE_PEN if ITEM_WTYPE[r] else 0.0))
         for r in rows.tolist()), dtype=np.float32, count=len(rows)) if wq else 0.0)
    # 동물 귀 폼-매칭도 관련도에 미리 반영: '동물' 지배에 묻힌 귀(모자)가 거리컷에 잘리지 않게 미리 끌어올린다.
    eq = ear_query(cleaned) or ear_query(q)
    if eq and len(rel):
        rel = rel + ear_adjust(rows, q)   # 갈래는 사용자가 쓴 말로만 정한다(사전이 푼 낱말에 모자가 섞여도 흔들리지 않게)
    # 부위를 말하지 않은 스타킹·니삭스·양말 질의는 **신발 부위의 것**(따로 신는 양말·스타킹)을 먼저 보여 준다.
    # 옷에 딸린 스타킹은 "한벌옷 스타킹"처럼 부위를 말했을 때 그 부위 안에서 찾는다(위의 _named).
    # 왜 가산이 큰가: 신발 부위의 양말은 대부분 `양말` 로만 적혀 있어 `스타킹`·`니삭스` 로는 낱말 일치를 받기 어렵다(HOSE_SHOES_BONUS).
    # 옷을 함께 말한 질의("스타킹 있는 원피스")는 옷을 찾는 것이므로 신발 우선을 걸지 않는다.
    _hose_plain = _cross and not has_slot and not any(c in q for c in _CLOTH_Q)
    if _hose_plain and len(rel):
        rel = rel + hose_adjust(rows)
    # ★ 거리 임계: 개수를 채우지 말고 "가까운 것만" 남긴다. 개념 질의는 top1 대비 상대임계로 무관한 꼬리를 컷.
    #   전체(슬롯 미지정)는 **슬롯별** top1 기준 → 한 슬롯이 전역을 독점해도 다른 슬롯 상위가 살아남음
    #   (예: 전체 '스타킹'에서 바지뿐 아니라 한벌옷/신발 스타킹도). 이름매칭 항상 포함. 순수 색 질의는 컷 안 함.
    if concept_words and len(rel):
        keep = (nb > 0)
        # ★ 낱말 일치의 단(段)으로 자른다: 가장 많이 맞은 아이템의 60% 에 못 미치게 맞은 것은 버린다.
        #   두 낱말 질의면 둘 다, 세 낱말이면 둘 이상 맞아야 남는다. 왜: "꽃 장식 검"에 꽃만 맞는 우산·꽃다발이,
        #   "기계 날개"에 빛 날개가 상위를 채웠다. 맞는 것이 적으면 적게 돌려준다(개수를 채우지 않는다).
        _lx = lex_bonus(rows, lex_words)
        _lmax = float(_lx.max()) if len(_lx) else 0.0
        _tier = (_lx >= 0.6 * _lmax) if _lmax > 0 else None
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
        if _tier is not None and int((keep & (_tier | (nb > 0))).sum()) >= 1:
            keep = keep & (_tier | (nb > 0))
        elif _tier is None and bool(IS_V2[rows].all()) and int(keep.sum()) > NO_LEX_CAP:
            # 낱말이 하나도 맞지 않았다(캡션에도 사전에도 없는 말) → 임베딩이 가깝다고 본 것 몇 개만 보여 준다.
            _ord = np.argsort(-np.where(keep, rel, -9.0))[:NO_LEX_CAP]
            keep = np.zeros(len(rows), dtype=bool)
            keep[_ord] = True
        # 색을 말한 질의("검은 마스크")는, 그 색이 캡션에 적힌 아이템이 남아 있으면 그것만 남긴다(tier v2 행에 한해).
        # 왜: 색은 말했을 때만 찾아 주기로 했다 — 말했는데 다른 색 마스크가 섞이면 안 된다.
        if qcolors and keep.any():
            _cm = np.fromiter(((ITEM_CAPCOLORS[r] is None) or bool(ITEM_CAPCOLORS[r] & qcolors) for r in rows.tolist()),
                              dtype=bool, count=len(rows))
            if int((keep & _cm).sum()) >= 1:
                keep = keep & _cm   # 이름이 맞아도 색이 다르면 뺀다("검은 마스크"에 이름만 마스크인 빨간 마스크)
        if keep.any():
            rows = rows[keep]
            base = cosine(rows, qv) + lex_bonus(rows, lex_words)
            nb = np.maximum(name_bonus_for(rows, cleaned), name_bonus_for(rows, detect_gender(q)[1]))
    scores = base + NAME_BONUS * nb
    # 스타킹·니삭스 질의의 순서 다듬기. (남성 한벌옷·하의는 위에서 이미 후보에서 뺐다.)
    if any(h in cleaned for h in ("스타킹", "타이츠", "팬티스타킹", "니삭스")):
        # 신발까지 달린 것은 스타킹 요소로는 덜 순수하다 → 확실히 뒤로(다른 스타킹 아이템이 먼저 나온다).
        scores = scores - 0.20 * np.fromiter(("신발 포함" in WORDS_JOINED[r] for r in rows.tolist()), dtype=bool, count=len(rows))
    if any(h in cleaned or h in q for h in ("스타킹", "타이츠", "니삭스", "양말")):
        # 옷 끝에서 틈·윗단 없이 바로 이어지는 불투명한 다리('이어진 다리 덮개')는 바지·레깅스로도 읽힌다
        # → 빼지는 않고 확실한 스타킹·양말(맨살 틈·윗단·비침) 뒤로 보낸다.
        scores = scores - HOSE_PANTS_PEN * np.fromiter(("이어진 다리 덮개" in WORDS_JOINED[r] for r in rows.tolist()), dtype=bool, count=len(rows))
    # 헤어·성형·무기: 최신 가산(상한 ID_BONUS_MAX).
    scores = scores + ID_BONUS[rows]
    # 전체(슬롯 미지정) 검색에서 옷 개념은 한벌옷을 소폭 우선(최근 신상·완성도 높은 아이템이 한벌옷 위주,
    # 대체로 더 예쁨). 무관 질의('총' 등)의 한벌옷은 코사인이 낮아 이 가산으로도 위로 오지 않는다.
    if not has_slot:
        _lc = SLOT_ROWS.get("longcoat")
        if _lc is not None:
            scores = scores + 0.10 * np.isin(rows, _lc)
    # 무기 타입-매칭: 정체성=타입. 계열 일치 가산/불일치 감점(총→총류, 검→검류...).
    if wq:
        scores = scores + np.fromiter(
            ((WTYPE_BONUS if ITEM_WTYPE[r] == wq else (-WTYPE_PEN if ITEM_WTYPE[r] else 0.0))
             for r in rows.tolist()), dtype=np.float32, count=len(rows))
    # 동물 귀 폼-매칭: 형태 정체성. 귀(모자) 가산 / 그 외 감점 → "동물 옷/무기"가 아니라 귀가 위로 온다.
    if eq:
        scores = scores + ear_adjust(rows, q)   # 갈래는 사용자가 쓴 말로만 정한다(사전이 푼 낱말에 모자가 섞여도 흔들리지 않게)
    if _hose_plain:
        scores = scores + hose_adjust(rows)
    # 색-매칭: 색은 2차. 가산은 같고(0.15), 감점만 순수 색 질의에서 조금 크다(0.12 / 개념이 있으면 0.10).
    if qcolors:
        qfam = set().union(*[_FAMILY.get(c, {c}) for c in qcolors])
        cb, cp = (COLOR_BONUS, COLOR_PEN) if color_strong else (0.15, 0.10)   # 색은 2차(염색으로 바뀐다) — 말했을 때만, 형태보다 약하게
        _ctoks = {t for t in cleaned.split() if len(t) >= 2}
        # 순수 색 질의("빨간 모자", "핑크 명찰")는 대표색(물건 전체의 색)만 본다 — 모든 부위.
        # 왜: 낱말이 색뿐이라 색 낱말 자체가 _ctoks 에 들어가, '빨간 리본' 처럼 부분만 그 색인 것도 다 가산을 받았다
        # (명찰에서 분홍 글자만 있는 보라 판이 맨 위에 왔다 — 사용자 지시로 모든 부위에 같게).
        if color_strong:
            _ctoks = set()
        def _cadj(r):
            cc = ITEM_CAPCOLORS[r]
            if cc is not None:
                # 말한 색이 **그 물건 자체의 색**이면 가산 — 대표색이 그 색이거나, 캡션의 한 낱말 안에
                # 색과 질의 낱말이 함께 있을 때("빨간 미니 드레스"). 부분의 색일 뿐이면("검은 드레스 · 빨간 리본") 0.
                if cc & qcolors:
                    if ITEM_PRIMARY[r] in qcolors:
                        return cb
                    for w in (ITEMS[r].get("words") or []):
                        if (canon_colors(w) & qcolors) and any(t in w.split() for t in _ctoks):
                            return cb
                    return 0.0
                return 0.0 if ITEM_PRIMARY[r] in qfam else -cp
            return cb if (ITEM_PRIMARY[r] in qfam) else (-cp if ITEM_PRIMARY[r] else 0.0)
        adj = np.fromiter(
            (_cadj(r)
             for r in rows.tolist()),
            dtype=np.float32, count=len(rows),
        )
        scores = scores + adj
    # 개념 질의는 상위 N개가 아니라 **거리 컷을 통과한 것을 전부** 돌려준다(상한 300). 순수 색 질의만 topK.
    k = min(max(req.topK, 300) if concept_words else req.topK, len(rows))
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
    "glove": "장갑", "cape": "망토", "weapon": "무기", "shield": "방패", "nameTag": "명찰", "chatBalloon": "말풍선",
}


class RateItem(BaseModel):
    slot: str
    name: str | None = None
    id: str | None = None  # 있으면 그 아이템의 캡션(생김새)을 붙여 보낸다 → 핑크빈이 이름이 아니라 "모습"을 보고 말한다


class RateReq(BaseModel):
    items: list[RateItem] = []
    tone: int | None = None   # 쓰지 않는다. 프런트가 보내므로 받기만 한다
    history: list[str] = []  # 직전에 한 말(반복 방지)
    recent_slots: list[str] = []  # 바로 앞 평가에서 본 부위(이번에는 다른 부위를 본다)


def _extract_bubbles(text: str) -> list[tuple[str, str]]:
    """모델 응답에서 (말, 보고 있는 부위) 목록을 꺼낸다(코드펜스/여분 텍스트 방어).
    말풍선은 {"s": 부위, "t": 말} 이지만, 모델이 문자열만 줄 때도 받는다(부위는 빈 문자열)."""
    def rows(obj) -> list[tuple[str, str]]:
        out = []
        for b in obj.get("bubbles", []):
            if isinstance(b, dict):
                t, sl = str(b.get("t") or b.get("text") or "").strip(), str(b.get("s") or b.get("slot") or "").strip()
            else:
                t, sl = str(b).strip(), ""
            if t:
                out.append((t, sl))
        return out
    try:
        return rows(json.loads(text))
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return rows(json.loads(m.group(0)))
        except Exception:
            pass
    return []


# ── 말풍선 안전망 ────────────────────────────────────────────────────────
# 말은 모델이 짓는다. 실제로 성폭력을 뜻하는 낱말이 섞인 말풍선이 나온 적이 있다.
# 지시문은 확률을 낮출 뿐 **못 막는다** → 나온 문장을 코드로 검사해 걸리면 버리고, 모자라면 한 번 더 받는다.
# 프런트(front/src/lib/safeText.ts)도 같은 규칙으로 화면 직전에 한 번 더 본다(양쪽 배포가 따로라서).
# 오탐은 말풍선 하나를 잃을 뿐이라 넉넉히 잡되, 패션 얘기에서 정상적으로 쓰이는 말은 일부러 뺐다:
#   노출(노출이 심한 옷) · 누드(누드톤) · 로리타(패션 장르) · 벗겨(칠이 벗겨진) · 죽여(죽여주는 코디)
#   · 사정(사정이 있어) · 보지/자지(보지 마, 만지지 — 어미와 겹친다) · 씹(씹다; 욕설 형태만 따로 잡는다)
_BAN_LOOSE = (  # 띄어쓰기·기호를 무시하고 찾는다("강 간", "강.간" 도 잡힌다)
    "강간", "윤간", "성폭행", "성폭력", "성추행", "강제추행", "성희롱", "몰카", "불법촬영", "그루밍범죄",
    "근친상간", "수간", "시간증", "아동포르노", "아동성착취", "미성년자성", "로리콘", "쇼타콘", "따먹",
    "섹스", "섹드립", "섹파", "야동", "포르노", "음란", "외설", "페티시", "노출증", "스와핑", "콘돔",
    "오르가슴", "자위행위", "발기", "정액", "애액", "유두", "젖꼭지", "성기", "음경", "고환", "질내",
    "삽입", "교미", "발정", "떡치", "야스", "19금", "성인물", "에로물", "변태성", "딜도", "사까시",
    "자살", "자해",
)
_BAN_TIGHT = ("좆", "씨발", "씹할", "씹새", "병신", "지랄", "개새끼", "니미", "썅", "창녀", "화냥")
_BAN_TIGHT_RE = re.compile(r"시발(?!점|역|탄)")  # '시발점'(시작점)은 정상 낱말이라 뺀다
_BAN_EN = re.compile(
    r"\b(?:rape|raping|molest|incest|bestiality|porn|porno|nsfw|orgasm|masturbate|masturbation|"
    r"blowjob|handjob|dildo|horny|boobs|tits|pussy|dick|penis|vagina|nipple|naked|sex|sexual|cum|slut|whore)\b",
    re.I,
)
_SQUEEZE = re.compile(r"[^0-9A-Za-z가-힣]+")


def _unsafe(text: str) -> bool:
    """화면에 올리면 안 되는 문장인가."""
    low = unicodedata.normalize("NFKC", text or "").lower()
    if not low.strip():
        return True
    if any(w in low for w in _BAN_TIGHT) or _BAN_TIGHT_RE.search(low):
        return True
    if _BAN_EN.search(low):
        return True
    squeezed = _SQUEEZE.sub("", low)
    return any(w in squeezed for w in _BAN_LOOSE)


def _safe_bubbles(bubbles: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [b for b in bubbles if not _unsafe(b[0])]


@app.post("/rate")
async def rate(req: RateReq):
    # 아이템 이름만 보내면 "윈디 헤어(여)" 처럼 생김새 정보가 없어 모델이 헤어를 무시하고
    # 이름이 튀는 옷/장식에만 반응했다(헤어를 바꿔도 같은 말이 나오던 원인).
    # → 캡션(words)을 함께 붙여 "모습"을 보고 말하게 한다.
    worn, words_of = [], {}
    for it in req.items:
        if not it.name:
            continue
        line = f"{SLOT_KO.get(it.slot, it.slot)}: {it.name}"
        meta = BY_ID.get(it.id or "")
        if meta and meta.get("words"):
            line += f" — {', '.join(meta['words'])}"
            words_of[it.slot] = list(meta["words"])
        worn.append(line)
    # 말풍선 개수: 보통 2개, 가끔(셋에 한 번) 3개. 모델에 맡기면 늘 2개만 나왔다.
    n = 3 if random.random() < 0.33 else 2
    # ── 매번 다른 말이 나오게: **어디를 볼지는 코드가 뽑는다.**
    # 모델에 맡기면 같은 코디에는 늘 같은 부위의 같은 낱말을 집어 같은 말을 한다("방금 한 말"을 줘도, 온도를 높여도 그랬다).
    # 그래서 입은 부위 가운데 볼 부위를 n 개 무작위로 뽑고(바로 앞에 본 부위는 되도록 피한다), 부위마다 생김새 낱말 둘을
    # 무작위로 골라 "이번에 눈에 띈 것"으로 준다. 같은 코디라도 누를 때마다 다른 부위의 다른 생김새를 두고 말하게 된다.
    worn_slots = [it.slot for it in req.items if it.name]
    recent = set(req.recent_slots or [])
    fresh = [sl for sl in worn_slots if sl not in recent]
    pool = fresh if len(fresh) >= min(n, len(worn_slots)) else worn_slots
    focus = random.sample(pool, min(n, len(pool)))
    parts = ["착용 코디:\n" + ("\n".join(worn) if worn else "(아무것도 안 입은 맨몸이야!)")]
    hist = [h for h in (req.history or []) if h][:6]
    if hist:
        parts.append("방금 한 말(겹치지 마):\n" + "\n".join(f"- {h}" for h in hist))
    if focus:
        look = []
        for sl in focus:
            ws = words_of.get(sl) or []
            hint = ", ".join(random.sample(ws, min(2, len(ws))))
            look.append(f"- {SLOT_KO.get(sl, sl)}" + (f" (이번에 눈에 띈 것: {hint})" if hint else ""))
        parts.append("이번에 볼 곳(이 순서로 한 마디씩, s 에는 그 부위 이름을 적어):\n" + "\n".join(look))
    parts.append(f"말풍선 {n if len(focus) >= n or not focus else len(focus)}개로 말해줘.")
    n = n if len(focus) >= n or not focus else len(focus)
    user = "\n\n".join(parts)

    async def _ask(extra: str = "") -> str:
        async with httpx.AsyncClient(timeout=18.0) as client:
            r = await client.post(
                f"{DASHSCOPE_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
                json={
                    "model": RATE_MODEL,
                    "messages": [
                        {"role": "system", "content": PINKBEAN_SYSTEM},
                        {"role": "user", "content": user + extra},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.9,
                    "max_tokens": 400,
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    try:
        content = await _ask()
    except Exception as e:
        return {"bubbles": ["뀨…? 지금은 좀 부끄러운걸!", "이따 다시 보여줘…"], "slots": ["", ""], "error": str(e)[:120]}
    raw = _extract_bubbles(content)
    bubbles = _safe_bubbles(raw)          # 성적·폭력적 표현이 섞인 말풍선은 버린다
    dropped = len(raw) - len(bubbles)
    # 다 버려졌으면 더 강하게 못 박아 한 번만 더 받는다(두 번까지만 — 더 하면 응답이 너무 늦다).
    if not bubbles and raw:
        try:
            bubbles = _safe_bubbles(_extract_bubbles(await _ask(
                "\n\n⚠️ 성적인 말·성폭력을 뜻하는 말·욕설·폭력이나 자해를 암시하는 말은 절대 쓰지 마. "
                "옷이나 장식을 의인화할 때도 그런 비유를 쓰지 마. 밝고 귀엽게만 말해줘.")))
        except Exception:
            bubbles = []
    # 방금 한 말과 같은 말은 버린다(글자만 비교한다 — 띄어쓰기 · 문장부호가 달라도 같은 말이다).
    said = {_SQUEEZE.sub("", h) for h in hist}
    kept = [b for b in bubbles if _SQUEEZE.sub("", b[0]) not in said]
    if kept:
        bubbles = kept
    if len(bubbles) > n:      # 모델이 개수를 어기면 코드로 맞춘다
        bubbles = bubbles[:n][:3]
    if not bubbles:
        bubbles = [("뀨? 뭔가 신기한 코디인데?", ""), ("부농부농! 마음에 들어!", "")]
    # slots = 말마다 핑크빈이 보고 있는 부위(프런트의 부위 키. 입지 않은 부위나 모르는 이름이면 빈 문자열).
    # bubbles 와 같은 순서, 같은 길이다. 화면은 이것으로 그 부위의 아이템을 말 옆에 보여 준다.
    ko_slot = {v: k for k, v in SLOT_KO.items()}
    slots = [sl if sl in worn_slots else ko_slot.get(sl, "") for _, sl in bubbles]
    slots = [sl if sl in worn_slots else "" for sl in slots]
    return {"bubbles": [t for t, _ in bubbles], "slots": slots, "model": RATE_MODEL, "dropped": dropped}


# ── 기능 안내(POST /guide) ─────────────────────────────────────────
# front '도움 및 건의함'의 도움 탭과 건의 등록 전 확인이 쓴다. 기능 목록(catalog)은 **front 가 보낸다**(front/src/lib/guide.ts 가 정본).
# 모델은 목록에서 맞는 항목을 **고르기만** 한다 — 화면은 고른 항목의 글을 그대로 보여 주므로 없는 메뉴를 지어낼 수 없다.
# reply 는 고른 항목을 이어 주는 짧은 한두 문장이고, 걸러지면 화면이 항목 글만 보여 준다.
GUIDE_MODEL = os.environ.get("GUIDE_MODEL", "qwen-flash")
GUIDE_SYSTEM = (
    "너는 메이플스토리 코디 미리보기 사이트 '핑크빈 커마샵'의 안내 도우미야. 사용자의 말을 보고, 아래 '기능 목록'에서 "
    "그 말에 답이 되는 기능을 골라 준다.\n"
    "반드시 JSON 으로만 답해: {\"kind\": \"feature\" | \"none\" | \"chat\", \"ids\": [\"기능 id\"], \"type\": \"bug\" | \"idea\" | \"other\", \"reply\": \"말\"}\n"
    "판단하는 법: 사용자가 **하고 싶은 일**을 한 줄로 바꿔 보고, 목록의 각 기능 설명과 낱말(/ 뒤)을 보며 그 일을 해 주는 기능이 있는지 찾아라. "
    "낱말이 꼭 같지 않아도 하려는 일이 같으면 그 기능이다(예: '투명하게 저장' 은 이미지 저장, '작게 보고 싶다' 는 배율). "
    "보기 설정으로 풀리는 불편(캐릭터가 커서 잘려요, 작게 보고 싶어요)은 feature 다. "
    "어떤 아이템을 입었더니 무엇이 사라지거나, 안 보이거나, 뒤에 있거나, 색이 다르거나, 안 꺼진다는 글은 **오류 신고라 none** 이다(숨기기 같은 기능을 권하지 마라).\n"
    "**화면의 버튼 이름**(목록 설명의 [ ] 안 이름)을 사용자가 그대로 말하면 그 버튼의 기능이다. 비슷한 기능이 따로 있어도 그 버튼 이름이 적힌 기능 하나만 고른다"
    "(예: '모션' 이라고 하면 [모션] 버튼, '액션' 이라고 하면 [액션] 버튼).\n"
    "- feature: 그 일을 해 주는 기능이 목록에 있다. ids 에는 **가장 맞는 하나만** 적는다. 사용자가 서로 다른 두 가지를 한 번에 물었을 때만 둘을 적는다"
    "(헷갈린다고 비슷한 것을 함께 적지 마라). reply 에 어디서 어떻게 하는지 "
    "한두 문장으로 알려 줘. **목록에 적힌 내용만** 말해라. 목록에 없는 메뉴 이름, 버튼, 단계, 기능을 지어내지 마라.\n"
    "**순서: 언제나 목록에서 그 일을 해 주는 기능부터 찾는다.** '~기능이 있으면 좋겠어요', '~해 주세요' 처럼 바라는 말이어도 "
    "목록에 그 일을 하는 기능이 있으면 feature 다. 목록에 없을 때만 none 이다.\n"
    "- none: 목록의 어떤 기능으로도 그 일을 할 수 없다. ids 는 []. 그 말이 무엇인지 type 에 적는다. "
    "**먼저 이 사이트(캐릭터 코디를 입혀 보고 꾸미는 화면)에 대한 말인지 본다** — 아니면 무조건 other 다"
    "(날씨, 음식, 잡담, 메이플스토리 게임 자체의 공략 · 이벤트 · 일정 · 시세 등). 이 사이트에 대한 말일 때만 bug 나 idea 다:\n"
    "  · bug = 오류 신고(무엇이 안 되거나, 사라지거나, 안 보이거나, 이상하게 보이거나, 게임과 다르다). reply 는 \"\".\n"
    "  · idea = 사이트에 바라는 새 기능, 있는 기능을 바꿔 달라는 말, 이 사이트에 아직 없는 기능을 묻는 말(화면 모양 · 버튼 · 보기 방식에 대한 것 포함). reply 는 \"\".\n"
    "  · other = 사이트와 상관없는 말(날씨, 게임 공략, 잡담 등). reply 에 이 사이트의 기능만 안내할 수 있다는 것을 짧고 부드럽게 말하고 "
    "사이트에서 궁금한 것을 물어봐 달라고 해. 그 질문에 직접 답하지는 마라.\n"
    "  feature 와 chat 의 type 은 \"\" 다.\n"
    "- chat: 인사나 고맙다는 말뿐일 때만. ids 는 [], reply 에 짧게 답하고 무엇이 궁금한지 물어봐.\n"
    "- 말투는 담백한 존댓말 '~해요'. 느낌표, 이모지, 과장 없이. reply 는 공백 포함 90자 이내.\n"
    "- 생김새나 동작 자체를 새로 만들어 달라는 말(머리카락이 흔들리게, 새 표정 추가 등)은 비슷한 기능이 있어도 none 이다.\n"
    "- 성적인 말, 욕설, 폭력적인 말은 쓰지 마라."
)
GUIDE_CHECK = (
    "\n\n지금 들어온 말은 '건의'로 올리려는 글이다. 건의가 바라는 일을 **목록의 기능으로 지금 할 수 있으면** feature 로 답하고, "
    "reply 는 '이미 있는 기능이에요.' 로 시작해 어디서 하는지 알려 줘. 바라는 것이 목록에 없는 새것(새 표정, 새 이펙트, 새 동작)이거나, "
    "있는 기능을 다르게 고쳐 달라는 말이거나, 오류 신고면 none 이다."
)


class GuideTurn(BaseModel):
    r: str = "u"   # u = 사용자, a = 안내
    t: str = ""


class GuideEntry(BaseModel):
    id: str
    t: str


class GuideReq(BaseModel):
    q: str
    mode: str = "ask"            # ask = 도움 탭, check = 건의 등록 전 확인
    history: list[GuideTurn] = []
    catalog: list[GuideEntry] = []


@app.post("/guide")
async def guide(req: GuideReq):
    q = (req.q or "").strip()[:300]
    cat = [e for e in req.catalog[:60] if e.id and e.t][:60]
    ids_ok = {e.id[:40] for e in cat}
    if not q or not cat:
        return {"kind": "none", "ids": [], "reply": ""}
    lines = "\n".join(f"- {e.id[:40]}: {e.t[:400]}" for e in cat)
    msgs = [{"role": "system", "content": GUIDE_SYSTEM + (GUIDE_CHECK if req.mode == "check" else "") + "\n\n기능 목록:\n" + lines}]
    # 앞의 대화(짧게) — "그건 어디 있어요?" 같은 이어지는 질문을 알아듣게
    for h in req.history[-6:]:
        t = (h.t or "").strip()[:300]
        if t:
            msgs.append({"role": "assistant" if h.r == "a" else "user", "content": t})
    msgs.append({"role": "user", "content": q})
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                f"{DASHSCOPE_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
                json={"model": GUIDE_MODEL, "messages": msgs, "response_format": {"type": "json_object"},
                      "temperature": 0, "max_tokens": 300},
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return {"kind": "error", "ids": [], "reply": "", "error": str(e)[:120]}
    try:
        m = re.search(r"\{.*\}", content, re.S)
        d = json.loads(m.group(0) if m else content)
    except Exception:
        d = {}
    ids = [i for i in (d.get("ids") or []) if isinstance(i, str) and i in ids_ok][:2]
    # 모델이 kind 자리에 기능 id 를 적는 일이 있다({"kind": "image", "ids": ["image"]}) — 맞는 id 를 골랐으면 기능 답으로 본다
    # 갈래(bug · idea · other)를 kind 자리에 적는 일도 있다({"kind": "other", "type": ""}) — 없는 기능 답으로 보고 그 갈래를 쓴다
    if d.get("kind") in ("bug", "idea", "other"):
        d["type"], d["kind"] = d["kind"], "none"
    kind = d.get("kind") if d.get("kind") in ("feature", "none", "chat") else ("feature" if ids else "none")
    reply = d.get("reply") if isinstance(d.get("reply"), str) else ""
    reply = reply.strip()[:140]
    if kind == "feature" and not ids:
        kind = "none"
    if kind != "feature":
        ids = []
    # 없는 기능의 갈래: bug(오류 신고 → 오류 칸으로) · idea(바라는 기능 → 건의 칸으로) · other(사이트와 상관없는 말 → reply 로 답한다)
    ntype = (d.get("type") if d.get("type") in ("bug", "idea", "other") else "idea") if kind == "none" else ""
    if (kind == "none" and ntype != "other") or _unsafe(reply):
        reply = ""
    return {"kind": kind, "ids": ids, "type": ntype, "reply": reply, "model": GUIDE_MODEL}


# ── 코디 조합 표본(POST /looks) ──────────────────────────────────────
# 넥슨 랭킹 캐릭터의 실제 코디 표본(parser looks/ 가 만들어 CDN looks/v1/looks.jsonl.gz 에 올린다)에서
# 아이템 여러 개를 함께 쓴 코디를 찾는다. 한 줄 = {i, n, sk, s}: i 번호(입은 사람이 많은 순), n 입은 사람 수, sk 피부 카드 id, s 스냅샷.
# 기동하면 뒤에서 받아 색인을 만든다 — 받기 전이나 실패하면 ready=False 로 답한다(검색 · 평가와 상관없이 이 기능만 꺼진다).
# 저장소에 커밋하지 않는다(표본은 30일마다 다시 만든다 — 넥슨 약관의 갱신 의무. 커밋하면 저장소가 매번 커진다).
import gzip
import threading
from collections import Counter, defaultdict

LOOKS_URL = os.environ.get("LOOKS_URL", "https://cdn.pinkbean-customize.com/looks/v1/looks.jsonl.gz")  # 파일 경로도 된다(로컬 확인)
LOOKS_DYES = 6  # 아이템마다 돌려줄 자주 쓴 염색 수
LOOKS = {"ready": False, "lines": [], "post": {}, "dyes": {}, "error": ""}


def _dye_key(s: dict, slot: str):
    """그 부위의 염색 — (종류, 값). 헤어 · 성형은 믹스(팔레트), 그 밖과 피부는 HSB. 없으면 None."""
    if slot in (s.get("dyePalette") or {}):
        return ("palette", json.dumps(s["dyePalette"][slot], sort_keys=True))
    if slot in (s.get("dyeHsb") or {}):
        return ("hsb", json.dumps(s["dyeHsb"][slot], sort_keys=True))
    return None


def _load_looks():
    try:
        t0 = time.time()
        if LOOKS_URL.startswith("http"):
            r = httpx.get(LOOKS_URL, timeout=180.0)
            r.raise_for_status()
            raw = gzip.decompress(r.content)
        else:
            with open(LOOKS_URL, "rb") as f:
                raw = gzip.decompress(f.read())
        lines, post, dyes = [], defaultdict(list), defaultdict(Counter)
        for i, l in enumerate(raw.splitlines()):
            if not l:
                continue
            o = json.loads(l)
            s, n, sk = o["s"], o["n"], o.get("sk")
            lines.append(l)  # 응답할 때만 푼다(바이트로 들고 있는 편이 가볍다)
            items = [(slot, iid) for slot, iid in (s.get("equipped") or {}).items() if iid]
            if sk:
                items.append(("skin", sk))
            for slot, iid in set(items):
                post[iid].append(i)
                dk = _dye_key(s, slot)
                if dk:
                    dyes[iid][dk] += n  # 입은 사람 수로 센다
        LOOKS["post"] = {k: np.asarray(v, dtype=np.int32) for k, v in post.items()}  # 줄 순서 = 많이 입은 순 → 교집합도 그 순서
        LOOKS["dyes"] = {k: [{"kind": kd, "v": json.loads(v), "n": c} for (kd, v), c in cnt.most_common(LOOKS_DYES)] for k, cnt in dyes.items()}
        LOOKS["lines"] = lines
        LOOKS["ready"] = True
        print(f"looks: {len(lines)}벌, 아이템 {len(post)}종, {time.time() - t0:.1f}초", flush=True)
    except Exception as e:
        LOOKS["error"] = str(e)[:200]
        print(f"looks: 적재 실패 {LOOKS['error']}", flush=True)


threading.Thread(target=_load_looks, daemon=True).start()


class LooksReq(BaseModel):
    ids: list[str]   # 아이템 id 1~4개(피부는 목록의 피부 카드 id) — 모두 쓴 코디만
    page: int = 0
    size: int = 30


@app.post("/looks")
def looks(req: LooksReq):
    ids = [x for x in dict.fromkeys(req.ids) if isinstance(x, str) and x][:4]
    page, size = max(0, int(req.page)), max(1, min(60, int(req.size)))
    base = {"ready": LOOKS["ready"], "total": 0, "page": page, "size": size, "looks": []}
    if not LOOKS["ready"] or not ids:
        return base
    arrs = [LOOKS["post"].get(x) for x in ids]
    if any(a is None for a in arrs):
        return {**base, **({"dyes": []} if len(ids) == 1 and page == 0 else {})}
    arrs.sort(key=len)
    hit = arrs[0]
    for a in arrs[1:]:
        hit = np.intersect1d(hit, a, assume_unique=True)
    sel = hit[page * size:(page + 1) * size]
    out = {**base, "total": int(len(hit)), "looks": [json.loads(LOOKS["lines"][int(i)]) for i in sel]}
    if len(ids) == 1 and page == 0:
        out["dyes"] = LOOKS["dyes"].get(ids[0], [])
    return out

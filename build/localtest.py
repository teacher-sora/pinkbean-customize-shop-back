"""로컬 검증: DashScope 임베딩 리전 확인 + 검색 스모크 테스트. (키는 env 로만)"""
import asyncio, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as A


async def main():
    if not A.QWEN_API_KEY:
        print("NO_KEY"); return
    # 1) 임베딩 리전 확인
    try:
        qv = await A.embed_query("세일러복에 절대영역 스타킹")
        print(f"EMBED_OK base={A.DASHSCOPE_BASE} dim={qv.shape[0]} norm={float((qv**2).sum())**.5:.3f}")
    except Exception as e:
        print(f"EMBED_FAIL {type(e).__name__}: {str(e)[:160]}"); return
    # 2) 검색 스모크(슬롯 필터 유/무)
    for slot in [None, "weapon", "hair"]:
        req = A.SearchReq(query="검은색 긴 세일러복 교복", slot=slot, topK=5)
        res = await A.search(req)
        print(f"\n[slot={slot}] ms={res['ms']}")
        for r in res["results"]:
            print(f"  {r['score']:.3f} {r['slot']:8} {r['id']} {r['name']}")

asyncio.run(main())

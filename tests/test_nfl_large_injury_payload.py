import unittest
from kalshi_external.player_collect import _get, API_ROOT


class Response:
    status_code=200
    headers={}
    def __init__(self,n):self.content=b"x"*n
    def json(self):return {"injuries":[]}


class Session:
    def __init__(self,n):self.n=n;self.calls=0
    def get(self,*args,**kwargs):
        self.calls+=1
        self.last_url=args[0]
        return Response(self.n)


class LargeNFLInjuryProvider(unittest.TestCase):
    def test_nfl_injuries_can_exceed_3mb_if_below_explicit_cap(self):
        s=Session(4_000_000)
        self.assertEqual(_get(s,f"{API_ROOT}/football/nfl/injuries"),{"injuries":[]})
        self.assertEqual(s.calls,1)

    def test_only_nfl_injury_endpoint_uses_large_cap(self):
        s=Session(4_000_000)
        with self.assertRaisesRegex(ValueError,"3000000-byte size limit"):
            _get(s,f"{API_ROOT}/basketball/nba/injuries")

    def test_nfl_payload_over_cap_is_rejected(self):
        s=Session(16_000_001)
        with self.assertRaisesRegex(ValueError,"16000000-byte size limit"):
            _get(s,f"{API_ROOT}/football/nfl/injuries")

if __name__=="__main__":
    unittest.main()

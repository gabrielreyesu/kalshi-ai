"""Official read-only Skorbot public API adapter, with required attribution.

Research observations only. No Skorbot content is used in model training
without establishing the rights to do so. Do not consume past final results
from this feed as if they were pre-match forecasts.
"""
from __future__ import annotations

import pandas as pd
import requests

from .core import future_match, make_record, utc

ENDPOINT="https://skorbot.live/api/v1/matches.json"
USER_AGENT="KalshiAI-Research/0.8 (+https://github.com/gabrielreyesu/kalshi-ai)"
CREDIT="Skorbot (skorbot.live)"


def parse_skorbot(payload, registry, captured):
    if not isinstance(payload,dict) or not isinstance(payload.get("matches"),list):
        raise ValueError("Malformed Skorbot response")
    if len(payload["matches"])>1000:
        raise ValueError("Excessive Skorbot match count")
    if payload.get("generated_at") and utc(payload["generated_at"])>utc(captured)+pd.Timedelta(minutes=5):
        raise ValueError("Future-dated Skorbot feed")
    results=[]
    for event in payload["matches"]:
        try:
            if event["status"]!="scheduled" or event.get("result") is not None:
                continue
            kick=utc(event["kickoff"])
            issued=utc(event["updated_at"])
            if issued>utc(captured) or issued>=kick-pd.Timedelta(minutes=15):
                continue
            found=None
            for sport in ("intl","club"):
                found=future_match(registry,sport,event["home"],event["away"],kick,captured)
                if found is not None:break
            if found is None:continue
            probabilities=event["probabilities"]
            p=[probabilities[k] for k in ("home","draw","away")]
            row=make_record(found,"skorbot_public_forecast",captured,p,
                            published=issued,source_kind="research_only")
            row["attribution"]=CREDIT
            results.append(row)
        except (KeyError,ValueError,TypeError):
            continue
    return results


def fetch_skorbot(session,registry,captured):
    try:
        res=session.get(ENDPOINT,headers={"User-Agent":USER_AGENT},timeout=15,allow_redirects=False)
        if res.status_code!=200 or len(res.content)>2_000_000:
            raise ValueError("Skorbot feed unavailable or oversized")
        return parse_skorbot(res.json(),registry,captured),[]
    except (requests.RequestException,ValueError,TypeError) as exc:
        return [],[f"Skorbot public API unavailable: {type(exc).__name__}"]

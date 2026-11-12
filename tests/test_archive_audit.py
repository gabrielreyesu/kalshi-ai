import json
import tempfile
import unittest
from pathlib import Path
from kalshi_external.archive_audit import audit_snapshot_archive, _load


def event():
    return {
        "source_id":"oddsapi_bookmaker_consensus","source_kind":"market",
        "sport":"nba","event_id":"nba-001",
        "kickoff_utc":"2026-11-13T20:00:00Z",
        "captured_utc":"2026-11-13T12:00:00Z",
        "published_utc":"2026-11-13T12:00:00Z",
        "p_H":.6,"p_D":None,"p_A":.4,
        "model_p_H":.52,"model_p_D":None,"model_p_A":.48,
    }


def weather():
    return {
        "source_id":"noaa_nws_hourly","source_kind":"environment",
        "sport":"nfl","event_id":"nfl-001",
        "kickoff_utc":"2026-11-13T20:00:00Z",
        "captured_utc":"2026-11-13T12:00:00Z",
        "published_utc":"2026-11-13T12:00:00Z",
        "temperature_f":42,"wind_speed_max_mph":12,
        "precip_probability_pct":40,
    }


class TestArchiveAudits(unittest.TestCase):
    def test_empty_is_valid(self):
        result=audit_snapshot_archive([],[])
        self.assertEqual(result["market_snapshots"],0)
        self.assertFalse(result["production_changed"])

    def test_realistic_snapshots_and_distinct_event_count(self):
        r=event()
        r2=dict(r);r2["captured_utc"]="2026-11-13T13:00:00Z"
        data=audit_snapshot_archive([r,r2],[weather()])
        self.assertEqual(data["market_snapshots"],2)
        self.assertEqual(data["weather_snapshots"],1)
        self.assertEqual(data["unique_sport_event_pairs"],2)

    def test_duplicate_same_timestamp_rejected(self):
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            audit_snapshot_archive([event(),event()],[])

    def test_invalid_bookmaker_probs_rejected(self):
        r=event();r["p_H"]=.88
        with self.assertRaisesRegex(ValueError,"bookmaker"):
            audit_snapshot_archive([r],[])

    def test_bad_model_probs_rejected(self):
        r=event();r["model_p_A"]=.9
        with self.assertRaisesRegex(ValueError,"baseline"):
            audit_snapshot_archive([r],[])

    def test_late_publication_and_capture_rejected(self):
        r=event();r["captured_utc"]="2026-11-13T19:59:00Z"
        with self.assertRaisesRegex(ValueError,"too close"):
            audit_snapshot_archive([r],[])
        r=event();r["published_utc"]="2026-11-13T12:01:00Z"
        with self.assertRaisesRegex(ValueError,"later"):
            audit_snapshot_archive([r],[])

    def test_weather_out_of_range_rejected(self):
        r=weather();r["wind_speed_max_mph"]=1000
        with self.assertRaisesRegex(ValueError,"wind_speed_max_mph"):
            audit_snapshot_archive([],[r])

    def test_corrupted_json_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"history.jsonl"
            p.write_text('{"valid":true}\nBROKEN\n')
            with self.assertRaisesRegex(ValueError,"Invalid JSON row 2"):
                _load(p)

    def test_real_3way_probs(self):
        r=event()
        r["sport"]="intl"
        r["p_H"]=.5;r["p_D"]=.2;r["p_A"]=.3
        r["model_p_H"]=.4;r["model_p_D"]=.3;r["model_p_A"]=.3
        self.assertEqual(audit_snapshot_archive([r],[])["sport_snapshots"]["intl"],1)


if __name__=="__main__":
    unittest.main()

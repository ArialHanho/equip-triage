import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.request import urlopen

from app import SAMPLE, classify_memo, render, replay


class ReplayTests(unittest.TestCase):
    def test_fixture(self):
        report = replay(SAMPLE)
        a, b, c = report["batches"]
        self.assertEqual([(x["equipment"], x["lot"]) for x in (a, b, c)],
                         [("SIM-A", "LOT-001"), ("SIM-B", "LOT-001"), ("SIM-C", "LOT-002")])
        self.assertEqual(a["sequence_gaps"], [{"from": 6, "to": 6}])
        self.assertEqual([e["sequence"] for e in a["events"]], [1, 2, 3, 4, 5, 7])
        self.assertEqual([a["status"], b["status"], c["status"]], ["판정 보류", "점검 후보", "코드 경고 없음"])
        self.assertEqual([n["kind"] for n in report["ingestion"]], ["duplicate", "late", "conflict", "invalid"])
        self.assertEqual(a["events"][3]["raw"]["time"], "2026-01-05T09:00:15+09:00")

    def test_bad_header(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.csv"
            p.write_text("lot,time\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                replay(p)

    def test_large_int_clock_reverse_unknown_missing_boundaries_and_escape(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "edge.csv"
            p.write_text("equipment,lot,sequence,time,code,memo\n"
                         "&SIM,LOT,1,2026-01-01T00:00:03Z,MEASURE,<script>\n"
                         "&SIM,LOT,2,2026-01-01T00:00:01Z,UNKNOWN,\n"
                         "&SIM,LOT,9999999999999999999999,2026-01-01T00:00:04Z,END,\n", encoding="utf-8")
            data = replay(p)
        self.assertEqual(data["ingestion"][0]["kind"], "invalid")
        self.assertEqual(data["batches"][0]["status"], "판정 보류")
        self.assertEqual(data["batches"][0]["reasons"], [
            "시작 기록 미확인", "종료 기록 미확인", "미등록 코드", "기록 시각과 순번 불일치"])
        self.assertIn("&lt;script&gt;", render(data))
        self.assertNotIn("<script>", render(data))

    def test_memo_rule_and_abstain(self):
        self.assertEqual(classify_memo("통신 TIMEOUT"), ("통신 기록 확인", "통신, timeout"))
        self.assertEqual(classify_memo("읽을 내용 없음")[0], "분류 보류")
        self.assertEqual(classify_memo("접촉과 통신 둘 다 확인")[0], "분류 보류")


class EntrypointTests(unittest.TestCase):
    def test_cli_json(self):
        result = subprocess.run([sys.executable, "app.py", "--json"], check=True, capture_output=True, text=True)
        self.assertEqual(len(json.loads(result.stdout)["batches"]), 3)

    def test_http_dashboard_and_json(self):
        proc = subprocess.Popen([sys.executable, "app.py", "--serve", "--port", "0"], stdout=subprocess.PIPE, text=True)
        try:
            url = proc.stdout.readline().strip()
            self.assertTrue(url.startswith("http://127.0.0.1:"))
            with urlopen(url + "?batch=1", timeout=5) as response:
                self.assertEqual(response.status, 200)
                html = response.read().decode("utf-8")
            self.assertIn("<h2>SIM-B / LOT-001</h2>", html)
            self.assertNotIn("<h2>SIM-A / LOT-001</h2>", html)
            with urlopen(url + "report.json", timeout=5) as response:
                self.assertEqual(len(json.loads(response.read())["batches"]), 3)
        finally:
            proc.terminate()
            proc.wait(timeout=5)
            proc.stdout.close()


if __name__ == "__main__":
    unittest.main()

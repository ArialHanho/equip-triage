"""Offline inspection event replay, using only the Python standard library."""
import argparse
import csv
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

SAMPLE = Path(__file__).parent / "examples" / "events.csv"
FIELDS = ("equipment", "lot", "sequence", "time", "code", "memo")
CODES = {"START": "검사 시작", "CHECK": "기준 샘플 확인", "CONTACT_WARN": "접촉 상태 경고",
         "MEASURE": "측정", "RETEST": "재검", "END": "검사 종료", "LINK_LOST": "통신 끊김"}
MEMO_RULES = {
    "측정·접촉 확인": ("프로브", "접촉", "재측정", "측정값", "튐"),
    "통신 기록 확인": ("연결", "통신", "패킷", "timeout"),
    "설정 변경 확인": ("레시피", "설정", "recipe", "조건 변경"),
}


def classify_memo(memo):
    matches = {label: [word for word in words if word in memo.casefold()]
               for label, words in MEMO_RULES.items()}
    matches = {label: words for label, words in matches.items() if words}
    if len(matches) == 1:
        return next(iter(matches)), ", ".join(next(iter(matches.values())))
    return "분류 보류", "점검 근거 부족" if not matches else "여러 점검 항목에 해당"


def normalize(raw):
    equipment, lot, seq, stamp, code, memo = (
        (raw.get(key) or "").strip() for key in FIELDS
    )
    if not equipment or not lot or not code:
        raise ValueError("장비 ID, lot 및 코드는 필수입니다.")
    seq = int(seq)
    if not 1 <= seq <= 2**63-1:
        raise ValueError("순번은 양의 64비트 정수여야 합니다.")
    dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if dt.utcoffset() is None:
        raise ValueError("시각에 시간대가 필요합니다.")
    return equipment, lot, seq, dt.astimezone(timezone.utc).isoformat(timespec="microseconds"), code.upper(), memo


def replay(path):
    with closing(sqlite3.connect(":memory:")) as db:
        db.row_factory = sqlite3.Row
        db.executescript("""
            CREATE TABLE raw(line INTEGER PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE event(
                equipment TEXT NOT NULL, lot TEXT NOT NULL, sequence INTEGER NOT NULL,
                time TEXT NOT NULL, code TEXT NOT NULL, memo TEXT NOT NULL,
                line INTEGER NOT NULL REFERENCES raw(line),
                PRIMARY KEY(equipment, lot, sequence)
            );
        """)
        notes = []
        highest = {}
        with Path(path).open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or []) != FIELDS:
                raise ValueError("CSV 헤더는 " + ",".join(FIELDS) + " 순서여야 합니다.")
            for line, raw in enumerate(reader, 2):
                payload = json.dumps(raw, ensure_ascii=False)
                db.execute("INSERT INTO raw VALUES (?,?)", (line, payload))
                try:
                    if None in raw:
                        raise ValueError("열이 너무 많습니다.")
                    value = normalize(raw)
                except (ValueError, OverflowError) as err:
                    notes.append({"kind": "invalid", "line": line, "detail": str(err), "raw": raw})
                    continue
                equipment, lot, seq, stamp, code, memo = value
                scope = equipment, lot
                key = *scope, seq
                dup = db.execute("SELECT time,code,memo FROM event WHERE equipment=? AND lot=? AND sequence=?", key).fetchone()
                if dup:
                    category = "duplicate" if tuple(dup) == (stamp, code, memo) else "conflict"
                    notes.append({"kind": category, "line": line, "equipment": equipment, "lot": lot,
                                  "sequence": seq, "raw": raw})
                    continue
                if seq < highest.get(scope, 0):
                    notes.append({"kind": "late", "line": line, "equipment": equipment, "lot": lot,
                                  "sequence": seq, "raw": raw})
                highest[scope] = max(highest.get(scope, 0), seq)
                db.execute("INSERT INTO event VALUES (?,?,?,?,?,?,?)", (*value, line))
        batches = {}
        for record in db.execute("SELECT e.*, r.payload FROM event e JOIN raw r USING(line) ORDER BY e.time, e.equipment, e.lot, e.sequence"):
            row = dict(record)
            row["raw"] = json.loads(row.pop("payload"))
            row["meaning"] = CODES.get(row["code"], "미등록 코드")
            row["memo_label"], row["memo_evidence"] = classify_memo(row["memo"]) if row["memo"] else ("", "")
            batches.setdefault((row["equipment"], row["lot"]), []).append(row)
        result = []
        for (equipment, lot), events in sorted(batches.items()):
            sequences = sorted(event["sequence"] for event in events)
            starts = [0] + sequences
            gaps = [{"from": prev+1, "to": curr-1} for prev, curr in zip(starts, sequences) if curr-prev > 1]
            by_seq = sorted(events, key=lambda e: e["sequence"])
            closed = by_seq[-1]["code"] == "END"
            has_start = by_seq[0]["code"] == "START"
            clock_back = any(a["time"] > b["time"] for a, b in zip(by_seq, by_seq[1:]))
            unknown = any(event["code"] not in CODES for event in events)
            conflict = any(note["kind"] == "conflict" and (note["equipment"], note["lot"]) == (equipment, lot) for note in notes)
            candidates = [{"line": e["line"], "text": CODES[e["code"]]} for e in events
                          if e["code"] in {"CONTACT_WARN", "LINK_LOST", "RETEST"}]
            reasons = [label for active, label in (
                (gaps, "순번 누락"), (not has_start, "시작 기록 미확인"),
                (not closed, "종료 기록 미확인"), (unknown, "미등록 코드"),
                (conflict, "동일 순번의 기록 충돌"), (clock_back, "기록 시각과 순번 불일치"),
            ) if active]
            result.append(dict(equipment=equipment, lot=lot, events=events, candidates=candidates,
                               sequence_gaps=gaps, reasons=reasons,
                               status="판정 보류" if reasons else "점검 후보" if candidates else "코드 경고 없음"))
        return dict(batches=result, ingestion=notes)


def render(report, selected=None):
    views = []
    menu = []
    batches = report["batches"]
    active = next((str(i) for i in range(len(batches)) if str(i) == selected), "0")
    for i, batch in enumerate(batches):
        tag = escape(batch["equipment"] + " / " + batch["lot"])
        menu.append(f'<a href="/?batch={i}" class="{"selected" if str(i)==active else ""}">{tag}</a>')
        if str(i) != active:
            continue
        alerts = "".join(f'<li>CSV {x["line"]}행 · {escape(x["text"])}</li>' for x in batch["candidates"]) or "<li>코드 기준 확인 후보 없음</li>"
        gaps = ", ".join(f'{g["from"]}–{g["to"]}' for g in batch["sequence_gaps"])
        reason = " · ".join(batch["reasons"]) + (f" ({gaps})" if gaps else "")
        rows = []
        for event in batch["events"]:
            memo = escape(event["memo"] or "—")
            label = escape(event["memo_label"])
            evidence = escape(event["memo_evidence"])
            raw = escape(json.dumps(event["raw"], ensure_ascii=False, indent=2))
            rows.append(f'<tr><td>{event["sequence"]}</td><td>{escape(event["time"][11:19])}</td>'
                        f'<td>{escape(event["meaning"])}<small>{escape(event["code"])}</small></td>'
                        f'<td>{memo}<small>{label} {"· " + evidence if evidence else ""}</small></td>'
                        f'<td><details><summary>CSV {event["line"]}행</summary><pre>{raw}</pre></details></td></tr>')
        status_style = {"판정 보류": "hold", "점검 후보": "warn", "코드 경고 없음": "clear"}[batch["status"]]
        views.append(f'<section><h2>{tag}</h2><strong class="{status_style}">{escape(batch["status"])}</strong>'
                     f'<p>{escape(reason)}</p><h3>먼저 확인할 근거</h3><ul>{alerts}</ul>'
                     '<div class="table"><table><thead><tr><th>순번</th><th>UTC 시각</th><th>이벤트</th>'
                     '<th>작업 메모 · 키워드 분류</th><th>원본</th></tr></thead><tbody>'
                     f'{"".join(rows)}</tbody></table></div></section>')
    kinds = {"late": "늦게 도착", "conflict": "충돌 (첫 수신 기록 유지)", "duplicate": "중복 제외", "invalid": "형식 오류"}
    intake = "".join(
        f'<li><details><summary>CSV {n["line"]}행 · {kinds[n["kind"]]}'
        f'{" · " + escape(n["detail"]) if n.get("detail") else ""}</summary>'
        f'<pre>{escape(json.dumps(n["raw"], ensure_ascii=False, indent=2))}</pre></details></li>'
        for n in report["ingestion"]
    ) or "<li>수신 품질 이슈 없음</li>"
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Equip Triage | 검사장비 로그 재현</title><style>
    :root{{font-family:system-ui,sans-serif;color:#162845;background:#edf2f7}}body{{margin:0}}header{{background:#12233d;color:#fff;padding:2rem max(5%,calc((100vw - 72rem)/2))}}h1{{margin:.3em 0}}main{{max-width:72rem;margin:auto;padding:1rem}}nav{{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1rem}}nav a{{padding:.5rem 1em;border:1px solid #7590ab;border-radius:.4rem;color:#183552;text-decoration:none}}nav .selected{{background:#173e61;color:white}}section{{background:white;border:1px solid #c2d0dd;border-radius:.7rem;padding:1.2rem;margin-bottom:1rem}}strong{{background:#e1eef9;padding:.4rem .7rem;border-radius:.3rem}}strong.hold{{background:#fde7eb;color:#80172c}}strong.warn{{background:#fff2d3;color:#654300}}strong.clear{{background:#e7f4ed;color:#15593d}}.table{{overflow:auto}}table{{width:100%;border-collapse:collapse;text-align:left}}td,th{{border-bottom:1px solid #e1e8ee;padding:.7em;vertical-align:top}}small{{display:block;color:#466377}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-width:22rem}}.eyebrow{{font-size:.85rem;color:#bedcf5}}
    </style></head><body><header><span class="eyebrow">EQUIP TRIAGE · 합성 데이터 데모</span><h1>검사장비 로그 재현</h1><p>lot별 장비 이벤트와 작업 메모를 모아 초기 점검 근거를 보여줍니다.</p></header>
    <main><nav aria-label="장비 및 lot">{"".join(menu)}</nav>{"".join(views) or "<section>표시 가능한 이벤트가 없습니다.</section>"}
    <section><h2>로그 수신 품질</h2><ul>{intake}</ul></section>
    <p>공정 원인 확정, 품질 판정, 장비 제어를 하지 않습니다.</p></main></body></html>'''


def serve(report, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            if url.path not in ("/", "/report.json"):
                self.send_error(404)
                return
            if url.path == "/report.json":
                body = json.dumps(report, ensure_ascii=False).encode()
                content = "application/json"
            else:
                body = render(report, parse_qs(url.query).get("batch", [None])[0]).encode()
                content = "text/html"
            self.send_response(200)
            self.send_header("Content-Type", content + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SAMPLE, help="Synthetic UTF-8 CSV to replay")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--serve", action="store_true", help="Serve read-only dashboard on localhost")
    mode.add_argument("--json", action="store_true", help="Print normalized report (default)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        report = replay(args.input)
    except (OSError, csv.Error, ValueError) as err:
        parser.error(str(err))
    if args.serve:
        serve(report, args.port)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

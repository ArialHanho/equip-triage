"""Original bilingual vector artwork for the two repository READMEs. Standard library only."""

from pathlib import Path
from xml.sax.saxutils import escape

ASSETS = Path(__file__).resolve().parents[2] / "docs" / "assets"
INK = "#152f42"
MUTED = "#4e6573"
RULE = "#c4d0d6"
BLUE = "#1d5982"
TEAL = "#00665d"
AMBER = "#974b19"


def draw(lang):
    ko = lang == "ko"

    def tr(en, kr):
        return kr if ko else en

    v = []

    def text(x, y, s, size=17, color=INK, weight=400, anchor="start", formula=False):
        font = (
            "DejaVu Sans, Noto Sans CJK KR, sans-serif"
            if formula
            else "Inter, Noto Sans CJK KR, DejaVu Sans, sans-serif"
        )
        v.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" font-family="{font}" font-size="{size}" font-weight="{weight}">{escape(s)}</text>'
        )

    def box(x, y, w, h, fill="white", stroke=RULE, r=5):
        v.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>'
        )

    def line(x1, y1, x2, y2, arrow=False, col=RULE):
        mark = ' marker-end="url(#arrow)"' if arrow else ""
        v.append(
            f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{col}" stroke-width="{1.8 if arrow else 1}" fill="none"{mark}/>'
        )

    def section(letter, y, label):
        line(30, y - 22, 1010, y - 22)
        box(30, y - 8, 31, 29, INK, INK, 3)
        text(45.5, y + 13, letter, 18, "white", 650, "middle")
        text(76, y + 14, label, 22, INK, 650)

    title = tr(
        "How two check candidates read a bilingual memo",
        "두 점검 후보가 한·영 메모를 읽는 과정",
    )
    desc = tr(
        "Original architecture diagram. Bind two memo lines to registered checks. A shared local encoder creates token states. Each candidate reads its matched tokens through four-head attention. Route only a single continuing issue with the other cleared; otherwise hold.",
        "직접 그린 구조도. 메모 두 줄을 등록된 점검에 연결하고, 공유 인코더가 줄마다 토큰 표현을 만듭니다. 각 후보는 네 개의 어텐션 헤드로 연결된 토큰을 읽습니다. 하나만 지속되고 다른 하나는 해소됐을 때 해당 점검으로 연결합니다.",
    )
    v.extend(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="814" viewBox="0 0 1040 814" role="img" aria-labelledby="title description">',
            '<title id="title">' + escape(title) + "</title>",
            '<desc id="description">' + escape(desc) + "</desc>",
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1 1 L9 5 L1 9 z" fill="#56717e"/></marker></defs>',
            '<rect width="1040" height="814" fill="#fff"/>',
        ]
    )
    text(30, 43, title, 32, INK, 700)
    text(
        31,
        72,
        tr(
            "Masked candidate-query model  ·  Illustration using the expected states of a synthetic packet",
            "후보별 마스크 어텐션 모델  ·  합성 기록의 정답 상태로 본 판정 원리",
        ),
        17,
        MUTED,
    )
    section(
        "A",
        111,
        tr(
            "Link observed references to event slots",
            "기록에 나타난 지칭 표현을 점검 순번에 연결",
        ),
    )
    text(40, 161, tr("Recorded candidates", "등록된 점검 후보"), 16, MUTED, 600)
    text(
        335,
        161,
        tr("Memo in received line order", "수신 순서대로 놓은 메모"),
        16,
        MUTED,
        600,
    )

    def row(x, y, w, ident, label, accent, fill):
        box(x, y, w, 35, fill, "none", 4)
        box(x, y, 4, 35, accent, "none", 0)
        text(x + 14, y + 23, ident, 16, accent, 700)
        text(x + 62, y + 23, label, 16, INK)

    row(
        40,
        174,
        238,
        "C1",
        tr("L7 · Data collection", "L7 · 데이터 수집"),
        TEAL,
        "#eaf5f1",
    )
    row(40, 216, 238, "C2", tr("P3 · Measurement", "P3 · 측정계"), BLUE, "#eaf2f9")
    row(
        334,
        174,
        360,
        "L1",
        tr("P3 passed on retest.", "P3 재검 결과 정상."),
        BLUE,
        "#eaf2f9",
    )
    row(
        334,
        216,
        360,
        "L2",
        tr("L7 failed again on retest.", "L7 재검에서도 불량."),
        TEAL,
        "#eaf5f1",
    )
    line(288, 209, 319, 209, True, "#56717e")
    line(704, 209, 734, 209, True, "#56717e")
    text(
        749,
        157,
        tr("Allowed lines per candidate", "후보가 읽을 수 있는 줄"),
        16,
        MUTED,
        600,
    )
    for j, x in enumerate((840, 908)):
        text(x + 29, 178, "L" + str(j + 1), 14, MUTED, 600, "middle")
    for i, y in enumerate((186, 220)):
        text(807, y + 20, "C" + str(i + 1), 15, INK, 650, "middle")
        for j, x in enumerate((840, 908)):
            on = i != j
            box(
                x, y, 58, 28, (TEAL if i == 0 else BLUE) if on else "#f2f5f6", "none", 3
            )
            text(
                x + 29,
                y + 20,
                str(int(on)),
                16,
                "white" if on else MUTED,
                650,
                "middle",
            )
    section(
        "B",
        296,
        tr("Read each candidate’s evidence", "후보마다 연결된 근거를 읽어 상태를 계산"),
    )
    box(40, 338, 245, 156, "#f4f7f9")
    box(333, 338, 307, 156, "#fbfcfd")
    box(690, 338, 309, 156, "#f1f7f5")
    text(54, 367, tr("Clause-local BERT [1]", "줄별 BERT 인코더 [1]"), 18, INK, 650)
    text(
        54, 394, tr("Shared weights · two blocks", "가중치 공유 · 블록 2개"), 15, MUTED
    )
    text(54, 426, "H  [B, 2, L, 32]", 18, BLUE, 600, formula=True)
    text(
        54,
        459,
        tr("Each line is encoded alone.", "각 줄을 독립적으로 인코딩"),
        15,
        MUTED,
    )
    text(347, 367, tr("Candidate queries [2, 3]", "후보별 쿼리 [2, 3]"), 18, INK, 650)
    text(
        347,
        396,
        tr("Q = Wq(domain + slot)", "Q ← 범주·순번 임베딩에서 투영"),
        15,
        MUTED,
    )
    text(347, 424, "Q       [B, 4, 2, 8]", 16, INK, 500, formula=True)
    text(347, 450, "K, V   [B, 4, 2L, 8]  ← H", 16, INK, 500, formula=True)
    text(
        347,
        477,
        tr("Four heads · eight features per head", "헤드 4개 · 헤드당 8차원"),
        14,
        MUTED,
    )
    text(
        704,
        367,
        tr("Masked token pooling", "마스크를 적용한 토큰 가중합"),
        18,
        INK,
        650,
    )
    text(704, 399, "S = QKᵀ / √8", 19, TEAL, 600, formula=True)
    text(704, 428, "A = softmax(mask(S))", 17, INK, 500, formula=True)
    text(704, 454, "Z = Wₒ concat(A V)", 17, INK, 500, formula=True)
    text(704, 479, "Z   [B, 2, 32]", 15, MUTED, 500, formula=True)
    line(292, 415, 321, 415, True, "#56717e")
    line(648, 415, 679, 415, True, "#56717e")
    text(
        42,
        520,
        tr(
            "The mask excludes other lines, header tokens and padding. Ablations vary eligible lines and learned vs uniform weights.",
            "마스크는 다른 줄·헤더·패딩을 제외합니다. 대조 실험에서는 줄의 범위와 학습 가중치·균등 가중치를 바꿉니다.",
        ),
        15,
        MUTED,
    )
    section(
        "C",
        562,
        tr(
            "Count continuing checks, then apply the validation cutoff",
            "지속 상태인 후보를 센 뒤 검증 임계값 적용",
        ),
    )
    box(40, 603, 255, 120, "#f4f7f9")
    box(339, 603, 382, 120, "#fff")
    box(765, 603, 234, 120, "#fcf5ec")
    text(54, 630, tr("Shared state head", "공유 상태 분류기"), 18, INK, 650)
    text(54, 659, "LN(E + Z) → 3", 18, INK, 500, formula=True)
    text(54, 684, tr("continues · cleared · unknown", "지속 · 해소 · 불명"), 15, MUTED)
    text(54, 709, "logits   [B, 2, 3]", 15, BLUE, 500, formula=True)
    text(353, 630, tr("Expected decisions", "정답 상태를 넣은 결과"), 18, INK, 650)
    text(
        353,
        659,
        tr("C1 continues + C2 cleared → C1", "C1 지속 + C2 해소 → C1 점검"),
        16,
        TEAL,
        650,
    )
    text(
        353,
        687,
        tr("Both continue → hold for review", "둘 다 지속 → 판정 보류"),
        16,
        AMBER,
        650,
    )
    text(
        353,
        711,
        tr(
            "Unknown, zero active or invalid input → hold",
            "하나라도 불명·둘 다 해소·기록 불완전 → 보류",
        ),
        14,
        MUTED,
    )
    text(779, 630, tr("Confidence gate", "신뢰도 기준"), 18, INK, 650)
    text(779, 659, tr("Use the lower of the two", "두 후보 중 더 낮은"), 15, MUTED)
    text(779, 682, tr("state confidences.", "상태 신뢰도를 사용"), 15, MUTED)
    text(779, 710, tr("Below cutoff → hold", "임계값 미만 → 보류"), 16, AMBER, 650)
    line(302, 663, 328, 663, True, "#56717e")
    line(728, 663, 754, 663, True, "#56717e")
    line(30, 746, 1010, 746)
    text(
        30,
        770,
        tr(
            "B = batch size · L = padded line length · E = candidate embedding · M broadcasts over the four heads.",
            "B = 배치 크기 · L = 패딩 후 줄 길이 · E = 후보 임베딩 · 마스크 M은 네 헤드에 공통으로 적용합니다.",
        ),
        15,
        MUTED,
    )
    text(
        30,
        797,
        tr(
            "[1] Devlin et al. (2019)   [2] Wang et al. (2016)   [3] Liu et al. (2021). Sources in the study report.",
            "[1] Devlin 외 (2019)   [2] Wang 외 (2016)   [3] Liu 외 (2021). 원문 링크는 연구 문서에 있습니다.",
        ),
        14,
        MUTED,
    )
    v.append("</svg>")
    return "\n".join(v) + "\n"


if __name__ == "__main__":
    for language in ("en", "ko"):
        (ASSETS / f"candidate-states-{language}.svg").write_text(
            draw(language), encoding="utf-8"
        )

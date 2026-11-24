# Equip Triage: 검사장비 로그 재현과 초동점검

검사장비 이벤트와 작업 메모를 lot별로 재현하고 **무슨 기록부터 확인할지** 원본 행과 함께 보여주는 로컬 포트폴리오 프로젝트입니다. 실제 장비·회사 자료는 사용하지 않았습니다. 자동 제어, 품질 판정, SECS/GEM 호환, 수율 개선 또는 양산 현장 검증을 주장하지 않습니다.

## 실행

Python 3.11 이상, 앱 실행에 별도 패키지 없음. 프로젝트 폴더에서:

```text
python app.py --serve
```

브라우저에서 <http://127.0.0.1:8765/>를 엽니다. Windows에서 `python`이 없다면 `py -3 app.py --serve`를 사용합니다. 서버는 로컬 주소만 사용하며, 끄려면 Ctrl+C를 누릅니다.

```text
python app.py --json
python app.py --input examples/events.csv --serve --port 8766
python -m unittest discover -s tests -v
```

- **SIM-A / LOT-001:** 접촉 경고와 재검 근거가 있으나 순번 누락 및 충돌 때문에 판정 보류. 충돌 시 첫 수신 기록은 원본과 함께 남기되 정상으로 간주하지 않습니다.
- **SIM-B / LOT-001:** A와 lot 번호가 같아도 별도 타임라인. 통신 단절 코드를 찾아 원본 근거를 표시합니다.
- **SIM-C / LOT-002:** 유효 이벤트에 등록된 경고가 없습니다. 전역 수신 품질에는 시간대 없는 행이 별도로 표시됩니다. 제품 합격 또는 완전한 데이터임을 뜻하지 않습니다.

## 입력과 한계

`equipment,lot,sequence,time,code,memo` 헤더의 UTF-8 CSV를 읽습니다. 순번은 장비·lot 안에서 1부터 시작하고 시각에는 UTC 오프셋 또는 Z가 필요합니다. 행 번호는 헤더를 포함한 CSV 레코드 번호입니다. 정규화 데이터와 원문 필드를 SQLite 메모리 DB에서 나눠 처리한 뒤 입력시각이 아닌 UTC 이벤트 시각으로 보여줍니다. 파일 재생이며 실시간 수집기나 영구 DB가 아닙니다.

메모는 **키워드 기준선**만 앱에 들어 있습니다. 근거가 없거나 둘 이상 범주가 겹치면 보류합니다. 앱 키워드 규칙은 측정·접촉, 통신, 설정 세 범주와 분류 보류를 사용합니다. 별도 NLP 연구의 라벨 정의와 데이터는 이 앱 규칙과 독립적이며, 실험 모델은 화면에 연결되지 않았습니다. 첫 버전은 낮은 운영 복잡도를 위해 읽기 전용으로만 동작합니다.

## 재현 연구와 공개 범위

- [연구 기록](research/RESEARCH.md): 합성 한국어 작업 메모 분류, 표현군을 분리한 데이터, 규칙 및 문자 TF-IDF 기준선, 무작위 초기화 이벤트 Transformer와 n-gram 비교.
- [작업 인덱스](INDEX.md) · [공개 전 확인](RELEASE_CHECKLIST.md) · [코드 라이선스](LICENSE)

## v0.2 참고 구현: 이벤트 시퀀스 Transformer

외부 사전학습 모델·가중치를 사용하거나 내려받지 않습니다. `research/event_transformer.py`는 이벤트 ID를 입력으로 다중 헤드 어텐션, 위치 정보, 잔차 연결, causal mask와 패딩 제외 손실을 직접 보여주는 **AI 작성 참고 코드**입니다. 앱 화면에는 아직 연결하지 않았습니다.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python research/event_transformer.py --out /tmp/event-results.json
python -m unittest discover -s research -p test_events.py -v
python research/long_context.py --out /tmp/long-context-results.json
```

Windows에서는 `/tmp/...` 대신 `event-local.json` 등 쓰기 가능한 경로를 사용합니다. 앱 자체에는 PyTorch가 필요하지 않습니다. [연구 및 Colab 재현](research/RESEARCH.md) · [직접 구현할 과제](CONTRIBUTIONS.md). 실행 가능한 self-contained [Colab용 노트북](notebooks/equip-triage-scratch.ipynb)도 포함합니다.

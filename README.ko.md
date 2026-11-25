[English](README.md) · **한국어**

# Equip Triage

**장비 이벤트 로그와 작업 메모를 함께 검토하는 오프라인 점검 도구**입니다. 로그는 장비·LOT별로 재구성하고, 순번 누락과 충돌 같은 이상을 원문과 함께 보여 줍니다. 메모의 문맥을 해석하기 위해 한·영 BERT 계열 트랜스포머도 PyTorch로 직접 구현했습니다.

[로그 화면](#csv-로그-검토) · [메모 모델 실행](#메모-모델-실행) · [구조도](#트랜스포머-구조) · [Python API](docs/MEMO_MODEL.md) · [검증 기록](docs/EVALUATION.md)

## CSV 로그 검토

Python 3.11 이상에서 별도 패키지 설치 없이 실행할 수 있습니다.

```sh
python app.py --serve
```

[한국어 화면](http://127.0.0.1:8765/?lang=ko)에서 `SIM-A / LOT-001`을 열면 순번 6 누락과 순번 7 충돌을 확인할 수 있습니다. CSV 열은 `equipment,lot,sequence,time,code,memo`이며, [예제 CSV](examples/events.csv)가 기본값입니다. 내 파일은 `python app.py --input path/to/events.csv --serve`로 열 수 있습니다. `python app.py --json`은 같은 검토 결과를 JSON으로 출력합니다.

**현재 연결 범위:** 이 화면에서 이벤트와 원문 메모를 같이 볼 수 있고, 경고 코드와 메모 키워드는 규칙으로 처리합니다. 아래 트랜스포머는 별도 실행 경로입니다. 등록된 점검 두 건과 메모 두 줄을 받아 후보별로 지속·해소·불명을 추정합니다. CSV 이벤트 순서를 어텐션에 넣거나 두 판정을 합치는 기능은 아직 없습니다.

## 메모 모델 실행

Python 3.11 이상에서 저장소 루트의 가상 환경을 활성화한 뒤 설치하고 실행합니다.

```sh
python -m pip install .
python -m equip_triage predict --model models/memo-demo.pt --input examples/memo-input.jsonl
```

[데모 가중치](models/memo-demo.pt)는 합성 기록 72건으로 만든 실행 확인용입니다. 예제의 출력은 `data`, `data`, `hold`, `hold`, `hold`입니다. [체크섬](models/SHA256SUMS)

```sh
python -m equip_triage train --data examples/memo-train.jsonl --out artifacts/memo.pt
```

이 명령으로 무작위 초기값에서 다시 학습할 수 있습니다. 현장 성능은 확인하지 않았습니다.

## 트랜스포머 구조

![토큰과 임베딩 위에 인코더 두 층을 쌓은 모델 구조도. 왼쪽은 상태 레이블로 학습할 때, 오른쪽은 저장한 가중치로 추론할 때의 출력입니다.](docs/assets/memo-transformer-ko.svg)

두 메모 줄은 따로 인코딩하며 같은 가중치를 씁니다. Q/K/V 투영, 패딩을 제외하는 양방향 어텐션, 두 잔차 연결, 층 정규화, GELU 피드포워드 층을 PyTorch로 직접 구현했습니다. 점검 후보마다 쿼리 하나를 만들어 등록 정보로 연결된 줄의 메모 토큰만 읽게 하고, 같은 분류기로 각 후보의 세 상태를 예측합니다. 마지막 점검 선택은 지속 건수와 점수 임계값으로 결정합니다. [편집 가능한 draw.io 원본 · 한글/영문 두 페이지](docs/assets/memo-transformer.drawio) · [모델 코드](equip_triage/model.py) · [도식 구성에 참고한 BERT 논문 그림 1](https://aclanthology.org/N19-1423.pdf#page=3)

```python
import json
from pathlib import Path
from equip_triage import load_checkpoint, predict

model = load_checkpoint("models/memo-demo.pt")
packet = json.loads(Path("examples/memo-input.jsonl").read_text(encoding="utf-8").splitlines()[1])
logits, attention = model(model.encode([packet]), return_attention=True)
print(predict(model, packet)["action"])  # data
```

[API, 텐서 차원, 체크포인트 형식, 직접 만든 데이터로 학습하는 방법](docs/MEMO_MODEL.md)

### 인코더 한 층 자세히 보기

![왼쪽: 셀프 어텐션과 GELU 피드포워드 층, 두 잔차 연결과 LayerNorm. 오른쪽: QKV 투영부터 패딩 마스크, Softmax, 값 결합, 출력 투영까지.](docs/assets/memo-encoder-ko.svg)

왼쪽 점선은 입력을 그대로 더하는 잔차 연결입니다. 덧셈 뒤에 LayerNorm을 적용합니다. 오른쪽은 셀프 어텐션을 펼친 것으로, 토큰마다 32차원인 입력을 한 번에 96차원으로 투영한 뒤 Q·K·V로 나눕니다. 각각 헤드 4개 × 8차원이고, 패딩 위치의 **키 점수**를 Softmax 전에 제외합니다. 앞뒤 순서를 가리는 인과 마스크는 쓰지 않습니다. 메모 두 줄은 같은 인코더를 쓰되 서로 섞이지 않습니다.

[인코더 코드](equip_triage/model.py#L39-L65) · [편집 가능한 draw.io 원본 · 한글/영문](docs/assets/memo-encoder.drawio) · [설계 참고: Transformer 원논문 그림 1·2](https://arxiv.org/pdf/1706.03762#page=3)

## 검증 기록

실행 검증과 낯선 표현에서의 성능은 별개입니다. [검증 기록](docs/EVALUATION.md)에서 이전 모델의 수치와 당시 코드·데이터 조건을 확인할 수 있습니다. 라이선스는 [MIT](LICENSE)입니다.

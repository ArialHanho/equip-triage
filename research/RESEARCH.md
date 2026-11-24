# 사전학습 없는 이벤트 시퀀스 연구

## 범위와 구현

최신 방향에 따라 기존 사전학습 선택 실험과 관련 의존성·실행 경로를 제거했다. 신규 학습·평가·데모는 외부 사전학습 모델이나 가중치를 사용·다운로드하지 않는다. 이 저장소는 학습 가중치도 저장하지 않는다.

`event_transformer.py`는 PyTorch 텐서 연산으로 4-head scaled dot-product attention, sinusoidal position, 2개 residual 연결, feed-forward를 구성한 9,129개 파라미터의 한 블록 모델이다. `nn.Transformer`나 `MultiheadAttention`을 사용하지 않는다. 라이브러리의 Linear, LayerNorm, Embedding, 자동미분과 AdamW는 사용한다. 전부 무작위 초기화한다.

입력 위치 t에는 BOS 또는 t 이전 이벤트만 있고 목표는 t의 이벤트다. 어텐션은 상삼각 미래 위치를 -inf로 마스킹한다. 손실은 shifted next-event cross entropy이며 패딩 target=-100을 제외한다. 이는 미래 양쪽을 참조하는 BERT식 마스킹 복원이 아니다. n-gram도 같은 train 시나리오만 학습하고 목표 이전 2개 이벤트까지 쓰는 trigram과 짧은 문맥 backoff, add-one smoothing을 쓴다. 최대 이벤트 surprise가 validation에서 고정한 임계값을 넘으면 후보로 표시한다. 각 시점 점수는 그 시점까지의 이벤트만 쓰지만 시나리오 전체의 최댓값 평가는 종료 뒤의 요약이다.

## 분할과 첫 파일럿

수작업 지정한 서로 다른 템플릿을 train/validation/test에 배정하고 측정 반복 횟수도 각각 1–6 / 7–9 / 10–12로 분리했다. 시나리오 내부 window를 랜덤 분할하지 않는다. 정상 12개로 학습, 검증 12개와 테스트 12개는 각각 정상 3개와 누락·조기 종료·미등록 WAIT 9개다. 같은 생성 규칙을 공유하므로 템플릿 분리가 실제 장비·작업자 독립성을 입증하지 않는다. train/test에 공통 이벤트와 짧은 패턴은 존재한다.

seed=42, 120 epochs, PyTorch 2.9.1+cpu, CPU 1 thread에서 실행했다. 결과: [event-results.json](event-results.json), 전체 합성 시나리오: [event_scenarios.csv](event_scenarios.csv).

| 방법 | 테스트 TP / FP / TN / FN | anomaly F1 |
|---|---|---:|
| trigram | 9 / 0 / 3 / 0 | 1.0 |
| 작은 Transformer | 9 / 0 / 3 / 0 | 1.0 |

작고 쉬운 인위적 이상을 두 방법 모두 맞혔다. 우위의 근거가 아니며 실제 현장 성능을 주장하지 않는다. 테스트 공개 후 추가 변경 평가는 독립 데이터가 필요하다.

## 별도 문맥 진단

`long_context.py`는 마지막 3개 관측 이벤트 LOAD, MEASURE, UNLOAD가 같아도 앞의 CHECK/RETEST 모드가 다음 목표를 결정하도록 만든 별도 진단이다. 측정 반복 길이를 분할별로 분리하고 목표 시점에는 관측 prefix만 전달한다. 최초 파일럿의 임계값·결과와 섞지 않는다. seed 42, 240 epochs 고정으로 검증·테스트 각 4건에서 trigram은 2/4, Transformer는 4/4 정답이었다. [실제 prefix·예측 기록](long-context-results.json)을 남겼다.

이는 길이가 제한된 n-gram이 놓치는 정보를 참고 모델이 기억할 수 있는지 확인하는 **생성 규칙에 맞춘 데모**다. 대상 토큰만 고른 소량 평가이며 전체 next-event 성능·이상 탐지 우위·실제 장비 적용성을 뜻하지 않는다. 시드 및 문맥 길이 비교, 더 강한 긴 문맥 기준선, 위치/어텐션 제거 실험과 독립 평가가 남아 있다.

## 재현 및 Colab

저장소 루트, Python 3.12 권장. 환경/가중치를 저장소에 넣지 않는다.

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python research/event_transformer.py --seed 42 --epochs 120 --out /tmp/events.json --export-csv /tmp/events.csv
python research/long_context.py --seed 42 --out /tmp/context.json
python -m unittest discover -s research -p test_events.py -v
python -m unittest discover -s tests -v
```

Windows에서는 `/tmp` 대신 쓰기 가능한 파일명을 사용한다. 결과의 경과 시간은 머신마다 달라진다. 첫 실험은 CPU 수초 내 실행되며 GPU가 필요하지 않다. torch 휠 설치는 프레임워크 설치이며 사전학습 가중치를 받지 않는다.

Colab에서는 [독립 노트북](../notebooks/equip-triage-scratch.ipynb)을 업로드하고 Python CPU 런타임에서 셀을 순서대로 실행한다. 이 노트북은 최초 seed 42 파일럿 동결본이고 장기 문맥 진단은 별도 소스다. Colab의 실제 브라우저 실행은 접속 오류로 검증하지 못했다. 노트북의 코드 셀 순차 실행은 클라우드 CPU에서 확인했다. 소스 재현은 위 명령을 사용한다.

## 메모 기준선 보존

`study.py`와 `memo_corpus.tsv`는 이전 v0.1의 별도 합성 한국어 메모 실험이다. 외부 사전학습 경로를 제거했으며 키워드/문자 TF-IDF + 로지스틱 회귀만 남겼다. 학습 24, 검증 12, 테스트 16건으로, 알려진 4종과 unknown을 포함한다. [pilot-results.json](pilot-results.json)의 macro-F1은 규칙 0.3500, TF-IDF 0.4917이며 보류·오류도 기록했다. 이벤트 시퀀스 실험과 과제·라벨·평가 단위가 달라 수치를 직접 비교하지 않는다.

```sh
python -m pip install -r research/requirements.txt
python -m unittest discover -s research -p test_study.py -v
python research/study.py --out /tmp/memo.json
```

## 검증과 기여

CLI와 실제 HTTP 진입점(`/`, `?batch=1`, `/report.json`)을 포함한 앱 6개, 메모 3개, 미래 비참조·패딩 손실·분할·n-gram prefix 검증 5개가 통과했다. 실장비 연결·실시간 운영·GPU·Colab 웹 실행은 검증하지 않았다. [기여 범위와 직접 구현 과제](../CONTRIBUTIONS.md)를 따른다. MIT 라이선스는 이 저장소 코드·직접 만든 합성 예시에 적용한다.

# Hindsight Experience Replay in GridWorld

64×64 GridWorld에서 DQN(Deep Q-Network)의 Hindsight Experience Replay(HER) 적용 여부와 보상 설계에 따른 학습 결과를 비교하는 프로젝트입니다.

## 실험 구성

- 시작 위치: `(0, 0)`
- 목표 위치: 에피소드마다 무작위 선택
- 행동 공간: 상·하·좌·우 4개
- 상태 입력: 정규화한 현재 위치와 목표 위치 `(x, y, goal_x, goal_y)`
- 학습 알고리즘: experience replay, target network, epsilon-greedy 탐색을 사용하는 DQN
- HER 전략: 에피소드에서 이후에 방문한 상태를 새로운 목표로 사용하는 future strategy

## 디렉터리

| 디렉터리 | 보상 설계 |
| --- | --- |
| `gridHER/` | 목표 도달 시 `0`, 그 외에는 `-1`인 sparse reward |
| `gridbadHER/` | 목표까지의 정규화된 맨해튼 거리의 음수를 사용하는 shaped reward |

각 디렉터리에는 다음 실험 코드가 들어 있습니다.

| 파일 | 설명 |
| --- | --- |
| `DQNnoHER.py` | 최대 128 step, HER 미사용 |
| `DQNyesHER.py` | 최대 128 step, HER 사용 |
| `DQNnoHER_256.py` | 최대 256 step, HER 미사용 |
| `DQNyesHER_256.py` | 최대 256 step, HER 사용 |
| `Qnetwork.py` | Q-network 정의 |
| `gridworld.py` | GridWorld 환경 정의 |
| `results/` | 실험별 성공률 그래프 |

## 설치

Python 3.11 환경을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r gridHER/requirements.txt
```

## 실행

실행하려는 실험 디렉터리로 이동한 뒤 스크립트를 실행합니다.

```bash
cd gridHER
python DQNnoHER.py
python DQNyesHER.py
python DQNnoHER_256.py
python DQNyesHER_256.py
```

거리 기반 보상 실험은 `gridbadHER` 디렉터리에서 같은 방식으로 실행할 수 있습니다.

```bash
cd ../gridbadHER
python DQNnoHER.py
python DQNyesHER.py
python DQNnoHER_256.py
python DQNyesHER_256.py
```

각 스크립트는 3,000개 에피소드를 학습하며 결과 그래프를 해당 디렉터리의 `results/128/` 또는 `results/256/`에 저장합니다.

## 주요 학습 설정

| 항목 | 값 |
| --- | ---: |
| Grid size | 64×64 |
| Episodes | 3,000 |
| Maximum steps | 128 또는 256 |
| Batch size | 64 |
| Replay buffer size | 100,000 |
| Discount factor (`gamma`) | 0.98 |
| Learning rate | 0.001 |
| Target update interval | 200 steps |
| HER future goals per transition | 4 |
| Random seed | 0 |

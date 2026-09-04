# Hindsight Experience Replay in GridWorld

64×64 GridWorld에서 목표 지점을 찾는 DQN(Deep Q-Network)을 학습하고, Hindsight Experience Replay(HER)의 적용 여부와 보상 설계에 따른 성공률을 비교하는 실험 프로젝트입니다.

## 실험 환경

- 시작 위치: `(0, 0)`
- 목표 위치: 에피소드마다 무작위 선택
- 행동 공간: 상·하·좌·우 4개
- 네트워크 입력: 정규화한 현재 좌표와 목표 좌표 `(x, y, goal_x, goal_y)`
- 네트워크 출력: 4개 행동에 대한 Q-value
- 학습 방식: experience replay, target network, epsilon-greedy 탐색을 사용하는 DQN

HER 버전은 에피소드에서 실제로 도달한 미래 상태를 새로운 목표로 재지정한 경험을 replay buffer에 추가합니다. 이를 통해 원래 목표에 도달하지 못한 경험도 학습에 활용합니다.

## 보상 설계

| 디렉터리 | 보상 |
| --- | --- |
| `gridHER/` | 목표 도달 시 `0`, 그 외 이동에는 `-1`을 부여하는 sparse reward |
| `gridbadHER/` | 목표까지의 맨해튼 거리를 최대 거리로 정규화한 뒤 음수로 부여하는 shaped reward |

각 디렉터리에서 HER 적용 여부와 에피소드당 최대 step 수를 바꾸어 실험할 수 있습니다.

## 파일 구성

```text
.
├── README.md
├── requirements.txt
├── gridHER/
│   ├── DQNnoHER.py
│   ├── DQNnoHER_256.py
│   ├── DQNyesHER.py
│   ├── DQNyesHER_256.py
│   ├── Qnetwork.py
│   ├── gridworld.py
│   └── results/
└── gridbadHER/
    ├── DQNnoHER.py
    ├── DQNnoHER_256.py
    ├── DQNyesHER.py
    ├── DQNyesHER_256.py
    ├── Qnetwork.py
    ├── gridworld.py
    └── results/
```

| 파일 | 설명 |
| --- | --- |
| `DQNnoHER.py` | 최대 128 step, HER 미사용 |
| `DQNyesHER.py` | 최대 128 step, future strategy HER 적용 |
| `DQNnoHER_256.py` | 최대 256 step, HER 미사용 |
| `DQNyesHER_256.py` | 최대 256 step, future strategy HER 적용 |
| `Qnetwork.py` | 4차원 입력을 받아 4개 행동의 Q-value를 출력하는 신경망 |
| `gridworld.py` | GridWorld 환경과 무작위 목표 생성 |
| `results/` | 학습 후 생성된 성공률 그래프 |

## 설치

Python 3.11 환경을 권장합니다. 저장소 루트에서 원하는 방식으로 가상환경을 만든 뒤 공통 의존성을 설치합니다.

### venv

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Conda

```bash
conda create -n her-grid python=3.11
conda activate her-grid
conda install pip
python -m pip install -r requirements.txt
```

## 실행

실행하려는 보상 실험 디렉터리로 이동한 뒤 원하는 스크립트를 실행합니다.

Sparse reward 실험:

```bash
cd gridHER
python DQNnoHER.py
python DQNyesHER.py
python DQNnoHER_256.py
python DQNyesHER_256.py
```

거리 기반 shaped reward 실험:

```bash
cd gridbadHER
python DQNnoHER.py
python DQNyesHER.py
python DQNnoHER_256.py
python DQNyesHER_256.py
```

각 스크립트는 3,000개 에피소드를 실행합니다. 에피소드별 성공 여부와 20-episode moving average를 나타내는 그래프는 실행한 디렉터리 아래에 저장됩니다.

- 128 step 실험: `results/128/dqn_noher.png`, `results/128/dqn_her.png`
- 256 step 실험: `results/256/dqn_noher.png`, `results/256/dqn_her.png`

## 주요 학습 설정

| 항목 | 값 |
| --- | ---: |
| Grid size | 64×64 |
| Episodes | 3,000 |
| Maximum steps per episode | 128 또는 256 |
| Batch size | 64 |
| Replay buffer size | 100,000 |
| Discount factor (`gamma`) | 0.98 |
| Learning rate | 0.001 |
| Target network update interval | 200 steps |
| HER future goals per transition | 4 |
| Random seed | 0 |

세부 설정은 각 학습 스크립트 상단의 상수에서 확인할 수 있습니다.

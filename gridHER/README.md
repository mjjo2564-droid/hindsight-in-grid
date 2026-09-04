# GridWorld DQN 및 HER 비교

64×64 GridWorld에서 목표 지점을 찾는 DQN(Deep Q-Network)을 학습하고, HER(Hindsight Experience Replay) 적용 여부에 따른 성공률을 비교하는 간단한 실험 프로젝트입니다.

## 실험 구성

- 시작 지점: `(0, 0)`
- 목표 지점: 에피소드마다 무작위로 선정
- 행동: 상·하·좌·우 4개
- 네트워크 입력: 정규화한 현재 좌표와 목표 좌표 `(x, y, goal_x, goal_y)`
- 보상: 목표 도달 시 `0`, 그 외의 이동은 `-1`
- 학습: experience replay, target network, epsilon-greedy 탐색을 사용하는 DQN

HER 버전은 에피소드에서 실제로 도달한 미래 상태를 새로운 목표로 재지정한 경험을 replay buffer에 추가합니다. 이를 통해 원래 목표에 도달하지 못한 경험도 학습에 활용할 수 있습니다.

## 파일 구성

| 파일 | 설명 |
| --- | --- |
| `DQNnoHER.py` | HER를 사용하지 않는 DQN 학습 |
| `DQNyesHER.py` | future strategy HER를 적용한 DQN 학습 |
| `Qnetwork.py` | 4차원 입력과 4개 행동 Q-value를 연결하는 신경망 |
| `gridworld.py` | GridWorld 환경과 무작위 목표 생성 |
| `requirements.txt` | 재현을 위해 고정한 Python 패키지 버전 |
| `results/` | 학습 후 생성되는 성공률 그래프 |

## 실행 환경

Python 3.11을 권장합니다. 의존성을 설치하려면 Python 패키지 관리자인 `pip`이 필요합니다. Conda를 사용하는 경우 다음과 같이 Python 환경과 `pip`을 준비할 수 있습니다.

```bash
conda create -n her-grid python=3.11
conda activate her-grid
conda install pip
python -m pip --version
python -m pip install -r requirements.txt
```

`python -m pip --version`이 버전 정보를 출력하면 `pip`이 준비된 것입니다. 이후 `requirements.txt`에 기록된 NumPy, PyTorch, Matplotlib을 설치합니다.

## 실행 방법

HER 없이 학습:

```bash
python DQNnoHER.py
```

HER를 적용하여 학습:

```bash
python DQNyesHER.py
```

각 실험은 기본으로 500개 에피소드를 실행합니다. 에피소드별 성공 여부와 20-episode moving average가 다음 파일에 저장됩니다.

- HER 미사용: `results/dqn_noher/success_rate.png`
- HER 사용: `results/dqn_her/success_rate.png`

## 주요 학습 설정

| 항목 | 값 |
| --- | ---: |
| Episodes | 500 |
| Maximum steps per episode | 128 |
| Batch size | 64 |
| Replay buffer size | 100,000 |
| Discount factor (`gamma`) | 0.98 |
| Learning rate | 0.001 |
| Target network update interval | 200 steps |
| HER future goals per transition | 4 |
| Random seed | 0 |

실험 설정은 `DQNnoHER.py`와 `DQNyesHER.py` 상단의 상수로 변경할 수 있습니다.

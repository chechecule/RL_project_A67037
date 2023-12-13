# RL_project_A67037


본 코드는 코랩으로 작성된 것 임을 알려드립니다.


파일 하나로 작성되었으므로 따로 조정없이 실행하시면 됩니다. 


**환경 및 결과에 대한 자세한 설명은 보고서 통하여 확인하실 수 있습니다.**


감사합니다.


# 1. 함수 Import

```python
import gym
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
```

# 2. 환경 생성 & Hyperparameters 설정

```python
env = gym.make('Taxi-v3')

alpha = 0.1 #Alpha (α) - 학습률(Learning Rate)
gamma = 0.9 # Gamma (γ) - 할인 계수(Discount Factor)
epsilon = 0.0  # 항상 최적의 행동 선택
epsilon_decay = 0.99  # epsilon-greedy 전략에서 사용
epsilon_end = 0.01 #epsilon-greedy 에서 마지막 값
```

# 3. 함수 정의
**3-1) Q-Learning 함수 정의**
```python
def q_learning(env, alpha, gamma, epsilon, epsilon_decay=0, num_episodes=1000):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    total_rewards = []

    for i in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # 무작위 행동
            else:
                action = np.argmax(q_table[state])  # 최적의 행동 선택

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            state = next_state

            # Epsilon 감소
            if epsilon_decay != 0:
                epsilon = max(epsilon_end, epsilon * epsilon_decay)

        total_rewards.append(total_reward)

    return total_rewards
```

**3-2) 시각화 함수**
```python
def plot_rewards(rewards, title):
    plt.plot(rewards, label=title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards Over Episodes')
    plt.legend()
```

**3-3) 통계값 리턴 함수**

```python
def calculate_statistics(rewards):
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)
    std_reward = np.std(rewards)

    return {'Mean': mean_reward, 'Median': median_reward, 'Max': max_reward, 'Min': min_reward, 'Std': std_reward}
```


# 4. 실행
```python
rewards_case1 = q_learning(env, alpha, gamma, epsilon, num_episodes=1000) #Q learning with no epsilon
rewards_case2 = q_learning(env, alpha, gamma, 0.1, num_episodes=1000) # epsilon-greedy
rewards_case3 = q_learning(env, alpha, gamma, 1.0, epsilon_decay, num_episodes=1000) # epsilon-greedy with decay
```

# 5 시각화
```python!

plt.figure(figsize=(12, 6))
plot_rewards(rewards_case1, "Case 1: Epsilon = 0")
plot_rewards(rewards_case2, "Case 2: Epsilon = 0.1")
plot_rewards(rewards_case3, "Case 3: Epsilon-Greedy")
plt.show()
```

# 6. 각 케이스별 통계 계산 및 출력
```python
stats_case1 = calculate_statistics(rewards_case1)
stats_case2 = calculate_statistics(rewards_case2)
stats_case3 = calculate_statistics(rewards_case3)

df_stats = pd.DataFrame([stats_case1, stats_case2, stats_case3], index=["Case 1", "Case 2", "Case 3"])
print(df_stats)
```

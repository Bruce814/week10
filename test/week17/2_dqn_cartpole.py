"""
Week17: 强化学习进阶 — DQN (Deep Q-Network) 解决 CartPole 问题.

运行: python tests/week17/2_dqn_cartpole.py
依赖: pip install gymnasium torch

核心概念:
  - DQN 用神经网络近似 Q(s, a) 函数，突破了表格方法对状态空间的限制
  - 两大关键技巧:
    1. 经验回放（Experience Replay）: 将交互数据存入缓冲区，随机采样打破时序相关性
    2. 目标网络（Target Network）: 用一个延迟更新的网络计算 TD 目标，稳定训练

与 REINFORCE 的区别:
  - DQN 是 off-policy 的（可以用旧数据学习），REINFORCE 是 on-policy（必须用当前策略数据）
  - DQN 学 Q 值，REINFORCE 直接学策略
  - DQN 样本效率更高（经验回放），REINFORCE 每条轨迹用完即弃
"""

import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import gymnasium as gym
except ImportError:
    print("请先安装 gymnasium: pip install gymnasium")
    raise

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

# ============================================================
# 1. Q 网络
# ============================================================

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 2. 经验回放缓冲区
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 3. DQN Agent
# ============================================================

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        target_update: int = 10,
        buffer_capacity: int = 10000,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            q_vals = self.q_net(torch.FloatTensor(state).unsqueeze(0))
            return q_vals.argmax(dim=1).item()

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(batch.state)
        actions = torch.LongTensor(batch.action).unsqueeze(1)
        rewards = torch.FloatTensor(batch.reward)
        next_states = torch.FloatTensor(batch.next_state)
        dones = torch.FloatTensor(batch.done)

        # 当前 Q 值: Q(s, a)
        q_values = self.q_net(states).gather(1, actions).squeeze()

        # 目标 Q 值: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# ============================================================
# 4. 训练循环
# ============================================================

def train(episodes: int = 300):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    reward_history = []
    solved = False

    print(f"\n  CartPole-v1: state_dim={state_dim}, action_dim={action_dim}")
    print(f"  目标: 连续 100 回合平均奖励 >= 475 即为\"解决\"\n")

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.train_step()

            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        if ep % agent.target_update == 0:
            agent.update_target()

        reward_history.append(total_reward)

        if ep % 50 == 0:
            avg = sum(reward_history[-100:]) / min(len(reward_history), 100)
            print(
                f"  Episode {ep:>4d} | Reward: {total_reward:>6.1f} "
                f"| Avg(100): {avg:>6.1f} | ε: {agent.epsilon:.4f}"
            )

        if len(reward_history) >= 100:
            avg_100 = sum(reward_history[-100:]) / 100
            if avg_100 >= 475 and not solved:
                print(f"\n  [SOLVED] 在第 {ep} 回合解决! 最近100回合平均: {avg_100:.1f}")
                solved = True

    env.close()
    plot_rewards(reward_history)
    return agent, reward_history


def plot_rewards(reward_history, window=50):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [跳过绘图] pip install matplotlib")
        return

    smoothed = []
    for i in range(len(reward_history)):
        start = max(0, i - window + 1)
        smoothed.append(sum(reward_history[start:i + 1]) / (i - start + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, alpha=0.3, color="steelblue", label="每回合奖励")
    plt.plot(smoothed, color="darkorange", linewidth=2, label=f"滑动平均 (window={window})")
    plt.axhline(y=475, color="red", linestyle="--", alpha=0.5, label="解决阈值 (475)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN on CartPole-v1 — 训练曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tests/week17/dqn_reward_curve.png", dpi=150)
    plt.show()
    print("  训练曲线已保存到 tests/week17/dqn_reward_curve.png")


# ============================================================
# 5. 主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    print("=" * 60)
    print("  Week17: DQN (Deep Q-Network) — CartPole")
    print("=" * 60)
    train(episodes=300)
    print("\n" + "=" * 60)
    print("  完成!")
    print("=" * 60)

"""
Week17: 强化学习入门 — Q-Learning 解决 GridWorld 问题.

运行: python tests/week17/0_q_learning_grid_world.py

核心概念:
  - Agent（智能体）: 在环境中做决策的主体
  - Environment（环境）: 智能体交互的外部世界
  - State（状态）: 环境在某一时刻的描述
  - Action（动作）: 智能体可以执行的操作
  - Reward（奖励）: 环境对动作的即时反馈
  - Policy（策略）: 状态到动作的映射，即"在什么状态下做什么"
  - Q-Value Q(s, a): 在状态 s 执行动作 a 后，未来能获得的期望累计回报

Q-Learning 更新公式:
  Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

本示例用一个 5×5 的网格世界演示 Q-Learning:
  - S: 起点 (0,0)
  - G: 目标 (4,4)，到达得 +100
  - X: 陷阱，踩到得 -100
  - 每走一步得 -1（鼓励找最短路径）
"""

import random
from collections import defaultdict

# ============================================================
# 1. 定义 GridWorld 环境
# ============================================================

GRID_ROWS = 5
GRID_COLS = 5

TRAPS = {(1, 1), (2, 3), (3, 0), (3, 2)}
GOAL = (4, 4)
START = (0, 0)

ACTIONS = ["up", "down", "left", "right"]
ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

REWARD_STEP = -1
REWARD_TRAP = -100
REWARD_GOAL = 100


class GridWorld:
    """5×5 网格世界环境，遵循 OpenAI Gym 风格接口."""

    def __init__(self):
        self.state = START

    def reset(self):
        self.state = START
        return self.state

    def step(self, action: str):
        dr, dc = ACTION_DELTAS[action]
        r, c = self.state
        nr, nc = r + dr, c + dc

        if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
            self.state = (nr, nc)

        if self.state == GOAL:
            return self.state, REWARD_GOAL, True
        if self.state in TRAPS:
            return self.state, REWARD_TRAP, True

        return self.state, REWARD_STEP, False

    def render(self, q_table=None):
        """打印网格，显示智能体当前位置或最优策略."""
        arrow = {"up": "^", "down": "v", "left": "<", "right": ">"}
        lines = []
        for r in range(GRID_ROWS):
            row_str = []
            for c in range(GRID_COLS):
                pos = (r, c)
                if pos == self.state:
                    row_str.append(" A ")
                elif pos == GOAL:
                    row_str.append(" G ")
                elif pos in TRAPS:
                    row_str.append(" X ")
                elif q_table and pos in q_table:
                    best = max(ACTIONS, key=lambda a: q_table[pos][a])
                    row_str.append(f" {arrow[best]} ")
                else:
                    row_str.append(" . ")
            lines.append("|".join(row_str))
        sep = "-" * len(lines[0])
        print("\n".join(f"  {sep}\n  {line}" for line in lines))
        print(f"  {sep}")


# ============================================================
# 2. Q-Learning 算法
# ============================================================

def q_learning(
    env: GridWorld,
    episodes: int = 1000,
    alpha: float = 0.1,        # 学习率
    gamma: float = 0.99,       # 折扣因子
    epsilon: float = 1.0,      # 初始探索率
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    max_steps: int = 200,
):
    """
    Q-Learning 训练循环.

    参数:
      alpha        — 学习率，控制 Q 值更新幅度
      gamma        — 折扣因子，衡量未来奖励的重要性
      epsilon      — ε-greedy 探索率，开始时大量探索，逐步转向利用
      epsilon_min  — 探索率下限
      epsilon_decay— 每回合探索率的衰减系数
    """
    q_table: dict[tuple, dict[str, float]] = defaultdict(
        lambda: {a: 0.0 for a in ACTIONS}
    )

    reward_history = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            # ε-greedy 策略选择动作
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                action = max(ACTIONS, key=lambda a: q_table[state][a])

            next_state, reward, done = env.step(action)
            total_reward += reward

            # Q 值更新（Bellman 方程的时序差分形式）
            best_next = max(q_table[next_state].values())
            td_target = reward + gamma * best_next * (not done)
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            state = next_state
            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        reward_history.append(total_reward)

        if ep % 200 == 0:
            avg = sum(reward_history[-50:]) / 50
            print(f"  Episode {ep:>5d} | Avg Reward (last 50): {avg:>7.1f} | ε: {epsilon:.4f}")

    return q_table, reward_history


# ============================================================
# 3. 用学到的策略走一遍并展示
# ============================================================

def demonstrate_policy(env: GridWorld, q_table, max_steps: int = 50):
    """用训练好的 Q-table 贪心走一遍，打印轨迹."""
    state = env.reset()
    trajectory = [state]

    for _ in range(max_steps):
        action = max(ACTIONS, key=lambda a: q_table[state][a])
        state, reward, done = env.step(action)
        trajectory.append(state)
        if done:
            break

    print("\n  轨迹:", " -> ".join(str(s) for s in trajectory))
    reached = trajectory[-1] == GOAL
    print(f"  结果: {'[OK] 到达目标!' if reached else '[FAIL] 未到达目标'}")
    return reached


# ============================================================
# 4. （可选）绘制训练曲线
# ============================================================

def plot_rewards(reward_history, window: int = 50):
    """用 matplotlib 绘制每回合奖励的滑动平均曲线."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [跳过绘图] 请安装 matplotlib: pip install matplotlib")
        return

    smoothed = []
    for i in range(len(reward_history)):
        start = max(0, i - window + 1)
        smoothed.append(sum(reward_history[start:i + 1]) / (i - start + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, alpha=0.3, color="steelblue", label="每回合奖励")
    plt.plot(smoothed, color="darkorange", linewidth=2, label=f"滑动平均 (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning on GridWorld — 训练曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tests/week17/q_learning_reward_curve.png", dpi=150)
    plt.show()
    print("  训练曲线已保存到 tests/week17/q_learning_reward_curve.png")


# ============================================================
# 5. 主程序
# ============================================================

if __name__ == "__main__":
    random.seed(42)

    print("=" * 60)
    print("  Week17: Q-Learning 强化学习 — GridWorld")
    print("=" * 60)

    print("\n[网格世界说明]")
    print("  S = 起点(0,0)  G = 目标(4,4)  X = 陷阱  A = 智能体")
    print(f"  陷阱位置: {sorted(TRAPS)}")
    print(f"  奖励: 到达目标 +{REWARD_GOAL}, 掉入陷阱 {REWARD_TRAP}, 每步 {REWARD_STEP}")

    env = GridWorld()

    print("\n[初始网格]")
    env.render()

    print("\n[开始 Q-Learning 训练 (1000 回合)]\n")
    q_table, rewards = q_learning(env, episodes=1000)

    print("\n[训练完成] 学到的最优策略:")
    env.reset()
    env.render(q_table)

    print("\n[演示] 用学到的策略走一遍:")
    demonstrate_policy(env, q_table)

    print("\n[绘图] 绘制训练曲线...")
    plot_rewards(rewards)

    print("\n" + "=" * 60)
    print("  完成!")
    print("=" * 60)

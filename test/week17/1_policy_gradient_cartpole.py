"""
Week17: 强化学习进阶 — REINFORCE (Policy Gradient) 解决 CartPole 问题.

运行: python tests/week17/1_policy_gradient_cartpole.py
依赖: pip install gymnasium torch
可视化: pip install matplotlib imageio  （实时训练曲线 + 训练后 GIF 演示）

学习过程可视化:
  - 训练时每 25 个 episode 更新一次「奖励曲线 + 损失曲线」窗口
  - 训练结束后保存 policy_gradient_reward_curve.png（奖励+损失双图）
  - 保存 policy_gradient_demo.gif（训练后策略演示动画）

核心概念:
  - 策略梯度（Policy Gradient）: 直接参数化策略 π_θ(a|s)，通过梯度上升最大化期望回报
  - REINFORCE 算法: 最基础的策略梯度方法，用蒙特卡洛采样估计梯度
  - 损失函数: L = -Σ log π_θ(a_t|s_t) * G_t
    其中 G_t 是从时刻 t 开始的折扣累计回报

与 Q-Learning 的区别:
  - Q-Learning 学的是 Q(s, a) 值函数（间接推导策略）
  - Policy Gradient 直接学策略网络 π_θ(a|s)
  - Policy Gradient 天然支持连续动作空间、随机策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

try:
    import gymnasium as gym
except ImportError:
    print("请先安装 gymnasium: pip install gymnasium")
    print("如果需要渲染: pip install gymnasium[classic-control]")
    raise

# ============================================================
# 1. 策略网络
# ============================================================

class PolicyNetwork(nn.Module):
    """两层 MLP，输出动作的概率分布."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 2. REINFORCE 算法
# ============================================================

class REINFORCE:
    """REINFORCE 策略梯度算法."""

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def compute_returns(self):
        """计算每个时刻的折扣累计回报 G_t = Σ γ^k * r_{t+k}."""
        returns = []
        g = 0
        for r in reversed(self.rewards):
            g = r + self.gamma * g
            returns.insert(0, g)
        returns = torch.tensor(returns, dtype=torch.float32)
        # 标准化 returns 减少方差
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self):
        """用一条完整轨迹更新策略网络."""
        returns = self.compute_returns()
        loss = torch.stack(
            [-lp * g for lp, g in zip(self.log_probs, returns)]
        ).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()

        return loss.item()


# ============================================================
# 3. 训练循环
# ============================================================

def train(episodes: int = 500, render_final: bool = True, visualize_live: bool = True):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCE(state_dim, action_dim, lr=1e-3, gamma=0.99)
    reward_history = []
    loss_history = []
    solved = False

    # 实时绘图句柄（可选）
    live_plot = None
    if visualize_live:
        live_plot = init_live_plot()

    print(f"\n  CartPole-v1: state_dim={state_dim}, action_dim={action_dim}")
    print(f"  目标: 连续 100 回合平均奖励 >= 475 即为\"解决\"\n")

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break

        loss = agent.update()
        reward_history.append(total_reward)
        loss_history.append(loss)

        if live_plot and ep % 25 == 0:
            update_live_plot(live_plot, reward_history, loss_history, ep, episodes)

        if ep % 50 == 0:
            avg = sum(reward_history[-100:]) / min(len(reward_history), 100)
            print(f"  Episode {ep:>4d} | Reward: {total_reward:>6.1f} | Avg(100): {avg:>6.1f} | Loss: {loss:.4f}")

        if len(reward_history) >= 100:
            avg_100 = sum(reward_history[-100:]) / 100
            if avg_100 >= 475 and not solved:
                print(f"\n  [SOLVED] 在第 {ep} 回合解决! 最近100回合平均: {avg_100:.1f}")
                solved = True

    env.close()

    if live_plot:
        close_live_plot(live_plot)

    if render_final:
        print("\n  [DEMO] 用训练好的策略演示一局...")
        demo_env = gym.make("CartPole-v1", render_mode="human")
        state, _ = demo_env.reset()
        total = 0
        while True:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = demo_env.step(action)
            total += reward
            if terminated or truncated:
                break
        demo_env.close()
        agent.log_probs.clear()
        agent.rewards.clear()
        print(f"  演示得分: {total}")

    # 保存训练曲线与演示 GIF
    plot_rewards(reward_history, loss_history)
    save_demo_gif(agent, state_dim, action_dim)
    return agent, reward_history


# ============================================================
# 4. 学习过程可视化
# ============================================================

def _smooth(data, window):
    out = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        out.append(sum(data[start : i + 1]) / (i - start + 1))
    return out


def init_live_plot():
    """初始化实时训练曲线（双图：奖励 + 损失）."""
    try:
        import matplotlib.pyplot as plt
        plt.ion()
    except (ImportError, Exception):
        return None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("REINFORCE on CartPole-v1 - Training (live)", fontsize=12)
    ax1.set_ylabel("Total Reward")
    ax1.set_ylim(0, 550)
    ax1.axhline(y=475, color="red", linestyle="--", alpha=0.5, label="Solved")
    ax1.legend(loc="lower right")
    ax2.set_ylabel("Policy Loss")
    ax2.set_xlabel("Episode")
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    return {"fig": fig, "ax1": ax1, "ax2": ax2, "line_r": None, "line_r_smooth": None, "line_loss": None}


def update_live_plot(live_plot, reward_history, loss_history, ep, total_ep):
    """每 25 个 episode 更新一次曲线."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ax1, ax2 = live_plot["ax1"], live_plot["ax2"]
    window = 50

    if live_plot["line_r"] is not None:
        live_plot["line_r"].set_data(range(1, len(reward_history) + 1), reward_history)
        smooth = _smooth(reward_history, window)
        live_plot["line_r_smooth"].set_data(range(1, len(smooth) + 1), smooth)
    else:
        x = range(1, len(reward_history) + 1)
        live_plot["line_r"], = ax1.plot(x, reward_history, alpha=0.3, color="steelblue", label="Reward")
        smooth = _smooth(reward_history, window)
        live_plot["line_r_smooth"], = ax1.plot(x, smooth, color="darkorange", linewidth=2, label="Moving avg")
        ax1.legend(loc="lower right")

    if live_plot["line_loss"] is None:
        live_plot["line_loss"], = ax2.plot(loss_history, color="green", alpha=0.6, label="Loss")
        loss_smooth = _smooth(loss_history, window)
        live_plot["line_loss_smooth"], = ax2.plot(loss_smooth, color="darkgreen", linewidth=2, label="Loss (平滑)")
        ax2.legend(loc="upper right")
    else:
        live_plot["line_loss"].set_data(range(len(loss_history)), loss_history)
        loss_smooth = _smooth(loss_history, window)
        live_plot["line_loss_smooth"].set_data(range(len(loss_smooth)), loss_smooth)

    ax1.relim()
    ax1.autoscale_view(scalex=False)
    ax2.relim()
    ax2.autoscale_view(scalex=False)
    live_plot["fig"].canvas.draw()
    live_plot["fig"].canvas.flush_events()


def close_live_plot(live_plot):
    try:
        import matplotlib.pyplot as plt
        plt.ioff()
        if live_plot and live_plot.get("fig"):
            plt.close(live_plot["fig"])
    except Exception:
        pass


def plot_rewards(reward_history, loss_history=None, window=50):
    """训练结束后绘制奖励曲线（及可选损失曲线）并保存."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [跳过绘图] pip install matplotlib")
        return

    n_plots = 2 if loss_history else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    smoothed = _smooth(reward_history, window)
    axes[0].plot(reward_history, alpha=0.3, color="steelblue", label="Per episode")
    axes[0].plot(smoothed, color="darkorange", linewidth=2, label=f"Moving avg (w={window})")
    axes[0].axhline(y=475, color="red", linestyle="--", alpha=0.5, label="Solved (475)")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("REINFORCE on CartPole-v1 - Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if loss_history:
        loss_smooth = _smooth(loss_history, window)
        axes[1].plot(loss_history, alpha=0.3, color="green", label="Per episode")
        axes[1].plot(loss_smooth, color="darkgreen", linewidth=2, label="Moving avg")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Policy Loss")
        axes[1].set_title("Policy Gradient Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[0].set_xlabel("Episode")

    plt.tight_layout()
    path = "tests/week17/policy_gradient_reward_curve.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  训练曲线已保存到 {path}")


def save_demo_gif(agent, state_dim, action_dim, max_frames=300, fps=30):
    """用训练好的策略跑一局，保存为 GIF 动画."""
    try:
        try:
            import imageio.v3 as imageio
        except ImportError:
            import imageio
    except ImportError:
        print("  [跳过 GIF] pip install imageio")
        return

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    frames = []
    state, _ = env.reset()
    for _ in range(max_frames):
        with torch.no_grad():
            probs = agent.policy(torch.FloatTensor(state).unsqueeze(0))
            action = probs.argmax(dim=1).item()
        frame = env.render()
        if frame is not None:
            frames.append(frame)
            imageio.imsave(f'tests/week17/demo/{_}.png', frame)
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()

    if not frames:
        return
    out_path = "tests/week17/policy_gradient_demo.gif"
    # imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"  演示动画已保存到 {out_path}")


# ============================================================
# 4. 主程序
# ============================================================

if __name__ == "__main__":
    import os as _os
    episodes = int(_os.environ.get("EPISODES", "500"))
    render_final = _os.environ.get("RENDER_FINAL", "false").lower() in ("1", "true", "yes")
    visualize_live = _os.environ.get("VISUALIZE_LIVE", "true").lower() in ("1", "true", "yes")

    print("=" * 60)
    print("  Week17: REINFORCE 策略梯度 — CartPole")
    print("=" * 60)
    train(episodes=episodes, render_final=render_final, visualize_live=visualize_live)
    print("\n" + "=" * 60)
    print("  完成!")
    print("=" * 60)

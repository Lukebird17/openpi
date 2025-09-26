#!/usr/bin/env python3
"""
JAX版本的PPO训练脚本
基于原始PyTorch notebook，使用JAX/Flax实现
支持离散动作(CartPole)和连续动作(Pendulum)环境
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, NamedTuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


@dataclass
class Config:
    """训练配置"""
    # 环境设置
    env_name: str = "CartPole-v1"
    seed: int = 0
    
    # 网络架构
    hidden_dim: int = 128
    
    # 训练参数
    num_episodes: int = 500
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.98
    lmbda: float = 0.95
    epochs: int = 10
    eps: float = 0.2
    
    # 设备
    device: str = "cpu"  # JAX会自动处理GPU


class Transition(NamedTuple):
    """经验转换"""
    states: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_states: jnp.ndarray
    dones: jnp.ndarray


class PolicyNet(nn.Module):
    """策略网络 - 离散动作"""
    hidden_dim: int
    action_dim: int
    
    def setup(self):
        self.fc1 = nn.Dense(self.hidden_dim)
        self.fc2 = nn.Dense(self.action_dim)
    
    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        return nn.softmax(self.fc2(x))


class PolicyNetContinuous(nn.Module):
    """策略网络 - 连续动作"""
    hidden_dim: int
    action_dim: int
    
    def setup(self):
        self.fc1 = nn.Dense(self.hidden_dim)
        self.fc_mu = nn.Dense(self.action_dim)
        self.fc_std = nn.Dense(self.action_dim)
    
    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        mu = 2.0 * jnp.tanh(self.fc_mu(x))
        std = nn.softplus(self.fc_std(x))
        return mu, std


class ValueNet(nn.Module):
    """价值网络"""
    hidden_dim: int
    
    def setup(self):
        self.fc1 = nn.Dense(self.hidden_dim)
        self.fc2 = nn.Dense(1)
    
    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        return self.fc2(x)


class PPOTrainer:
    """PPO训练器 - 离散动作"""
    
    def __init__(self, config: Config):
        self.config = config
        self.key = jax.random.PRNGKey(config.seed)
        
        # 创建环境
        self.env = gym.make(config.env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # 初始化网络
        self.policy_net = PolicyNet(config.hidden_dim, self.action_dim)
        self.value_net = ValueNet(config.hidden_dim)
        
        # 初始化参数
        dummy_state = jnp.ones((1, self.state_dim))
        self.policy_params = self.policy_net.init(self.key, dummy_state)
        self.value_params = self.value_net.init(self.key, dummy_state)
        
        # 优化器
        self.policy_optimizer = optax.adam(config.actor_lr)
        self.value_optimizer = optax.adam(config.critic_lr)
        
        self.policy_opt_state = self.policy_optimizer.init(self.policy_params)
        self.value_opt_state = self.value_optimizer.init(self.value_params)
    
    def take_action(self, state: jnp.ndarray) -> Tuple[int, jnp.ndarray]:
        """选择动作"""
        self.key, subkey = jax.random.split(self.key)
        probs = self.policy_net.apply(self.policy_params, state)
        action = jax.random.categorical(subkey, probs)
        return action, probs
    
    def compute_gae(self, rewards: jnp.ndarray, values: jnp.ndarray, 
                    dones: jnp.ndarray) -> jnp.ndarray:
        """计算广义优势估计(GAE)"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.lmbda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return jnp.array(advantages)
    
    def update(self, transitions: Transition):
        """更新网络参数"""
        states = transitions.states
        actions = transitions.actions
        rewards = transitions.rewards
        next_states = transitions.next_states
        dones = transitions.dones
        
        # 计算价值
        values = self.value_net.apply(self.value_params, states).squeeze()
        next_values = self.value_net.apply(self.value_params, next_states).squeeze()
        
        # 计算TD目标和优势
        td_targets = rewards + self.config.gamma * next_values * (1 - dones)
        advantages = self.compute_gae(rewards, jnp.concatenate([values, next_values[-1:]]), dones)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算旧的对数概率
        old_probs = self.policy_net.apply(self.policy_params, states)
        old_log_probs = jnp.log(old_probs[jnp.arange(len(actions)), actions])
        
        def policy_loss(policy_params):
            probs = self.policy_net.apply(policy_params, states)
            log_probs = jnp.log(probs[jnp.arange(len(actions)), actions])
            ratio = jnp.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = jnp.clip(ratio, 1 - self.config.eps, 1 + self.config.eps) * advantages
            return -jnp.mean(jnp.minimum(surr1, surr2))
        
        def value_loss(value_params):
            values = self.value_net.apply(value_params, states).squeeze()
            return jnp.mean((values - td_targets) ** 2)
        
        # 更新策略网络
        for _ in range(self.config.epochs):
            policy_grads = jax.grad(policy_loss)(self.policy_params)
            updates, self.policy_opt_state = self.policy_optimizer.update(
                policy_grads, self.policy_opt_state)
            self.policy_params = optax.apply_updates(self.policy_params, updates)
        
        # 更新价值网络
        for _ in range(self.config.epochs):
            value_grads = jax.grad(value_loss)(self.value_params)
            updates, self.value_opt_state = self.value_optimizer.update(
                value_grads, self.value_opt_state)
            self.value_params = optax.apply_updates(self.value_params, updates)
    
    def train(self) -> list:
        """训练PPO智能体"""
        return_list = []
        episodes_per_iteration = max(1, self.config.num_episodes // 10)
        
        for i in range(10):
            with tqdm(total=episodes_per_iteration, desc=f'Iteration {i}') as pbar:
                for i_episode in range(episodes_per_iteration):
                    episode_return = 0
                    states, actions, rewards, next_states, dones = [], [], [], [], []
                    
                    # 重置环境
                    state, _ = self.env.reset(seed=self.config.seed)
                    state = jnp.array(state)
                    done = False
                    
                    while not done:
                        # 选择动作
                        action, _ = self.take_action(state)
                        action = int(action)
                        
                        # 执行动作
                        next_state, reward, terminated, truncated, _ = self.env.step(action)
                        next_state = jnp.array(next_state)
                        done = terminated or truncated
                        
                        # 存储经验
                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)
                        next_states.append(next_state)
                        dones.append(done)
                        
                        state = next_state
                        episode_return += reward
                    
                    # 创建转换对象
                    transitions = Transition(
                        states=jnp.array(states),
                        actions=jnp.array(actions),
                        rewards=jnp.array(rewards),
                        next_states=jnp.array(next_states),
                        dones=jnp.array(dones)
                    )
                    
                    # 更新网络
                    self.update(transitions)
                    
                    return_list.append(episode_return)
                    
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({
                            'episode': f'{episodes_per_iteration * i + i_episode + 1}',
                            'return': f'{np.mean(return_list[-10:]):.3f}'
                        })
                    pbar.update(1)
        
        self.env.close()
        return return_list


class PPOContinuousTrainer:
    """PPO训练器 - 连续动作"""
    
    def __init__(self, config: Config):
        self.config = config
        self.key = jax.random.PRNGKey(config.seed)
        
        # 创建环境
        self.env = gym.make(config.env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # 初始化网络
        self.policy_net = PolicyNetContinuous(config.hidden_dim, self.action_dim)
        self.value_net = ValueNet(config.hidden_dim)
        
        # 初始化参数
        dummy_state = jnp.ones((1, self.state_dim))
        self.policy_params = self.policy_net.init(self.key, dummy_state)
        self.value_params = self.value_net.init(self.key, dummy_state)
        
        # 优化器
        self.policy_optimizer = optax.adam(config.actor_lr)
        self.value_optimizer = optax.adam(config.critic_lr)
        
        self.policy_opt_state = self.policy_optimizer.init(self.policy_params)
        self.value_opt_state = self.value_optimizer.init(self.value_params)
    
    def take_action(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """选择动作"""
        self.key, subkey = jax.random.split(self.key)
        mu, std = self.policy_net.apply(self.policy_params, state)
        action = mu + std * jax.random.normal(subkey, mu.shape)
        return action, (mu, std)
    
    def compute_gae(self, rewards: jnp.ndarray, values: jnp.ndarray, 
                    dones: jnp.ndarray) -> jnp.ndarray:
        """计算广义优势估计(GAE)"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.lmbda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return jnp.array(advantages)
    
    def update(self, transitions: Transition):
        """更新网络参数"""
        states = transitions.states
        actions = transitions.actions
        rewards = transitions.rewards
        next_states = transitions.next_states
        dones = transitions.dones
        
        # 计算价值
        values = self.value_net.apply(self.value_params, states).squeeze()
        next_values = self.value_net.apply(self.value_params, next_states).squeeze()
        
        # 计算TD目标和优势
        td_targets = rewards + self.config.gamma * next_values * (1 - dones)
        advantages = self.compute_gae(rewards, jnp.concatenate([values, next_values[-1:]]), dones)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算旧的对数概率
        old_mu, old_std = self.policy_net.apply(self.policy_params, states)
        old_log_probs = -0.5 * jnp.sum(((actions - old_mu) / old_std) ** 2 + jnp.log(old_std) + jnp.log(2 * jnp.pi), axis=1)
        
        def policy_loss(policy_params):
            mu, std = self.policy_net.apply(policy_params, states)
            log_probs = -0.5 * jnp.sum(((actions - mu) / std) ** 2 + jnp.log(std) + jnp.log(2 * jnp.pi), axis=1)
            ratio = jnp.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = jnp.clip(ratio, 1 - self.config.eps, 1 + self.config.eps) * advantages
            return -jnp.mean(jnp.minimum(surr1, surr2))
        
        def value_loss(value_params):
            values = self.value_net.apply(value_params, states).squeeze()
            return jnp.mean((values - td_targets) ** 2)
        
        # 更新策略网络
        for _ in range(self.config.epochs):
            policy_grads = jax.grad(policy_loss)(self.policy_params)
            updates, self.policy_opt_state = self.policy_optimizer.update(
                policy_grads, self.policy_opt_state)
            self.policy_params = optax.apply_updates(self.policy_params, updates)
        
        # 更新价值网络
        for _ in range(self.config.epochs):
            value_grads = jax.grad(value_loss)(self.value_params)
            updates, self.value_opt_state = self.value_optimizer.update(
                value_grads, self.value_opt_state)
            self.value_params = optax.apply_updates(self.value_params, updates)
    
    def train(self) -> list:
        """训练PPO智能体"""
        return_list = []
        episodes_per_iteration = max(1, self.config.num_episodes // 10)
        
        for i in range(10):
            with tqdm(total=episodes_per_iteration, desc=f'Iteration {i}') as pbar:
                for i_episode in range(episodes_per_iteration):
                    episode_return = 0
                    states, actions, rewards, next_states, dones = [], [], [], [], []
                    
                    # 重置环境
                    state, _ = self.env.reset(seed=self.config.seed)
                    state = jnp.array(state)
                    done = False
                    
                    while not done:
                        # 选择动作
                        action, _ = self.take_action(state)
                        action = np.array(action)
                        
                        # 执行动作
                        next_state, reward, terminated, truncated, _ = self.env.step(action)
                        next_state = jnp.array(next_state)
                        done = terminated or truncated
                        
                        # 存储经验
                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)
                        next_states.append(next_state)
                        dones.append(done)
                        
                        state = next_state
                        episode_return += reward
                    
                    # 创建转换对象
                    transitions = Transition(
                        states=jnp.array(states),
                        actions=jnp.array(actions),
                        rewards=jnp.array(rewards),
                        next_states=jnp.array(next_states),
                        dones=jnp.array(dones)
                    )
                    
                    # 更新网络
                    self.update(transitions)
                    
                    return_list.append(episode_return)
                    
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({
                            'episode': f'{episodes_per_iteration * i + i_episode + 1}',
                            'return': f'{np.mean(return_list[-10:]):.3f}'
                        })
                    pbar.update(1)
        
        self.env.close()
        return return_list


def plot_results(return_list: list, env_name: str):
    """绘制训练结果"""
    episodes_list = list(range(len(return_list)))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'PPO on {env_name}')
    
    plt.subplot(1, 2, 2)
    # 计算移动平均
    window_size = min(9, len(return_list))
    if window_size > 1:
        mv_return = []
        for i in range(len(return_list)):
            start = max(0, i - window_size + 1)
            mv_return.append(np.mean(return_list[start:i+1]))
        plt.plot(episodes_list, mv_return)
    else:
        plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns (Moving Average)')
    plt.title(f'PPO on {env_name} (Moving Average)')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='JAX PPO Training')
    parser.add_argument('--env', type=str, default='CartPole-v1', 
                       choices=['CartPole-v1', 'Pendulum-v1'],
                       help='Environment name')
    parser.add_argument('--episodes', type=int, default=500, 
                       help='Number of episodes')
    parser.add_argument('--seed', type=int, default=0, 
                       help='Random seed')
    parser.add_argument('--plot', action='store_true', 
                       help='Plot results')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config(
        env_name=args.env,
        num_episodes=args.episodes,
        seed=args.seed
    )
    
    print(f"Training PPO on {config.env_name}")
    print(f"Episodes: {config.num_episodes}")
    print(f"Seed: {config.seed}")
    print(f"JAX devices: {jax.devices()}")
    
    # 选择训练器
    if config.env_name == 'CartPole-v1':
        trainer = PPOTrainer(config)
    elif config.env_name == 'Pendulum-v1':
        trainer = PPOContinuousTrainer(config)
    else:
        raise ValueError(f"Unsupported environment: {config.env_name}")
    
    # 训练
    return_list = trainer.train()
    
    print(f"Training completed!")
    print(f"Final returns: {return_list[-10:]}")
    print(f"Average return: {np.mean(return_list):.3f}")
    
    if args.plot:
        plot_results(return_list, config.env_name)


if __name__ == "__main__":
    main()



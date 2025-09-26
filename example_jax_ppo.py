#!/usr/bin/env python3
"""
JAX PPO 使用示例
演示如何使用JAX版本的PPO训练脚本
"""

import jax
import numpy as np
from train_ppo_jax import Config, PPOTrainer, PPOContinuousTrainer, plot_results


def train_cartpole():
    """训练CartPole环境"""
    print("=" * 50)
    print("Training PPO on CartPole-v1 (Discrete Actions)")
    print("=" * 50)
    
    # 创建配置
    config = Config(
        env_name="CartPole-v1",
        num_episodes=500,
        hidden_dim=128,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.98,
        lmbda=0.95,
        epochs=10,
        eps=0.2,
        seed=0
    )
    
    # 创建训练器
    trainer = PPOTrainer(config)
    
    # 训练
    return_list = trainer.train()
    
    # 显示结果
    print(f"Training completed!")
    print(f"Final 10 returns: {return_list[-10:]}")
    print(f"Average return: {np.mean(return_list):.3f}")
    print(f"Max return: {np.max(return_list):.3f}")
    
    # 绘制结果
    plot_results(return_list, "CartPole-v1")
    
    return return_list


def train_pendulum():
    """训练Pendulum环境"""
    print("=" * 50)
    print("Training PPO on Pendulum-v1 (Continuous Actions)")
    print("=" * 50)
    
    # 创建配置
    config = Config(
        env_name="Pendulum-v1",
        num_episodes=2000,
        hidden_dim=128,
        actor_lr=1e-4,
        critic_lr=5e-3,
        gamma=0.9,
        lmbda=0.9,
        epochs=10,
        eps=0.2,
        seed=0
    )
    
    # 创建训练器
    trainer = PPOContinuousTrainer(config)
    
    # 训练
    return_list = trainer.train()
    
    # 显示结果
    print(f"Training completed!")
    print(f"Final 10 returns: {return_list[-10:]}")
    print(f"Average return: {np.mean(return_list):.3f}")
    print(f"Max return: {np.max(return_list):.3f}")
    
    # 绘制结果
    plot_results(return_list, "Pendulum-v1")
    
    return return_list


def compare_performance():
    """比较不同配置的性能"""
    print("=" * 50)
    print("Performance Comparison")
    print("=" * 50)
    
    # 测试不同学习率
    learning_rates = [1e-4, 3e-4, 1e-3]
    results = {}
    
    for lr in learning_rates:
        print(f"\nTesting with actor_lr={lr}")
        config = Config(
            env_name="CartPole-v1",
            num_episodes=100,  # 快速测试
            actor_lr=lr,
            seed=0
        )
        
        trainer = PPOTrainer(config)
        return_list = trainer.train()
        results[lr] = np.mean(return_list[-10:])
        print(f"Average return: {results[lr]:.3f}")
    
    print("\nResults summary:")
    for lr, avg_return in results.items():
        print(f"  lr={lr}: {avg_return:.3f}")


if __name__ == "__main__":
    # 显示JAX信息
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    # 选择要运行的实验
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "cartpole":
            train_cartpole()
        elif sys.argv[1] == "pendulum":
            train_pendulum()
        elif sys.argv[1] == "compare":
            compare_performance()
        else:
            print("Usage: python example_jax_ppo.py [cartpole|pendulum|compare]")
    else:
        # 默认运行CartPole
        train_cartpole()






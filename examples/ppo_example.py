#!/usr/bin/env python3
"""Example usage of PPO training with OpenPI."""

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models.pi0_fast import Pi0FAST, Pi0FASTConfig
from openpi.rl.actor_critic import ActorCritic
from openpi.rl.reward import HumanRewardCLI, ExternalReward
from openpi.rl.ppo import PPOConfig


def example_external_reward(observation: dict, actions: np.ndarray) -> float:
    """Example external reward function.
    
    This is a simple example that could be replaced with:
    - Environment simulation
    - Human preference model
    - Task-specific reward function
    """
    # Simple reward based on action magnitude (example)
    action_norm = np.linalg.norm(actions)
    reward = -0.1 * action_norm  # Penalize large actions
    
    # Add some task-specific logic here
    # For example, check if the robot reached a target position
    # or completed a manipulation task
    
    return float(reward)


def main():
    """Example of setting up PPO training."""
    
    # Create model configuration
    config = Pi0FASTConfig(
        action_dim=32,
        action_horizon=32,
        max_token_len=250,
    )
    
    # Initialize model
    rng = jax.random.key(42)
    model = config.create(rng)
    
    # Create actor-critic
    fake_obs = config.fake_obs(1)
    rng, feat_rng = jax.random.split(rng)
    feats = model.value_features(feat_rng, fake_obs)
    
    ac_model = ActorCritic(
        model,
        value_in_dim=feats.shape[-1],
        hidden_dim=1024,
        rngs=nnx.Rngs(rng)
    )
    
    # Create reward providers
    human_reward = HumanRewardCLI()
    external_reward = ExternalReward(example_external_reward)
    
    # Example of getting rewards
    print("Testing reward providers...")
    
    # Test external reward
    test_obs = {"state": np.random.randn(32), "image": {"base_0_rgb": np.random.randn(224, 224, 3)}}
    test_actions = np.random.randn(32, 32)
    
    ext_reward = external_reward.get(test_obs, test_actions)
    print(f"External reward: {ext_reward}")
    
    # Test human reward (uncomment to test)
    # human_reward_val = human_reward.get(test_obs, test_actions)
    # print(f"Human reward: {human_reward_val}")
    
    # PPO configuration
    ppo_config = PPOConfig(
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        gamma=0.99,
        lam=0.95,
        lr=5e-5,
    )
    
    print(f"PPO config: {ppo_config}")
    print("PPO training setup complete!")


if __name__ == "__main__":
    main()









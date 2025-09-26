# PPO Training for OpenPI

This document describes how to use the PPO (Proximal Policy Optimization) training framework with OpenPI models.

## Overview

The PPO framework extends OpenPI models to support reinforcement learning by:

1. **Actor Model**: Uses the existing OpenPI model (e.g., Pi0FAST) as the policy/actor
2. **Value Function**: Adds an MLP value head on top of the model's hidden representations
3. **Reward Interface**: Supports external rewards and human feedback
4. **PPO Algorithm**: Implements GAE (Generalized Advantage Estimation) and PPO loss

## Architecture

```
Observation → OpenPI Model → Token Logits → Actions (Actor)
                    ↓
            Hidden Features → MLP → Value (Critic)
```

## Key Components

### 1. RL API (`openpi/models/rl_api.py`)
Defines the interface that models must implement for RL training:
- `sample_actions()`: Sample actions from policy
- `policy_logprobs()`: Compute log probabilities of actions
- `value_features()`: Extract features for value function

### 2. Actor-Critic (`openpi/rl/actor_critic.py`)
Combines the policy model with a value head:
- `ActorCritic`: Main class combining actor and critic
- `ValueHead`: MLP for value estimation

### 3. PPO Algorithm (`openpi/rl/ppo.py`)
Implements PPO training:
- `gae_advantages()`: Compute GAE advantages
- `ppo_loss()`: PPO loss function
- `ppo_step()`: Single PPO update step

### 4. Reward Interface (`openpi/rl/reward.py`)
Supports multiple reward sources:
- `HumanRewardCLI`: Command-line human feedback
- `HumanRewardWebSocket`: WebSocket-based human feedback
- `HumanRewardHTTP`: HTTP-based human feedback
- `ExternalReward`: External reward function
- `CompositeReward`: Combine multiple reward sources

### 5. Training Script (`scripts/train_ppo.py`)
Main training script for PPO:
- Rollout collection
- PPO updates
- Checkpointing
- Logging

## Usage

### Basic Training

```bash
# Train with human CLI feedback
python scripts/train_ppo.py \
    --model.pi0_fast \
    --reward_type human_cli \
    --num_train_steps 1000 \
    --rollout_length 128

# Train with external reward function
python scripts/train_ppo.py \
    --model.pi0_fast \
    --reward_type external \
    --external_reward_fn my_reward_function \
    --num_train_steps 1000
```

### Custom Reward Function

```python
def my_reward_function(observation: dict, actions: np.ndarray) -> float:
    """Custom reward function."""
    # Implement your reward logic here
    # observation: dict with keys like 'state', 'image', etc.
    # actions: np.ndarray of shape [action_horizon, action_dim]
    
    # Example: penalize large actions
    action_norm = np.linalg.norm(actions)
    reward = -0.1 * action_norm
    
    return float(reward)

# Use in training
from openpi.rl.reward import ExternalReward
reward_provider = ExternalReward(my_reward_function)
```

### Human Feedback via WebSocket

```python
# Server side (for human feedback collection)
import asyncio
import websockets
import json

async def reward_server(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        observation = data["observation"]
        actions = data["actions"]
        
        # Present to human for feedback
        print(f"Observation: {observation}")
        print(f"Actions: {actions}")
        reward = float(input("Enter reward [-1, 1]: "))
        
        await websocket.send(json.dumps({"reward": reward}))

# Start server
start_server = websockets.serve(reward_server, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
```

### Programmatic Usage

```python
import jax
from openpi.models.pi0_fast import Pi0FAST, Pi0FASTConfig
from openpi.rl.actor_critic import ActorCritic
from openpi.rl.ppo import PPOConfig, ppo_step

# Create model
config = Pi0FASTConfig()
model = config.create(jax.random.key(42))

# Create actor-critic
ac_model = ActorCritic(model, value_in_dim=1024)

# PPO training step
rng = jax.random.key(42)
obs = config.fake_obs(1)
actions = model.sample_actions(rng, obs)
rewards = jnp.array([0.5])  # Example reward
dones = jnp.array([False])
old_logp = model.policy_logprobs(rng, obs, actions)
old_values = ac_model.value(rng, obs)

loss, metrics = ppo_step(
    ac_model, rng, obs, actions, rewards, dones, old_logp, old_values, PPOConfig()
)
```

## Configuration

### PPO Hyperparameters

```python
ppo_config = PPOConfig(
    clip_ratio=0.2,      # PPO clipping ratio
    value_coef=0.5,      # Value loss coefficient
    entropy_coef=0.01,   # Entropy bonus coefficient
    gamma=0.99,          # Discount factor
    lam=0.95,            # GAE lambda
    lr=5e-5,             # Learning rate
    train_iters=4,        # PPO iterations per update
    minibatch_size=64,   # Minibatch size
)
```

### Training Configuration

```python
train_config = PPOTrainConfig(
    # Model config
    model=Pi0FASTConfig(),
    
    # PPO config
    ppo=ppo_config,
    
    # Reward config
    reward_type="human_cli",
    reward_config=RewardConfig(min_reward=-1.0, max_reward=1.0),
    
    # Training params
    rollout_length=128,
    num_ppo_epochs=4,
    num_train_steps=1000,
)
```

## Extending the Framework

### Adding New Reward Sources

```python
class MyCustomReward(RewardProvider):
    def get(self, observation: dict, actions: np.ndarray) -> float:
        # Implement your reward logic
        return 0.0
```

### Custom Value Head

```python
class MyValueHead(nnx.Module):
    def __init__(self, in_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.layers = [
            nnx.Linear(in_dim, 512, rngs=rngs),
            nnx.Linear(512, 256, rngs=rngs),
            nnx.Linear(256, 1, rngs=rngs),
        ]
    
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)[..., 0]
```

## Tips for Training

1. **Start with small rollouts**: Begin with short episodes to debug
2. **Monitor reward distribution**: Ensure rewards are well-distributed
3. **Use human feedback sparingly**: Human feedback is expensive, use it for key decisions
4. **Combine reward sources**: Use external rewards for frequent feedback, human for critical decisions
5. **Tune hyperparameters**: PPO is sensitive to hyperparameters, especially learning rate and clipping ratio

## Troubleshooting

### Common Issues

1. **NaN losses**: Check reward values and learning rate
2. **No learning**: Verify reward signal is informative
3. **Instability**: Reduce learning rate or increase clipping ratio
4. **Memory issues**: Reduce batch size or rollout length

### Debugging

```python
# Check reward distribution
rewards = [reward_provider.get(obs, acts) for obs, acts in rollout_data]
print(f"Reward stats: mean={np.mean(rewards)}, std={np.std(rewards)}")

# Check policy ratio
old_logp = model.policy_logprobs(rng, obs, actions)
new_logp = model.policy_logprobs(rng, obs, actions)
ratio = jnp.exp(new_logp - old_logp)
print(f"Policy ratio: mean={jnp.mean(ratio)}, std={jnp.std(ratio)}")
```









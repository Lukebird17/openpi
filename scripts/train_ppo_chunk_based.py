#!/usr/bin/env python3
"""
Action Chunk Based PPO Training Script for Pi0FAST Actor-Critic with Aloha Real Environment.

This script implements chunk-level PPO where:
- Each action chunk gets a single reward
- GAE is computed at the chunk level
- Actor loss uses classic PPO formulation with token-level log probabilities

Usage:
    python scripts/train_ppo_chunk_based.py --config=debug --exp_name=chunk_ppo_aloha
"""

import logging
import pathlib
import sys
import time
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import flax.nnx as nnx
import orbax.checkpoint as ocp

# Add src to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from openpi.models.pi0_fast_actor_critic import Pi0FASTActorCritic, Pi0FASTActorCriticConfig
from openpi.models import model as _model
import flax.traverse_util as traverse_util

# Add examples to path for aloha_real
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "examples"))
from aloha_real import env as aloha_env

logger = logging.getLogger("openpi")


def init_logging():
    """Initialize logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
        datefmt="%H:%M:%S",
    )


@dataclass
class PPOConfig:
    """PPO Training Configuration."""
    
    # Model configuration
    model_config: Pi0FASTActorCriticConfig
    
    # Pretrained model loading
    pretrained_checkpoint_path: str = None  # Path to pretrained Pi0FAST model
    
    # PPO hyperparameters
    total_timesteps: int = 100_000
    num_chunks_per_update: int = 64      # Number of action chunks per update
    num_epochs_per_update: int = 4       # PPO epochs per update
    batch_size: int = 16                 # Batch size for PPO updates
    learning_rate: float = 3e-4
    clip_coef: float = 0.2              # PPO clipping coefficient
    ent_coef: float = 0.01              # Entropy coefficient
    vf_coef: float = 0.5                # Value function coefficient
    max_grad_norm: float = 0.5          # Gradient clipping
    gamma: float = 0.99                 # Discount factor
    gae_lambda: float = 0.95            # GAE lambda
    
    # Environment settings
    max_episode_steps: int = 200
    
    # Logging and evaluation
    log_frequency: int = 5
    save_frequency: int = 20
    eval_frequency: int = 10
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints/ppo_chunk_aloha"
    
    # Random seed
    seed: int = 42


@dataclass
class TrainingConfig:
    """Overall training configuration."""
    
    exp_name: str = "ppo_chunk_aloha_experiment"
    project_name: str = "openpi_ppo_chunk"
    
    ppo_config: PPOConfig = None
    
    log_to_wandb: bool = True
    use_gpu: bool = True
    
    # Aloha environment settings
    render_height: int = 224
    render_width: int = 224
    reset_position: list = None


class ChunkBuffer:
    """Buffer for storing action chunk level data."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.reset()
    
    def reset(self):
        """Reset buffer."""
        self.observations = []
        self.action_chunks = []
        self.chunk_rewards = []
        self.chunk_values = []
        self.chunk_log_probs = []
        self.chunk_advantages = []
        self.chunk_returns = []
        self.size = 0
    
    def add(self, obs, actions, reward, value, log_prob):
        """Add a chunk-level transition."""
        if self.size < self.capacity:
            self.observations.append(obs)
            self.action_chunks.append(actions)
            self.chunk_rewards.append(reward)
            self.chunk_values.append(value)
            self.chunk_log_probs.append(log_prob)
            self.size += 1
        else:
            raise ValueError("Buffer is full!")
    
    def compute_gae(self, gamma: float, lam: float):
        """Compute GAE at chunk level."""
        rewards = np.array(self.chunk_rewards)
        values = np.array(self.chunk_values)
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = 0  # Terminal value
            else:
                next_value = values[t + 1]
            
            # Standard GAE computation
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.chunk_advantages = advantages.tolist()
        self.chunk_returns = returns.tolist()
    
    def get_batch(self, batch_size: int):
        """Get a random batch."""
        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False)
        
        batch = {
            'observations': [self.observations[i] for i in indices],
            'action_chunks': [self.action_chunks[i] for i in indices],
            'old_log_probs': [self.chunk_log_probs[i] for i in indices],
            'advantages': [self.chunk_advantages[i] for i in indices],
            'returns': [self.chunk_returns[i] for i in indices],
            'old_values': [self.chunk_values[i] for i in indices],
        }
        
        return batch


class AlohaRealEnvironmentWrapper:
    """Wrapper for Aloha real environment."""
    
    def __init__(
        self, 
        reset_position=None, 
        render_height: int = 224, 
        render_width: int = 224,
        max_episode_steps: int = 200
    ):
        self.env = aloha_env.AlohaRealEnvironment(
            reset_position=reset_position,
            render_height=render_height,
            render_width=render_width
        )
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
    def reset(self):
        """Reset environment and return initial observation."""
        self.env.reset()
        self.current_step = 0
        obs_dict = self.env.get_observation()
        
        # Convert to model observation format
        observation = _model.Observation(
            state=jnp.array(obs_dict["state"], dtype=jnp.float32),
            images={k: jnp.array(v, dtype=jnp.float32) for k, v in obs_dict["images"].items()}
        )
        
        return observation
    
    def step_chunk(self, action_chunk: _model.Actions):
        """Execute an entire action chunk and return cumulative reward."""
        total_reward = 0.0
        num_steps = 0
        
        for step_idx in range(action_chunk.actions.shape[0]):
            if self.current_step >= self.max_episode_steps:
                break
                
            # Execute single action
            single_action_dict = {"actions": np.array(action_chunk.actions[step_idx])}
            self.env.apply_action(single_action_dict)
            
            # Get step reward
            step_reward = float(self.env.get_reward())
            total_reward += step_reward
            
            self.current_step += 1
            num_steps += 1
            
            # Check if episode is done
            if self.env.is_episode_complete():
                break
        
        # Get final observation
        obs_dict = self.env.get_observation()
        next_observation = _model.Observation(
            state=jnp.array(obs_dict["state"], dtype=jnp.float32),
            images={k: jnp.array(v, dtype=jnp.float32) for k, v in obs_dict["images"].items()}
        )
        
        terminated = self.env.is_episode_complete()
        truncated = self.current_step >= self.max_episode_steps
        
        info = {
            "chunk_steps": num_steps,
            "cumulative_reward": total_reward,
            "episode_step": self.current_step,
        }
        
        return next_observation, total_reward, terminated, truncated, info


class ChunkPPOTrainer:
    """PPO trainer for action chunk level training."""
    
    def __init__(self, config: PPOConfig, env: AlohaRealEnvironmentWrapper):
        self.config = config
        self.env = env
        
        # Initialize random key
        self.key = jax.random.key(config.seed)
        self.global_step = 0
        
        # Initialize model
        self.key, model_key = jax.random.split(self.key)
        rngs = nnx.Rngs(model_key)
        self.model = Pi0FASTActorCritic(config.model_config, rngs=rngs)
        
        # Load pretrained weights if specified
        if config.pretrained_checkpoint_path:
            logger.info(f"Loading pretrained Pi0FAST weights from: {config.pretrained_checkpoint_path}")
            try:
                self._load_pretrained_weights(config.pretrained_checkpoint_path)
                logger.info("Successfully loaded pretrained Pi0FAST weights!")
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {e}")
                logger.warning("Continuing with random initialization")
        
        # Setup optimizer
        learning_rate = optax.linear_schedule(
            init_value=config.learning_rate,
            end_value=config.learning_rate * 0.1,
            transition_steps=config.total_timesteps // config.num_chunks_per_update
        )
        
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate, eps=1e-5)
        )
        
        # Initialize optimizer state
        self.opt_state = self.optimizer.init(nnx.state(self.model))
        
        # Initialize chunk buffer
        self.buffer = ChunkBuffer(capacity=config.num_chunks_per_update)
        
        # Metrics tracking
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "chunk_rewards": [],
            "training_losses": [],
        }
    
    def _load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained weights into the Actor-Critic model.
        
        This method supports two formats:
        1. Pi0FAST IL parameters (maps to actor_backbone)
        2. RL Actor-Critic parameters (direct loading)
        3. Auto-detects format and handles accordingly
        """
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load parameters directly from local path
        pretrained_params = _model.restore_params(
            checkpoint_path,
            restore_type=jax.Array,
            dtype=jnp.bfloat16
        )
        
        logger.info(f"Loaded parameters with keys: {list(pretrained_params.keys())}")
        
        # Get current model state
        current_state = nnx.state(self.model)
        current_flat = traverse_util.flatten_dict(current_state, sep="/")
        pretrained_flat = traverse_util.flatten_dict(pretrained_params, sep="/")
        
        logger.info(f"Current model parameters: {len(current_flat)}")
        logger.info(f"Loaded parameters: {len(pretrained_flat)}")
        
        # Auto-detect parameter format
        has_actor_backbone = any(key.startswith("actor_backbone/") for key in pretrained_flat.keys())
        has_value_head = any(key.startswith("value_head/") for key in pretrained_flat.keys())
        
        if has_actor_backbone and has_value_head:
            # Format: RL Actor-Critic parameters - direct loading
            logger.info("🔄 Detected RL Actor-Critic checkpoint format")
            updated_params = self._load_rl_checkpoint(current_flat, pretrained_flat)
            
        elif has_actor_backbone and not has_value_head:
            # Format: Partial RL checkpoint (actor only)
            logger.info("🔄 Detected partial RL checkpoint format (actor only)")
            updated_params = self._load_partial_rl_checkpoint(current_flat, pretrained_flat)
            
        else:
            # Format: Pi0FAST IL parameters - map to actor_backbone
            logger.info("🔄 Detected Pi0FAST IL checkpoint format")
            updated_params = self._load_il_checkpoint(current_flat, pretrained_flat)
        
        # Update model with loaded parameters
        updated_state = traverse_util.unflatten_dict(updated_params, sep="/")
        nnx.update(self.model, updated_state)
        
        logger.info("✅ Checkpoint loaded successfully!")
    
    def _load_rl_checkpoint(self, current_flat, pretrained_flat):
        """Load complete RL Actor-Critic checkpoint."""
        updated_params = {}
        actor_count = 0
        value_count = 0
        
        for current_key, current_value in current_flat.items():
            if current_key in pretrained_flat:
                pretrained_value = pretrained_flat[current_key]
                if current_value.shape == pretrained_value.shape:
                    updated_params[current_key] = pretrained_value
                    if current_key.startswith("actor_backbone/"):
                        actor_count += 1
                    elif current_key.startswith("value_head/"):
                        value_count += 1
                else:
                    logger.warning(f"Shape mismatch for {current_key}: "
                                 f"current={current_value.shape} vs pretrained={pretrained_value.shape}")
                    updated_params[current_key] = current_value
            else:
                updated_params[current_key] = current_value
        
        logger.info(f"Loaded {actor_count} actor parameters, {value_count} value parameters")
        return updated_params
    
    def _load_partial_rl_checkpoint(self, current_flat, pretrained_flat):
        """Load partial RL checkpoint (actor only)."""
        updated_params = {}
        actor_count = 0
        value_count = 0
        
        for current_key, current_value in current_flat.items():
            if current_key.startswith("actor_backbone/") and current_key in pretrained_flat:
                pretrained_value = pretrained_flat[current_key]
                if current_value.shape == pretrained_value.shape:
                    updated_params[current_key] = pretrained_value
                    actor_count += 1
                else:
                    logger.warning(f"Shape mismatch for {current_key}")
                    updated_params[current_key] = current_value
            else:
                # Keep current parameters (including value_head)
                updated_params[current_key] = current_value
                if current_key.startswith("value_head/"):
                    value_count += 1
        
        logger.info(f"Loaded {actor_count} actor parameters, kept {value_count} random value parameters")
        return updated_params
    
    def _load_il_checkpoint(self, current_flat, pretrained_flat):
        """Load Pi0FAST IL checkpoint (map to actor_backbone)."""
        updated_params = {}
        mapped_count = 0
        value_count = 0
        
        for current_key, current_value in current_flat.items():
            if current_key.startswith("actor_backbone/"):
                # Remove "actor_backbone/" prefix to match IL pretrained keys
                il_key = "/".join(current_key.split("/")[1:])
                
                if il_key in pretrained_flat:
                    pretrained_value = pretrained_flat[il_key]
                    if current_value.shape == pretrained_value.shape:
                        updated_params[current_key] = pretrained_value
                        mapped_count += 1
                    else:
                        logger.warning(f"Shape mismatch for {current_key}")
                        updated_params[current_key] = current_value
                else:
                    logger.warning(f"No IL parameter found for {current_key}")
                    updated_params[current_key] = current_value
            
            elif current_key.startswith("value_head/"):
                # Keep randomly initialized value head
                updated_params[current_key] = current_value
                value_count += 1
            
            else:
                updated_params[current_key] = current_value
        
        logger.info(f"Mapped {mapped_count} IL parameters to actor, kept {value_count} random value parameters")
        return updated_params
    
    def _save_checkpoint(self, update_step: int):
        """Save checkpoint in unified format for future loading.
        
        Saves in Actor-Critic format (actor_backbone + value_head) which can be 
        loaded by both IL->RL and RL->RL scenarios.
        """
        # Create checkpoint directory
        checkpoint_dir = pathlib.Path(self.config.checkpoint_dir) / f"step_{update_step}"
        params_dir = checkpoint_dir / "params" 
        os.makedirs(params_dir, exist_ok=True)
        
        logger.info(f"Saving checkpoint to: {checkpoint_dir}")
        
        try:
            # Get current model state (already in Actor-Critic format)
            model_state = nnx.state(self.model)
            
            # Save using Orbax PyTreeCheckpointer (compatible with restore_params)
            with ocp.PyTreeCheckpointer() as ckptr:
                ckptr.save(
                    params_dir,
                    {"params": model_state},
                    ocp.args.StandardSave()
                )
            
            # Save training metadata
            metadata = {
                "update_step": update_step,
                "total_chunks": len(self.metrics["chunk_rewards"]),
                "total_episodes": len(self.metrics["episode_rewards"]),
                "config": {
                    "action_dim": self.config.model_config.action_dim,
                    "action_horizon": self.config.model_config.action_horizon,
                    "max_token_len": self.config.model_config.max_token_len,
                    "format": "rl_actor_critic",  # Format identifier
                }
            }
            
            with open(checkpoint_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Checkpoint saved successfully!")
            logger.info(f"   Format: RL Actor-Critic (actor_backbone + value_head)")
            logger.info(f"   Can be loaded by: IL->RL or RL->RL scenarios")
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting Chunk-based PPO training with Pi0FAST Actor-Critic on Aloha Real")
        
        num_updates = self.config.total_timesteps // self.config.num_chunks_per_update
        
        for update in range(1, num_updates + 1):
            # Collect chunk-level rollouts
            self._collect_chunk_rollouts()
            
            # Update model using PPO
            training_metrics = self._update_model()
            
            # Log progress
            if update % self.config.log_frequency == 0:
                self._log_progress(update, num_updates, training_metrics)
            
            # Save checkpoint
            if update % self.config.save_frequency == 0:
                self._save_checkpoint(update)
        
        logger.info("Training completed!")
        return self.metrics
    
    def _collect_chunk_rollouts(self):
        """Collect action chunk rollouts."""
        self.buffer.reset()
        
        obs = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        for chunk_idx in range(self.config.num_chunks_per_update):
            # Sample action chunk
            self.key, action_key, value_key = jax.random.split(self.key, 3)
            
            # Use train=True for proper gradient computation during rollout
            actions = self.model.sample_actions(
                rng=action_key,
                observation=obs,
                temperature=1.0,
                train=True  # Now we can use train=True for consistent training mode!
            )
            
            # Compute chunk value using our extended method
            # Use train=True for proper gradient computation during training
            chunk_value = self.model.compute_action_chunk_value(
                observation=obs,
                actions=actions,
                rng=value_key,
                train=True
            )
            
            # Compute chunk log probability  
            # Use train=True for proper gradient computation during training
            chunk_log_prob = self.model.compute_action_chunk_log_prob(
                observation=obs,
                actions=actions,
                rng=action_key,
                train=True
            )
            
            # Execute action chunk in environment
            next_obs, chunk_reward, terminated, truncated, info = self.env.step_chunk(actions)
            
            episode_reward += chunk_reward
            episode_length += info["chunk_steps"]
            
            # Store chunk-level transition
            self.buffer.add(
                obs=obs,
                actions=actions,
                reward=chunk_reward,
                value=float(chunk_value),
                log_prob=float(chunk_log_prob)
            )
            
            self.metrics["chunk_rewards"].append(chunk_reward)
            
            obs = next_obs
            
            # Handle episode end
            if terminated or truncated:
                logger.info(f"Episode finished: reward={episode_reward:.2f}, length={episode_length}")
                self.metrics["episode_rewards"].append(episode_reward)
                self.metrics["episode_lengths"].append(episode_length)
                
                obs = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
    
    def _update_model(self):
        """Update model using PPO on chunk-level data."""
        
        # Compute GAE at chunk level
        self.buffer.compute_gae(self.config.gamma, self.config.gae_lambda)
        
        # Update model for multiple epochs
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for epoch in range(self.config.num_epochs_per_update):
            
            epoch_loss = 0.0
            num_batches = 0
            
            # Create batches and update
            num_batches_per_epoch = max(1, self.buffer.size // self.config.batch_size)
            
            for _ in range(num_batches_per_epoch):
                batch = self.buffer.get_batch(self.config.batch_size)
                
                # Compute PPO loss (uses stored old_log_probs directly)
                loss_info = self._ppo_update_step(batch)
                
                epoch_loss += loss_info['total_loss']
                total_policy_loss += loss_info['policy_loss']
                total_value_loss += loss_info['value_loss']
                num_batches += 1
            
            total_loss += epoch_loss / num_batches_per_epoch
        
        avg_loss = total_loss / self.config.num_epochs_per_update
        avg_policy_loss = total_policy_loss / (self.config.num_epochs_per_update * num_batches_per_epoch)
        avg_value_loss = total_value_loss / (self.config.num_epochs_per_update * num_batches_per_epoch)
        
        self.metrics["training_losses"].append(float(avg_loss))
        
        return {
            "avg_loss": avg_loss,
            "avg_policy_loss": avg_policy_loss,
            "avg_value_loss": avg_value_loss,
        }
    
    def _ppo_update_step(self, batch):
        """Single PPO update step using classic actor loss formulation."""
        
        def loss_fn(model_state):
            # Reconstruct model from state
            model = nnx.merge(nnx.graphdef(self.model), model_state)
            
            batch_size = len(batch['observations'])
            total_policy_loss = 0.0
            total_value_loss = 0.0
            
            for i in range(batch_size):
                obs = batch['observations'][i]
                actions = batch['action_chunks'][i]
                old_log_prob = batch['old_log_probs'][i]
                advantage = batch['advantages'][i]
                return_val = batch['returns'][i]
                
                # Compute new log probability (classic actor component)
                # Use train=True for proper gradient computation during PPO updates
                self.key, step_key = jax.random.split(self.key)
                new_log_prob = model.compute_action_chunk_log_prob(
                    observation=obs,
                    actions=actions,
                    rng=step_key,
                    train=True  # Training mode for gradient computation
                )
                
                # Compute new value
                # Use train=True for proper gradient computation during PPO updates
                new_value = model.compute_action_chunk_value(
                    observation=obs,
                    actions=actions,
                    rng=step_key,
                    train=True  # Training mode for gradient computation
                )
                
                # Classic PPO Actor Loss
                log_ratio = new_log_prob - old_log_prob
                ratio = jnp.exp(log_ratio)
                
                # PPO clipped surrogate loss
                surr1 = ratio * advantage
                surr2 = jnp.clip(ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef) * advantage
                policy_loss = -jnp.minimum(surr1, surr2)
                
                # Value function loss
                value_loss = jnp.square(new_value - return_val)
                
                total_policy_loss += policy_loss
                total_value_loss += value_loss
            
            # Average losses
            avg_policy_loss = total_policy_loss / batch_size
            avg_value_loss = total_value_loss / batch_size
            
            # Combined loss
            total_loss = avg_policy_loss + self.config.vf_coef * avg_value_loss
            
            return total_loss, {
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'total_loss': total_loss,
            }
        
        # Compute gradients and update
        model_state = nnx.state(self.model)
        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_state)
        
        # Apply gradients
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        model_state = optax.apply_updates(model_state, updates)
        nnx.update(self.model, model_state)
        
        return loss_info
    
    def _log_progress(self, update_step, total_updates, training_metrics):
        """Log training progress."""
        recent_episode_rewards = self.metrics["episode_rewards"][-10:]
        recent_chunk_rewards = self.metrics["chunk_rewards"][-50:]
        
        avg_episode_reward = np.mean(recent_episode_rewards) if recent_episode_rewards else 0.0
        avg_chunk_reward = np.mean(recent_chunk_rewards) if recent_chunk_rewards else 0.0
        
        logger.info(
            f"Update {update_step}/{total_updates} | "
            f"Avg Episode Reward: {avg_episode_reward:.2f} | "
            f"Avg Chunk Reward: {avg_chunk_reward:.3f} | "
            f"Policy Loss: {training_metrics['avg_policy_loss']:.4f} | "
            f"Value Loss: {training_metrics['avg_value_loss']:.4f}"
        )
        
        # Log to wandb if enabled
        if wandb.run is not None:
            wandb.log({
                "update_step": update_step,
                "global_step": self.global_step,
                "avg_episode_reward": avg_episode_reward,
                "avg_chunk_reward": avg_chunk_reward,
                "policy_loss": training_metrics["avg_policy_loss"],
                "value_loss": training_metrics["avg_value_loss"],
                "total_loss": training_metrics["avg_loss"],
            })
    
    def _save_checkpoint(self, update_step):
        """Save model checkpoint."""
        checkpoint_path = pathlib.Path(self.config.checkpoint_dir) / f"checkpoint_{update_step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_state = nnx.state(self.model)
        with open(checkpoint_path / "model_state.pkl", "wb") as f:
            pickle.dump(model_state, f)
        
        # Save training metrics
        with open(checkpoint_path / "metrics.pkl", "wb") as f:
            pickle.dump(self.metrics, f)
        
        logger.info(f"Saved checkpoint at update {update_step}")


def create_ppo_configs():
    """Create predefined PPO configurations."""
    
    # Base Pi0FAST Actor-Critic configuration - match pretrained model!
    # NOTE: Using Pi0FAST base model configuration to ensure compatibility
    # with pretrained checkpoints (gs://openpi-assets/checkpoints/pi0_fast_base/params)
    base_model_config = Pi0FASTActorCriticConfig(
        action_dim=7,           # Pi0FAST base was trained with 7DOF (single arm)
        action_horizon=10,      # Pi0FAST base uses 10-step action chunks
        max_token_len=180,      # Token sequence length (matches pretrained)
        paligemma_variant="gemma_2b",  # Use 2B parameter model
        value_head_hidden_dim=512,     # Value head MLP size
        max_value_steps=32,     # Max steps for value computation
        # NOTE: For Aloha dual-arm (14DOF), you may need to:
        # 1. Fine-tune with action_dim=14, or
        # 2. Use single-arm control for initial PPO experiments
    )
    
    # Debug configuration for quick testing
    debug_config = PPOConfig(
        model_config=base_model_config,
        pretrained_checkpoint_path="./checkpoints/pi0_fast_base/params",  # Local checkpoint path
        total_timesteps=5_000,
        num_chunks_per_update=32,
        num_epochs_per_update=2,
        batch_size=8,
        learning_rate=1e-4,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        log_frequency=2,
        save_frequency=10,
        checkpoint_dir="./checkpoints/ppo_chunk_debug",
    )
    
    # Full training configuration
    full_config = PPOConfig(
        model_config=base_model_config,
        pretrained_checkpoint_path="./checkpoints/pi0_fast_base/params",  # Local checkpoint path
        total_timesteps=500_000,
        num_chunks_per_update=128,
        num_epochs_per_update=4,
        batch_size=32,
        learning_rate=3e-4,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        log_frequency=5,
        save_frequency=50,
        checkpoint_dir="./checkpoints/ppo_chunk_full",
    )
    
    return {
        "debug": debug_config,
        "full": full_config,
    }


def setup_logging_and_checkpointing(config: TrainingConfig):
    """Setup logging and checkpointing."""
    
    # Create checkpoint directory
    checkpoint_dir = pathlib.Path(config.ppo_config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup WandB logging
    if config.log_to_wandb:
        wandb.init(
            project=config.project_name,
            name=config.exp_name,
            config={
                "ppo_config": config.ppo_config.__dict__,
                "training_config": config.__dict__,
            }
        )
        logger.info("Initialized WandB logging")


def main():
    """Main training function."""
    init_logging()
    logger.info("Starting Chunk-based PPO training for Pi0FAST Actor-Critic")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Chunk-based PPO Training for Pi0FAST")
    parser.add_argument("--config", default="debug", choices=["debug", "full"],
                       help="Training configuration to use")
    parser.add_argument("--exp_name", default="chunk_ppo_aloha_experiment", 
                       help="Experiment name for logging and checkpoints")
    parser.add_argument("--no_wandb", action="store_true", 
                       help="Disable WandB logging")
    args = parser.parse_args()
    
    # Get PPO configuration
    ppo_configs = create_ppo_configs()
    ppo_config = ppo_configs[args.config]
    
    # Create training configuration
    training_config = TrainingConfig(
        exp_name=args.exp_name,
        ppo_config=ppo_config,
        log_to_wandb=not args.no_wandb,
    )
    
    logger.info(f"Using configuration: {args.config}")
    logger.info(f"Experiment name: {args.exp_name}")
    
    # Setup JAX
    if training_config.use_gpu and jax.device_count("gpu") > 0:
        logger.info(f"Using {jax.device_count('gpu')} GPU(s)")
    else:
        logger.info("Using CPU")
    
    # Setup logging and checkpointing
    setup_logging_and_checkpointing(training_config)
    
    # Create Aloha real environment
    env = AlohaRealEnvironmentWrapper(
        reset_position=training_config.reset_position,
        render_height=training_config.render_height,
        render_width=training_config.render_width,
        max_episode_steps=ppo_config.max_episode_steps,
    )
    logger.info("Created Aloha real environment wrapper")
    
    # Create PPO trainer
    trainer = ChunkPPOTrainer(config=ppo_config, env=env)
    
    # Start training
    try:
        training_metrics = trainer.train()
        logger.info("Training completed successfully!")
        logger.info(f"Final metrics summary:")
        logger.info(f"  Total episodes: {len(training_metrics['episode_rewards'])}")
        logger.info(f"  Avg episode reward: {np.mean(training_metrics['episode_rewards'][-10:]):.2f}")
        logger.info(f"  Total chunks: {len(training_metrics['chunk_rewards'])}")
        logger.info(f"  Avg chunk reward: {np.mean(training_metrics['chunk_rewards'][-50:]):.3f}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
        
    finally:
        # Cleanup
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()

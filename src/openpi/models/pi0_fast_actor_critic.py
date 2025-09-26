import dataclasses
import logging
from typing import Tuple, Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.pi0_fast import Pi0FAST, Pi0FASTConfig, make_attn_mask, left_to_right_align, PALIGEMMA_EOS_TOKEN
import numpy as np
import openpi.models.gemma_fast as _gemma
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


class ValueHead(nnx.Module):
    """Value head MLP for predicting state values from prelogits."""
    
    def __init__(self, embed_dim: int, hidden_dim: int = 1024, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(embed_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim // 2, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim // 2, 1, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, rngs=rngs)

    def __call__(self, x: at.Array, *, train: bool = False) -> at.Array:
        """
        Forward pass through value head.
        
        Args:
            x: [batch_size, embed_dim] prelogits features
            train: Whether in training mode
            
        Returns:
            [batch_size] state values
        """
        x = nnx.gelu(self.linear1(x))
        x = self.dropout(x, deterministic=not train)
        x = nnx.gelu(self.linear2(x))
        x = self.dropout(x, deterministic=not train)
        x = self.linear3(x)
        return x.squeeze(-1)


@dataclasses.dataclass(frozen=True)
class Pi0FASTActorCriticConfig(Pi0FASTConfig):
    """Configuration for PPO Actor-Critic model."""
    value_head_hidden_dim: int = 1024
    max_value_steps: int = 32  # Maximum steps for value computation


class Pi0FASTActorCritic(_model.BaseModel):
    """
    Actor-Critic model using Pi0FAST as backbone.
    
    This model combines the Pi0FAST architecture with a value head for PPO training.
    - Actor: Uses Pi0FAST for autoregressive action generation
    - Critic: Uses averaged prelogits from action generation through a value head
    """
    
    def __init__(self, config: Pi0FASTActorCriticConfig, *, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        
        # Initialize the actor backbone (Pi0FAST)
        self.actor_backbone = Pi0FAST(config, rngs=rngs)
        
        # Get embedding dimension from Gemma config
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        embed_dim = paligemma_config.width
        
        # Initialize value head
        self.value_head = ValueHead(embed_dim, config.value_head_hidden_dim, rngs=rngs)
        
        # Store config for later use
        self.config = config

    def get_value(
        self, 
        observation: _model.Observation, 
        rng: at.KeyArrayLike | None = None,
        *, 
        train: bool = False,
        max_steps: int = None,
        temperature: float = 0.0,
    ) -> at.Array:
        """
        Compute state value by performing autoregressive generation and averaging prelogits.
        
        Args:
            observation: Input observation
            rng: Random key for generation
            train: Whether in training mode
            max_steps: Maximum generation steps (defaults to config.max_value_steps)
            temperature: Sampling temperature (0.0 for greedy)
            
        Returns:
            [batch_size] state values
        """
        collected_logits, final_step = self.get_prelogits(observation, rng, train=train, max_steps=max_steps, temperature=temperature)
        
        # Average the collected prelogits (only valid steps)
        valid_mask = jnp.arange(max_steps)[:, None] < final_step[None, :]  # [max_steps, batch]
        valid_prelogits = collected_logits * valid_mask[..., None]  # [max_steps, batch, embed_dim]
        
        # Compute average (handle case where no valid steps)
        num_valid = jnp.sum(valid_mask, axis=0, keepdims=True)  # [1, batch]
        num_valid = jnp.clip(num_valid, 1)  # Avoid division by zero
        
        avg_prelogits = jnp.sum(valid_prelogits, axis=0) / num_valid.T  # [batch, embed_dim]
        

        # Compute value using the value head
        value = self.value_head(avg_prelogits, train=train)
        # # CRITICAL: Stop gradients to prevent value loss from affecting LLM parameters
        # # The value head should only train on the value prediction task, not influence action generation
        # avg_prelogits_stopped = jax.lax.stop_gradient(avg_prelogits)
        
        # # Compute value using the value head (only value_head parameters will be updated)
        # value = self.value_head(avg_prelogits_stopped, train=train)
        
        return value

    def infer_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int = 256,
        temperature: float = 0.0,
    ) -> _model.Actions:
        """
        Infer actions using the actor backbone (Pi0FAST) in evaluation mode.
        This uses train=False internally for stable inference.
        """
        return self.actor_backbone.sample_actions(
            rng=rng,
            observation=observation,
            max_decoding_steps=max_decoding_steps,
            temperature=temperature
        )

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int = 256,
        temperature: float = 0.0,
        train: bool = True,
    ) -> _model.Actions:
        """
        Sample actions using the actor backbone (Pi0FAST) with train mode control.
        
        Args:
            rng: Random key
            observation: Input observation
            max_decoding_steps: Maximum decoding steps
            temperature: Sampling temperature
            train: Training mode flag. If False, uses infer_actions for compatibility.
            
        Returns:
            Sampled actions
        """
        if not train:
            # For backward compatibility and stable inference
            return self.infer_actions(
                rng=rng,
                observation=observation,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature
            )
        
        # Training mode implementation - we need to implement train=True version
        # For now, we'll preprocess with train=True
        processed_obs = _model.preprocess_observation(
            rng, observation, train=True, image_keys=list(observation.images.keys())
        )
        
        # Use the existing Pi0FAST logic but with train=True preprocessing
        # Embed inputs and prepare for autoregressive generation
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.actor_backbone.embed_inputs(processed_obs)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        
        # Left to right align
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len
        
        # First fill KV cache with a forward pass of the prefix
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        prefix_logits, kv_cache, _ = self.actor_backbone.PaliGemma.llm(
            embedded_prefix=prefix_token_embeddings, 
            mask=prefix_attn_mask, 
            positions=prefix_positions, 
            decode=True
        )
        
        # Prepare decoding
        last_logit = prefix_logits[:, -1:]
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=jnp.int32)
        
        def step(carry):
            rng_key, last_logit, output_tokens, cache, _, step = carry
            
            # Sample next token
            rng_key, step_rng = jax.random.split(rng_key)
            
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(step_rng, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            
            # Store token
            output_tokens = output_tokens.at[:, step].set(token.squeeze(-1))
            
            # Check for early stopping (EOS token)
            from openpi.models.pi0_fast import PALIGEMMA_EOS_TOKEN
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)
            
            # Get token embedding for next step
            token_embedding = self.actor_backbone.PaliGemma.llm(token, embed_only=True)
            
            # Prepare attention mask and positions for this step
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] 
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            
            # Forward pass to get next logits
            next_logit, cache, _ = self.actor_backbone.PaliGemma.llm(
                embedded_prefix=token_embedding,
                mask=mask,
                positions=positions,
                decode=True,
                kv_cache=cache,
                return_prelogits=False
            )
            
            return rng_key, next_logit, output_tokens, cache, all_eos, step + 1
        
        def cond_fn(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)
        
        # Run generation loop
        final_rng, _, final_tokens, _, _, _ = jax.lax.while_loop(
            cond_fn, step, (rng, last_logit, output_tokens, kv_cache, False, 0)
        )
        
        # Extract actions from generated tokens using tokenizer
        if not hasattr(self, '_tokenizer'):
            from openpi.models.tokenizer import FASTTokenizer
            self._tokenizer = FASTTokenizer(
                max_len=self.config.max_token_len,
                fast_tokenizer_path="physical-intelligence/fast"
            )
        
        # Convert tokens to actions
        actions = self._tokenizer.extract_actions(
            np.array(final_tokens[0]), 
            self.action_horizon, 
            self.action_dim
        )
        
        return _model.Actions(actions=jnp.array(actions))

    @override
    def compute_loss(
        self, 
        rng: at.KeyArrayLike, 
        observation: _model.Observation, 
        actions: _model.Actions, 
        *, 
        train: bool = False
    ) -> at.Array:
        """
        Compute loss using the actor backbone (Pi0FAST).
        This is a direct delegation to the Pi0FAST compute_loss method.
        """
        return self.actor_backbone.compute_loss(
            rng=rng,
            observation=observation,
            actions=actions,
            train=train
        )

    def get_prelogits(
        self, 
        observation: _model.Observation, 
        rng: at.KeyArrayLike | None = None,
        *, 
        train: bool = False,
        max_steps: int = None,
        temperature: float = 0.0,
    ) -> at.Array:
        """
        Compute state value by performing autoregressive generation and averaging prelogits.
        
        Args:
            observation: Input observation
            rng: Random key for generation
            train: Whether in training mode
            max_steps: Maximum generation steps (defaults to config.max_value_steps)
            temperature: Sampling temperature (0.0 for greedy)
            
        Returns:
            [batch_size] state values
        """
        if rng is None:
            rng = jax.random.key(42)
        
        if max_steps is None:
            max_steps = self.config.max_value_steps
            
        # Preprocess observation
        processed_obs = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )
        
        # Embed inputs and prepare for autoregressive generation
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.actor_backbone.embed_inputs(processed_obs)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        
        # Left to right align
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len
        batch_size = prefix_token_embeddings.shape[0]
        
        # Initialize KV cache with prefix
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        
        # Get initial prelogits and KV cache
        prefix_prelogits, kv_cache, _ = self.actor_backbone.PaliGemma.llm(
            embedded_prefix=prefix_token_embeddings,
            mask=prefix_attn_mask,
            positions=prefix_positions,
            decode=True,
            return_prelogits=True
        )
        
        # Start autoregressive generation to collect prelogits
        last_prelogit = prefix_prelogits[:, -1:]  # [batch, 1, embed_dim]
        collected_prelogits = []
        
        # Convert to logits for sampling (only when needed)
        embedder_decode = lambda x: self.actor_backbone.PaliGemma.llm.module.embedder.decode(x)
        
        def step_fn(carry):
            rng_key, last_prelogit, cache, step, collected_prelogits = carry
            
            # Sample next token from prelogits
            rng_key, step_rng = jax.random.split(rng_key)
            # logits = embedder_decode(last_prelogit)  # Convert to logits for sampling
            logits, _ = self.actor_backbone.PaliGemma.llm(
                pre_logits=last_prelogit
            )

            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(step_rng, logits / temperature, axis=-1),
                lambda _: jnp.argmax(logits, axis=-1),
                operand=None,
            )
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

            # Check for early stopping
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)
            
            # Get token embedding for next step
            token_embedding = self.actor_backbone.PaliGemma.llm(token, embed_only=True)
            
            # Prepare attention mask and positions for this step
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_steps)[None, None, :] 
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            
            # Forward pass to get next prelogits
            next_prelogit, cache, _ = self.actor_backbone.PaliGemma.llm(
                embedded_prefix=token_embedding,
                mask=mask,
                positions=positions,
                decode=True,
                kv_cache=cache,
                return_prelogits=True
            )
            
            # Collect prelogits for value computation
            collected_prelogits = collected_prelogits.at[step].set(next_prelogit[:, 0, :])  # [batch, embed_dim]
            
            return rng_key, next_prelogit, cache, step + 1, collected_prelogits, all_eos
        
        def cond_fn(carry):
            _, _, _, step, _, all_eos = carry
            return (~all_eos) & (step < max_steps)
        
        # Initialize collected prelogits array
        init_collected = jnp.zeros((max_steps, batch_size, last_prelogit.shape[-1]))
        
        # Run autoregressive loop
        final_rng, _, _, final_step, final_collected, _ = jax.lax.while_loop(
            cond_fn, 
            step_fn, 
            (rng, last_prelogit, kv_cache, 0, init_collected, False)
        )
        
        return final_collected, final_step


    def get_logits(
        self, 
        observation: _model.Observation, 
        rng: at.KeyArrayLike | None = None,
        *, 
        train: bool = False,
        max_steps: int = None,
        temperature: float = 0.0,
    ) -> at.Array:
        """
        Compute state value by performing autoregressive generation and averaging prelogits.
        
        Args:
            observation: Input observation
            rng: Random key for generation
            train: Whether in training mode
            max_steps: Maximum generation steps (defaults to config.max_value_steps)
            temperature: Sampling temperature (0.0 for greedy)
            
        Returns:
            [batch_size] state values
        """
        if rng is None:
            rng = jax.random.key(42)
        
        if max_steps is None:
            max_steps = self.config.max_value_steps
            
        # Preprocess observation
        processed_obs = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )
        
        # Embed inputs and prepare for autoregressive generation
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.actor_backbone.embed_inputs(processed_obs)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        
        # Left to right align
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len
        batch_size = prefix_token_embeddings.shape[0]
        
        # Initialize KV cache with prefix
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        
        # Get initial prelogits and KV cache
        prefix_logits, kv_cache, _ = self.actor_backbone.PaliGemma.llm(
            embedded_prefix=prefix_token_embeddings,
            mask=prefix_attn_mask,
            positions=prefix_positions,
            decode=True,
        )
        
        # Start autoregressive generation to collect prelogits
        last_logit = prefix_logits[:, -1:]  # [batch, 1, embed_dim]
        collected_logits = []
        
        # Convert to logits for sampling (only when needed)
        # embedder_decode = lambda x: self.actor_backbone.PaliGemma.llm.module.embedder.decode(x)
        
        def step_fn(carry):
            rng_key, last_logit, cache, step, collected_logits = carry
            
            # Sample next token from prelogits
            rng_key, step_rng = jax.random.split(rng_key)
            # logits = embedder_decode(last_logit)  # Convert to logits for sampling
            
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(step_rng, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)
            # Check for early stopping
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)
            
            # Get token embedding for next step
            token_embedding = self.actor_backbone.PaliGemma.llm(token, embed_only=True)
            
            # Prepare attention mask and positions for this step
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_steps)[None, None, :] 
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            
            # Forward pass to get next prelogits
            next_logit, cache, _ = self.actor_backbone.PaliGemma.llm(
                embedded_prefix=token_embedding,
                mask=mask,
                positions=positions,
                decode=True,
                kv_cache=cache,
                return_prelogits=False
            )
            
            # Collect prelogits for value computation
            collected_logits = collected_logits.at[step].set(last_logit[:, 0, :])  # [batch, embed_dim]
            
            return rng_key, next_logit, cache, step + 1, collected_logits, all_eos
        
        def cond_fn(carry):
            _, _, _, step, _, all_eos = carry
            return (~all_eos) & (step < max_steps)
        
        # Initialize collected prelogits array
        init_collected = jnp.zeros((max_steps, batch_size, last_logit.shape[-1]))
        
        # Run autoregressive loop
        final_rng, _, _, final_step, final_collected, _ = jax.lax.while_loop(
            cond_fn, 
            step_fn, 
            (rng, last_logit, kv_cache, 0, init_collected, False)
        )
        
        return final_collected, final_step

    def compute_action_chunk_log_prob(
        self,
        observation: _model.Observation,
        actions: _model.Actions,
        rng: at.KeyArrayLike | None = None,
        *,
        train: bool = False,
    ) -> at.Array:
        """
        Compute log probability of entire action chunk (sum of all token log probs).
        This is used for PPO actor loss computation.
        
        Args:
            observation: Input observation
            actions: Action chunk to compute probability for
            rng: Random key
            train: Whether in training mode
            
        Returns:
            Log probability of the entire action chunk
        """
        if rng is None:
            rng = jax.random.key(42)
        
        # Import tokenizer here to avoid circular imports
        from openpi.models.tokenizer import FASTTokenizer
        import numpy as np
        
        # Initialize tokenizer if not exists
        if not hasattr(self, '_tokenizer'):
            self._tokenizer = FASTTokenizer(
                max_len=self.config.max_token_len,
                fast_tokenizer_path="physical-intelligence/fast"
            )
        
        # Convert actions to tokens to get target token sequence
        action_array = np.array(actions.actions)
        tokens, _, _, loss_mask = self._tokenizer.tokenize(
            prompt="Perform the task.",
            state=np.zeros(14),  # Simplified state for Aloha
            actions=action_array
        )
        action_tokens = tokens[jnp.where(loss_mask)[0]]
        
        # Get model's predicted logits
        logits_sequence, final_step = self.get_logits(
            observation=observation,
            rng=rng,
            train=train,
            max_steps=self.config.max_value_steps
        )
        
        # Compute log probabilities for the action token sequence
        if len(action_tokens) > 0 and final_step > 0:
            steps = min(len(action_tokens), int(final_step))
            action_logits = logits_sequence[-steps:, 0, :]  # [steps, vocab_size]
            target_tokens = action_tokens[-steps:]
            
            # Sum log probabilities of all tokens in the chunk (classic approach)
            log_probs = jax.nn.log_softmax(action_logits, axis=-1)
            token_log_probs = log_probs[jnp.arange(steps), target_tokens]
            chunk_log_prob = jnp.sum(token_log_probs)
            
            return chunk_log_prob
        
        return jnp.array(0.0)

    def compute_action_chunk_value(
        self,
        observation: _model.Observation,
        actions: _model.Actions,
        rng: at.KeyArrayLike | None = None,
        *,
        train: bool = False,
    ) -> at.Array:
        """
        Compute value for a specific observation-action chunk pair.
        This gives us V(s,a) instead of just V(s).
        
        Args:
            observation: Input observation
            actions: Action chunk to compute value for
            rng: Random key
            train: Whether in training mode
            
        Returns:
            Value estimate for this obs-action pair
        """
        # For simplicity, we'll use the existing get_value method
        # In practice, you might want to condition the value on the action
        return self.get_value(observation, rng, train=train)
    

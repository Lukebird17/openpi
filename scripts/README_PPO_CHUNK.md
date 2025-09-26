# Action Chunk Based PPO Training for Pi0FAST Actor-Critic

这是一个基于action chunk的PPO训练实现，专门为Pi0FAST Actor-Critic模型和Aloha真实环境设计。

## 核心特性

### 🔄 **Action Chunk级别的PPO**
- **数据粒度**: 每个action chunk对应一个reward
- **GAE计算**: 在chunk级别计算优势估计
- **经典PPO损失**: 使用标准的actor loss公式

### 🧠 **模型架构增强**
- **扩展的ActorCritic类**: 添加了chunk级别的方法
- **Token级别概率**: 通过logits直接计算action chunk概率
- **Value函数**: 支持obs-action chunk对的价值估计

### 🦾 **Aloha环境集成**
- **真实机器人**: 支持Aloha双臂机器人
- **Chunk执行**: 完整执行action sequence并收集累积reward
- **环境包装**: 适配PPO训练接口

## 文件结构

```
scripts/
├── train_ppo_chunk_based.py        # 主训练脚本
├── README_PPO_CHUNK.md            # 本说明文档
└── action_chunk_ppo_example.py     # 理论示例代码

src/openpi/models/
└── pi0_fast_actor_critic.py       # 扩展的ActorCritic类
```

## 🎯 预训练模型加载

该训练脚本支持从本地预训练Pi0FAST模型进行online fine-tuning：

```python
# 配置本地预训练模型路径
debug_config = PPOConfig(
    pretrained_checkpoint_path="./checkpoints/pi0_fast_base/params",  # 本地路径
    model_config=Pi0FASTActorCriticConfig(
        action_dim=7,           # ✅ 匹配预训练模型
        action_horizon=10,      # ✅ 匹配预训练模型
        max_token_len=180,      # ✅ 匹配预训练模型
    ),
)
```

**统一格式支持**：
- 🔄 **自动检测**: 智能识别Pi0FAST IL格式 vs RL格式
- ✅ **IL→RL**: Pi0FAST预训练参数 → `actor_backbone` + 随机`value_head`
- ✅ **RL→RL**: 完整Actor-Critic参数直接加载
- ✅ **形状验证**: 自动检查参数兼容性
- ✅ **优雅降级**: 加载失败时自动回退到随机初始化

**支持的加载场景**：
```python
# 1. 从Pi0FAST IL预训练开始 (7DOF → actor_backbone)
pretrained_checkpoint_path = "./checkpoints/pi0_fast_base/params"

# 2. 从之前的RL训练继续 (完整Actor-Critic)  
pretrained_checkpoint_path = "./checkpoints/ppo_chunk_aloha/step_100/params"

# 3. 在不同RL实验间切换
pretrained_checkpoint_path = "./checkpoints/experiment_A/step_500/params"
```

**保存格式**：
- 💾 **统一格式**: 所有保存的checkpoint都使用RL格式 (`actor_backbone` + `value_head`)
- 📂 **目录结构**: `./checkpoints/{config.checkpoint_dir}/step_{N}/params/`
- 📄 **元数据**: 包含训练状态和模型配置信息 (`metadata.json`)
- 🔄 **兼容性**: 保存的checkpoint可被后续训练直接加载

## 核心算法

### 1. Action Chunk概率计算

```python
def compute_action_chunk_log_prob(self, observation, actions, rng):
    # 1. 将action chunk转换为token序列
    tokens = tokenizer.tokenize(actions)  # [t1, t2, ..., tn]
    
    # 2. 获得模型对每个token的预测logits
    logits_sequence = model.get_logits(observation)  # [steps, vocab_size]
    
    # 3. 计算token序列的总log概率（经典方法）
    log_probs = log_softmax(logits_sequence)
    token_log_probs = log_probs[range(n), tokens]
    chunk_log_prob = sum(token_log_probs)  # 所有token概率之和
    
    return chunk_log_prob
```

### 2. 经典PPO Actor Loss

```python
def compute_ppo_loss(obs, actions, old_log_prob, advantage):
    # 1. 计算新模型的action chunk概率
    new_log_prob = model.compute_action_chunk_log_prob(obs, actions)
    
    # 2. 计算概率比值
    ratio = exp(new_log_prob - old_log_prob)
    
    # 3. PPO clipped损失（经典公式）
    surr1 = ratio * advantage
    surr2 = clip(ratio, 1-ε, 1+ε) * advantage
    actor_loss = -min(surr1, surr2)
    
    return actor_loss
```

### 3. Chunk级别GAE

```python
def compute_chunk_gae(chunk_rewards, chunk_values, gamma, lambda):
    advantages = zeros_like(chunk_rewards)
    
    gae = 0
    for t in reversed(range(len(chunk_rewards))):
        delta = chunk_rewards[t] + gamma * next_value - chunk_values[t]
        gae = delta + gamma * lambda * gae
        advantages[t] = gae
    
    return advantages
```

## 使用方法

### 快速开始

```bash
# 调试模式（快速测试）
python scripts/train_ppo_chunk_based.py --config=debug --exp_name=test_chunk_ppo

# 完整训练
python scripts/train_ppo_chunk_based.py --config=full --exp_name=aloha_chunk_ppo_production

# 禁用WandB日志
python scripts/train_ppo_chunk_based.py --config=debug --exp_name=test_chunk_ppo --no_wandb
```

### 配置选项

#### Debug配置（快速测试）
```python
debug_config = PPOConfig(
    total_timesteps=5_000,          # 5K步
    num_chunks_per_update=32,       # 每次更新32个chunks
    num_epochs_per_update=2,        # 2个PPO epochs
    batch_size=8,                   # 小批次
    learning_rate=1e-4,            # 较低学习率
)
```

#### Full配置（生产训练）
```python
full_config = PPOConfig(
    total_timesteps=500_000,        # 500K步
    num_chunks_per_update=128,      # 每次更新128个chunks
    num_epochs_per_update=4,        # 4个PPO epochs
    batch_size=32,                  # 标准批次
    learning_rate=3e-4,            # 标准学习率
)
```

### 模型配置

```python
model_config = Pi0FASTActorCriticConfig(
    action_dim=14,              # Aloha 14自由度
    action_horizon=16,          # Action chunk大小
    max_token_len=180,          # 最大token长度
    paligemma_variant="gemma_2b",  # 2B模型
    value_head_hidden_dim=512,     # Value头维度
    max_value_steps=32,         # Value计算最大步数
)
```

## 训练流程

### 1. 数据收集阶段
```python
for chunk_idx in range(num_chunks_per_update):
    # 采样action chunk
    actions = model.sample_actions(observation)
    
    # 计算chunk价值和概率
    chunk_value = model.compute_action_chunk_value(obs, actions)
    chunk_log_prob = model.compute_action_chunk_log_prob(obs, actions)
    
    # 执行完整chunk，收集累积reward
    next_obs, chunk_reward, done, info = env.step_chunk(actions)
    
    # 存储chunk级别数据
    buffer.add(obs, actions, chunk_reward, chunk_value, chunk_log_prob)
```

### 2. GAE计算阶段
```python
# 在chunk级别计算GAE
rewards = [chunk_rewards...]
values = [chunk_values...]

for t in reversed(range(len(rewards))):
    delta = rewards[t] + gamma * next_value - values[t]
    gae = delta + gamma * lambda * gae
    advantages[t] = gae
```

### 3. 模型更新阶段
```python
for epoch in range(num_epochs_per_update):
    for batch in create_batches(buffer):
        # 计算新的chunk概率
        new_log_probs = [model.compute_action_chunk_log_prob(obs, actions) 
                        for obs, actions in batch]
        
        # PPO损失
        ratios = exp(new_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = clip(ratios) * advantages
        actor_loss = -min(surr1, surr2)
        
        # 更新模型
        update_model(actor_loss + value_loss)
```

## 关键优势

### ✅ **理论正确性**
- 符合Pi0FAST的token化设计
- 使用经典PPO公式
- 在合适的抽象层级操作

### ✅ **计算效率**
- Chunk级别的GAE减少计算量
- 批量处理token序列
- 有效利用GPU并行化

### ✅ **实现简洁**
- 标准PPO算法，无复杂修改
- 清晰的数据流
- 易于调试和扩展

### ✅ **实际可用**
- 支持真实Aloha机器人
- 完整的日志和检查点系统
- 灵活的配置选项

## 🚀 使用方法

### 1. 准备预训练模型

确保预训练Pi0FAST模型放置在正确的本地路径：

```bash
# 预训练模型目录结构
./checkpoints/
└── pi0_fast_base/
    └── params/          # Pi0FAST预训练参数
        ├── checkpoint
        └── ... (其他checkpoint文件)
```

### 2. 运行训练

```bash
# 使用debug配置（快速测试，包含预训练模型）
python scripts/train_ppo_chunk_based.py --config=debug --exp_name=ppo_test

# 使用完整配置（长时间训练）  
python scripts/train_ppo_chunk_based.py --config=full --exp_name=ppo_full_run
```

### 3. 自定义预训练路径

```python
# 修改配置中的预训练模型路径
pretrained_checkpoint_path = "/path/to/your/pi0fast/params"
```

### 4. 无预训练模式

```python
# 设置为None以从头开始训练
pretrained_checkpoint_path = None
```

## 监控指标

训练过程中会记录以下指标：

- **Episode Reward**: 每个episode的总奖励
- **Chunk Reward**: 每个action chunk的奖励
- **Policy Loss**: PPO actor损失
- **Value Loss**: 价值函数损失
- **Training Progress**: 更新进度和性能

## 故障排除

### 常见问题

1. **内存不足**
   - 减少`num_chunks_per_update`
   - 降低`batch_size`
   - 使用CPU训练

2. **收敛缓慢**
   - 调整学习率
   - 修改PPO参数（`clip_coef`, `ent_coef`）
   - 检查reward函数设计

3. **环境连接问题**
   - 确认Aloha机器人硬件连接
   - 检查环境配置
   - 验证权限设置

### 调试技巧

```bash
# 使用debug配置进行快速测试
python scripts/train_ppo_chunk_based.py --config=debug --no_wandb

# 检查chunk reward分布
# 在代码中添加: print(f"Chunk reward: {chunk_reward}")

# 监控概率比值
# 在代码中添加: print(f"Ratio: {ratio}")
```

## 扩展方向

### 可能的改进

1. **更复杂的Value函数**
   - 真正的V(s,a)计算
   - 基于attention的obs-action融合

2. **自适应chunk大小**
   - 动态调整action_horizon
   - 基于任务复杂度的适应

3. **多模态reward**
   - 环境reward + 人类反馈
   - 内在动机奖励

4. **分布式训练**
   - 多GPU并行
   - 异步Actor-Learner架构

## 参考

- [PPO原始论文](https://arxiv.org/abs/1707.06347)
- [Pi0FAST论文](相关链接)
- [OpenPI框架文档](../README.md)
- [Aloha机器人文档](../examples/aloha_real/README.md)

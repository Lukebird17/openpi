# JAX版本的PPO训练脚本

这是基于原始PyTorch notebook的JAX版本实现，使用JAX/Flax框架进行强化学习训练。

## 特性

- 🚀 **高性能**: 使用JAX的JIT编译和向量化操作
- 🎯 **支持多种环境**: CartPole (离散动作) 和 Pendulum (连续动作)
- 🔧 **易于配置**: 通过Config类轻松调整超参数
- 📊 **可视化**: 自动绘制训练结果和移动平均
- 🎲 **可重现**: 支持随机种子设置

## 安装依赖

```bash
pip install -r requirements_jax.txt
```

或者手动安装：

```bash
pip install jax jaxlib flax optax gymnasium numpy matplotlib tqdm
```

## 使用方法

### 1. 命令行使用

```bash
# 训练CartPole环境
python train_ppo_jax.py --env CartPole-v1 --episodes 500

# 训练Pendulum环境
python train_ppo_jax.py --env Pendulum-v1 --episodes 2000

# 显示训练结果图表
python train_ppo_jax.py --env CartPole-v1 --episodes 500 --plot
```

### 2. 作为模块使用

```python
from train_ppo_jax import Config, PPOTrainer, PPOContinuousTrainer

# 创建配置
config = Config(
    env_name="CartPole-v1",
    num_episodes=500,
    hidden_dim=128,
    actor_lr=3e-4,
    critic_lr=1e-3
)

# 创建训练器并训练
trainer = PPOTrainer(config)
returns = trainer.train()
```

### 3. 使用示例脚本

```bash
# 运行CartPole训练
python example_jax_ppo.py cartpole

# 运行Pendulum训练
python example_jax_ppo.py pendulum

# 比较不同学习率的性能
python example_jax_ppo.py compare
```

## 配置参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `env_name` | "CartPole-v1" | 环境名称 |
| `num_episodes` | 500 | 训练episode数 |
| `hidden_dim` | 128 | 隐藏层维度 |
| `actor_lr` | 3e-4 | 策略网络学习率 |
| `critic_lr` | 1e-3 | 价值网络学习率 |
| `gamma` | 0.98 | 折扣因子 |
| `lmbda` | 0.95 | GAE参数 |
| `epochs` | 10 | 每次更新的epoch数 |
| `eps` | 0.2 | PPO裁剪参数 |
| `seed` | 0 | 随机种子 |

## 环境支持

### 离散动作环境
- **CartPole-v1**: 经典的杆平衡任务
- 动作空间: 离散 (0, 1)
- 状态空间: 4维连续

### 连续动作环境
- **Pendulum-v1**: 倒立摆控制任务
- 动作空间: 1维连续 [-2, 2]
- 状态空间: 3维连续

## 性能优势

相比PyTorch版本，JAX版本具有以下优势：

1. **JIT编译**: 自动优化计算图
2. **向量化**: 高效的批处理操作
3. **内存效率**: 更好的内存管理
4. **GPU加速**: 自动GPU支持
5. **函数式编程**: 更清晰的代码结构

## 训练结果

### CartPole-v1
- 目标: 获得200+的episode奖励
- 通常需要: 100-300 episodes
- 最终性能: 稳定的200+奖励

### Pendulum-v1
- 目标: 最小化累积惩罚 (接近-200)
- 通常需要: 1000-2000 episodes
- 最终性能: 稳定的低惩罚值

## 故障排除

### 常见问题

1. **JAX安装问题**:
   ```bash
   # 对于CPU
   pip install jax jaxlib
   
   # 对于GPU (CUDA)
   pip install jax[cuda] jaxlib
   ```

2. **内存不足**:
   - 减少 `num_episodes`
   - 减少 `hidden_dim`
   - 减少 `epochs`

3. **训练不稳定**:
   - 调整学习率
   - 调整 `eps` 参数
   - 增加 `epochs`

## 扩展功能

### 添加新环境

1. 继承 `PPOTrainer` 或 `PPOContinuousTrainer`
2. 重写 `take_action` 方法
3. 根据需要调整网络架构

### 自定义网络

```python
class CustomPolicyNet(nn.Module):
    def setup(self):
        # 自定义网络结构
        pass
    
    def __call__(self, x):
        # 前向传播
        pass
```

## 许可证

与原始项目保持一致的许可证。






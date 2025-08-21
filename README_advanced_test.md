# FLUX API 高级吞吐量测试工具

基于优化思路的高级吞吐量测试工具，可以测试不同优化策略对性能的影响。

## 🚀 主要特性

### 1. **多策略测试**
- **baseline**: 基准测试 (当前实现)
- **model_pool**: 模型池优化测试
- **batch**: 批量推理测试  
- **async**: 异步推理测试

### 2. **系统监控**
- 实时 GPU 利用率监控
- VRAM 使用情况跟踪
- CPU 和内存使用率监控
- 自动数据收集和分析

### 3. **性能分析**
- 吞吐量对比分析
- 响应时间分布 (P50, P95, P99)
- 效率分数计算
- 资源利用率分析

### 4. **可视化报告**
- 性能对比图表
- 雷达图分析
- 详细统计报告

## 📦 安装依赖

```bash
# 基础依赖
pip install aiohttp psutil numpy

# 可选: 图表生成
pip install matplotlib
```

## 🎯 使用方法

### 基本测试
```bash
# 测试基准性能
python test_throughput_advanced.py --requests 20 --concurrent 3

# 测试所有优化策略
python test_throughput_advanced.py --requests 20 --concurrent 3 --strategies all

# 测试特定策略
python test_throughput_advanced.py --requests 30 --concurrent 5 --strategies model_pool batch
```

### 高级测试
```bash
# 生成对比图表
python test_throughput_advanced.py --requests 50 --concurrent 5 --strategies all --plot

# 保存详细结果
python test_throughput_advanced.py --requests 100 --concurrent 10 --strategies all --save

# 完整测试
python test_throughput_advanced.py --requests 50 --concurrent 5 --strategies all --plot --save
```

## 📊 命令行参数

| 参数 | 说明 | 默认值 | 选项 |
|------|------|--------|------|
| `--url` | API 基础URL | `http://localhost:8000` | 任意URL |
| `--requests` | 每个策略的请求数量 | `20` | 正整数 |
| `--concurrent` | 最大并发数 | `3` | 正整数 |
| `--strategies` | 测试策略 | `baseline` | `baseline`, `model_pool`, `batch`, `async`, `all` |
| `--plot` | 生成图表 | `False` | 布尔值 |
| `--save` | 保存结果 | `False` | 布尔值 |

## 📈 测试策略说明

### 1. **Baseline (基准)**
- 使用当前的 `/generate` 端点
- 测试现有实现的性能
- 作为其他策略的对比基准

### 2. **Model Pool (模型池)**
- 假设实现了模型池优化
- 测试并发模型访问的性能
- 预期吞吐量提升 3-5x

### 3. **Batch (批量推理)**
- 假设实现了批量推理端点 `/batch-generate`
- 测试批量处理的性能
- 预期吞吐量提升 1.5-2x

### 4. **Async (异步推理)**
- 假设实现了异步推理端点 `/async-generate`
- 测试异步处理的性能
- 预期响应时间减少 30-50%

## 📋 输出示例

```
🚀 FLUX API 高级吞吐量测试工具
📍 API URL: http://localhost:8000
📊 测试配置: 20 请求/策略, 3 并发
🎯 测试策略: ['baseline', 'model_pool', 'batch', 'async']

📊 测试基准吞吐量: 20 请求, 3 并发
🏊 测试模型池优化: 20 请求, 5 并发
📦 测试批量推理: 20 请求, 3 并发
⚡ 测试异步推理: 20 请求, 3 并发

====================================================================================================
📊 优化策略对比报告
====================================================================================================
策略            吞吐量      成功率   平均响应时间   P95响应时间   GPU利用率  效率分数  
----------------------------------------------------------------------------------------------------
baseline       0.25        100.0%   15.23s       18.45s       45.2%      0.342     
model_pool     1.15        100.0%   12.87s       15.23s       78.5%      0.623     
batch          0.42        100.0%   13.45s       16.78s       65.3%      0.456     
async          0.28        100.0%   10.89s       13.45s       52.1%      0.478     
====================================================================================================

🏆 最佳策略: model_pool
   效率分数: 0.623
   吞吐量: 1.15 请求/秒
   平均响应时间: 12.87秒

📈 性能提升分析 (相对于基准):
   model_pool: 吞吐量 +360.0%, 响应时间 -15.5%
   batch: 吞吐量 +68.0%, 响应时间 -11.7%
   async: 吞吐量 +12.0%, 响应时间 -28.5%
====================================================================================================
```

## 🔧 自定义测试

### 修改测试参数
```python
# 在代码中修改测试参数
payload = {
    "prompt": prompt,
    "num_inference_steps": 25,  # 修改推理步数
    "guidance_scale": 3.5,      # 修改引导比例
    "width": 512,               # 修改图像宽度
    "height": 512               # 修改图像高度
}
```

### 添加新的测试策略
```python
# 在 AdvancedFluxTester 类中添加新方法
async def test_custom_throughput(self, num_requests: int, max_concurrent: int = 3):
    """测试自定义优化策略"""
    return await self._test_with_strategy("custom", num_requests, max_concurrent)

# 在 _test_single_request 方法中添加新端点
elif strategy == "custom":
    endpoint = "/custom-generate"  # 自定义端点
```

## 📊 性能指标说明

### 1. **吞吐量 (Throughput)**
- 单位: 请求/秒
- 衡量系统处理能力
- 越高越好

### 2. **响应时间 (Response Time)**
- 平均响应时间: 所有请求的平均值
- P95 响应时间: 95% 请求的响应时间
- P99 响应时间: 99% 请求的响应时间

### 3. **效率分数 (Efficiency Score)**
- 综合考虑吞吐量、响应时间和 GPU 利用率
- 范围: 0-1，越高越好
- 计算公式: `throughput_score * 0.5 + response_time_score * 0.3 + gpu_score * 0.2`

### 4. **资源利用率**
- GPU 利用率: GPU 计算能力使用百分比
- VRAM 使用率: 显存使用百分比
- CPU 使用率: CPU 使用百分比
- 内存使用率: 系统内存使用百分比

## 🎯 优化建议

### 基于测试结果的优化方向

1. **如果 GPU 利用率低 (< 50%)**
   - 增加并发数
   - 实现模型池
   - 启用批量推理

2. **如果响应时间长 (> 20s)**
   - 减少推理步数
   - 使用更快的模型
   - 实现异步推理

3. **如果 VRAM 使用率高 (> 90%)**
   - 优化内存管理
   - 使用量化模型
   - 减少批大小

4. **如果吞吐量低 (< 0.5 请求/秒)**
   - 实现模型池
   - 增加并发数
   - 优化推理过程

## 📄 输出文件

### JSON 结果文件
```json
{
  "test_config": {
    "url": "http://localhost:8000",
    "requests_per_strategy": 20,
    "concurrent": 3,
    "strategies": ["baseline", "model_pool", "batch", "async"]
  },
  "performance_results": [...],
  "system_metrics": [...],
  "detailed_results": [...]
}
```

### 图表文件
- `performance_comparison.png`: 性能对比图表
- 包含吞吐量、响应时间、效率分数和雷达图

## 🔍 故障排除

### 常见问题

1. **API 端点不存在**
   ```bash
   # 只测试基准策略
   python test_throughput_advanced.py --strategies baseline
   ```

2. **GPU 监控失败**
   ```bash
   # 检查 nvidia-smi 是否可用
   nvidia-smi
   ```

3. **内存不足**
   ```bash
   # 减少请求数量
   python test_throughput_advanced.py --requests 10 --concurrent 2
   ```

4. **图表生成失败**
   ```bash
   # 安装 matplotlib
   pip install matplotlib
   ```

这个高级测试工具可以帮助你全面评估不同优化策略的效果，为性能优化提供数据支持。 
# FLUX API 吞吐量测试工具

测试 FLUX API 的并发处理能力和响应时间。

## 快速开始

```bash
# 安装依赖
pip install aiohttp

# 运行测试
python test_throughput_simple.py --requests 10 --concurrent 3
```

## 命令行参数

- `--url`: API URL (默认: http://localhost:8000)
- `--requests`: 测试请求数量 (默认: 10)
- `--concurrent`: 最大并发数 (默认: 3)
- `--test-type`: 测试类型 (concurrent)
- `--save`: 保存结果到JSON文件

## 使用示例

```bash
# 基本测试
python test_throughput_simple.py

# 高负载测试
python test_throughput_simple.py --requests 50 --concurrent 5



# 保存结果
python test_throughput_simple.py --save
```

## 测试指标

- 吞吐量 (请求/秒)
- 响应时间统计
- 成功率
- VRAM 使用情况
- 生成时间分析 

这个工具用于测试 FLUX API 的并发处理能力和响应时间，帮助你了解系统的性能表现。

## 📁 文件说明

- `test_throughput_fp4.py` - 完整版测试工具（包含图表功能）
- `test_throughput_simple.py` - 简化版测试工具（无图表依赖）
- `README_throughput_test.md` - 本说明文档

## 🚀 快速开始

### 1. 安装依赖

```bash
# 完整版（需要matplotlib）
pip install aiohttp matplotlib numpy

# 简化版（只需要aiohttp）
pip install aiohttp
```

### 2. 启动 FLUX API 服务

```bash
# 确保 FLUX API 正在运行
python main_fp4.py
```

### 3. 运行测试

```bash
# 使用简化版进行基本测试
python test_throughput_simple.py --requests 10 --concurrent 3

# 使用完整版并生成图表
python test_throughput_fp4.py --requests 20 --concurrent 5 --plot

# 测试远程服务器
python test_throughput_simple.py --url http://your-server:8000 --requests 50
```

## 📊 测试类型

### 并发测试 (Concurrent)
- 同时发送多个请求到 `/generate` 端点
- 测试系统的并发处理能力
- 使用信号量控制最大并发数

## ⚙️ 命令行参数

```bash
python test_throughput_simple.py [选项]

选项:
  --url URL               API 基础URL (默认: http://localhost:8000)
  --requests N            测试请求数量 (默认: 10)
  --concurrent N          最大并发数 (默认: 3)
  --test-type TYPE        测试类型: concurrent/queue/both (默认: both)
  --save                  保存详细结果到JSON文件
  --plot                  生成图表 (仅完整版)
```

## 📈 测试指标

### 核心指标
- **吞吐量**: 每秒处理的请求数 (requests/second)
- **响应时间**: 从发送请求到收到响应的总时间
- **生成时间**: 模型实际生成图像的时间
- **成功率**: 成功完成的请求百分比

### 统计指标
- 平均响应时间
- 中位数响应时间
- 最小/最大响应时间
- 响应时间标准差
- VRAM 使用情况

## 🎯 使用场景

### 1. 性能基准测试
```bash
# 测试不同并发数的性能
python test_throughput_simple.py --requests 50 --concurrent 1
python test_throughput_simple.py --requests 50 --concurrent 3
python test_throughput_simple.py --requests 50 --concurrent 5
```

### 2. 负载测试
```bash
# 高负载测试
python test_throughput_simple.py --requests 100 --concurrent 10 --save
```



### 3. 远程服务器测试
```bash
# 测试远程部署的API
python test_throughput_simple.py --url http://your-server:8000 --requests 20
```

## 📋 输出示例

```
🚀 FLUX API 吞吐量测试工具 (简化版)
📍 API URL: http://localhost:8000
📊 测试配置: 10 请求, 3 并发
🎯 测试类型: both

🔍 检查系统状态...
模型状态: True
GPU 信息: {'gpu_count': 1, 'gpu_0': {...}}
队列统计: {'queue_size': 0, 'processing': 0}

🔄 开始并发测试...
🚀 开始并发测试: 10 个请求, 最大并发数: 3

============================================================
📊 并发请求 测试报告
============================================================
📈 总体统计:
   总请求数: 10
   成功请求: 10
   失败请求: 0
   成功率: 100.00%
   总耗时: 45.23秒
   吞吐量: 0.22 请求/秒

⏱️  响应时间:
   平均响应时间: 13.45秒
   中位数响应时间: 12.87秒
   最小响应时间: 11.23秒
   最大响应时间: 18.92秒
   响应时间标准差: 2.34秒

🎨 生成时间:
   平均生成时间: 12.89秒
   中位数生成时间: 12.45秒
============================================================

📋 测试结果汇总表
================================================================================
请求ID    状态     响应时间(s)   生成时间(s)   VRAM使用    
================================================================================
req_0000  ✅      13.45        12.89        8.5GB        
req_0001  ✅      12.87        12.45        8.5GB        
req_0002  ✅      14.23        13.67        8.5GB        
...
================================================================================
```

## 🔧 高级配置

### 自定义测试参数
你可以修改脚本中的测试参数：

```python
# 在 test_single_request 方法中修改
payload = {
    "prompt": prompt,
    "num_inference_steps": 25,  # 修改推理步数
    "guidance_scale": 3.5,      # 修改引导比例
    "width": 512,               # 修改图像宽度
    "height": 512               # 修改图像高度
}
```

### 自定义提示词
修改 `test_prompts` 列表来使用不同的测试提示词：

```python
self.test_prompts = [
    "你的自定义提示词1",
    "你的自定义提示词2",
    # ...
]
```

## 📊 结果分析

### 性能评估标准
- **优秀**: 吞吐量 > 0.5 请求/秒，成功率 > 95%
- **良好**: 吞吐量 0.2-0.5 请求/秒，成功率 > 90%
- **一般**: 吞吐量 < 0.2 请求/秒，成功率 > 80%

### 优化建议
1. **低吞吐量**: 考虑增加 GPU 数量或使用更快的模型
2. **高响应时间**: 检查网络延迟或模型加载状态
3. **高失败率**: 检查系统资源或模型配置
4. **VRAM 不足**: 考虑使用量化模型或减少并发数

## 🐛 故障排除

### 常见问题

1. **连接超时**
   ```bash
   # 增加超时时间
   timeout=aiohttp.ClientTimeout(total=600)  # 10分钟
   ```

2. **内存不足**
   ```bash
   # 减少并发数
   python test_throughput_simple.py --concurrent 1
   ```

3. **API 未响应**
   ```bash
   # 检查服务状态
   curl http://localhost:8000/health
   ```

4. **图表显示问题**
   ```bash
   # 使用简化版
   python test_throughput_simple.py --requests 10
   ```

### 调试模式
```bash
# 启用详细日志
export PYTHONPATH=.
python -u test_throughput_simple.py --requests 5 --concurrent 1
```

## 📄 输出文件

### JSON 结果文件
使用 `--save` 参数会生成包含详细测试数据的 JSON 文件：

```json
{
  "test_config": {
    "url": "http://localhost:8000",
    "requests": 10,
    "concurrent": 3,
    "test_type": "both"
  },
  "system_status": {
    "model_status": {...},
    "gpu_info": {...},
    "queue_stats": {...}
  },
  "test_results": [
    {
      "request_id": "req_0000",
      "start_time": 1234567890.123,
      "end_time": 1234567903.456,
      "response_time": 13.333,
      "status_code": 200,
      "success": true,
      "generation_time": 12.89,
      "vram_usage": "8.5GB"
    }
  ]
}
```

### 图表文件 (仅完整版)
使用 `--plot` 参数会生成性能分析图表：
- `throughput_test_results.png` - 包含响应时间分布、生成时间分布、请求时间线和实时吞吐量

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个测试工具！

## 📝 许可证

本项目采用 MIT 许可证。 
 
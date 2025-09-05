# EigenAI  && Sekai图像需求说明文档

## 🎯 核心开发目标

为背景图生成 API 集成 NSFW 内容审核和 S3 文件上传功能，简化客户端处理，提升整体服务性能和稳定性。

## 📝 详细需求

### 1. NSFW 内容审核

- **模型**: 使用 `Falconsai/nsfw_image_detection` 模型进行 NSFW 内容检测。
- **集成点**: 在图片生成后，立即使用该模型进行检测。
- **输出**: 返回 `nsfw: true/false` 结果。
- **性能**: 检测时间应尽可能短，不影响整体生成速度。

### 2. S3 文件上传

- **功能**: 将经过审核并通过的图片上传到指定的 S3 存储桶。
- **配置**: 接收客户端提供的 `s3_prefix`（预签名URL）。
- **流程**: 图片生成后，立即上传到 S3。
- **文件名处理**: 自动从预签名URL中提取占位符文件名（如 `1.png`），并替换为 `{image_hash}.jpg`，其中 `image_hash` 是生成图片的MD5哈希值。
- **输出**: 返回更新后的 S3 URL (`s3_url`)，其中包含实际的文件名 `{image_hash}.jpg`。

## 🛠️ 技术实现细节

### 1. NSFW 检测模块

- **模型加载**: 建议在服务启动时预加载模型到内存，以减少检测延迟。
- **错误处理**: 检测失败时设置 nsfw_score = 1.0，但继续生成和上传图片。
- **超时机制:** 超过5s时设置 nsfw_score = 1.0，但继续生成和上传图片。
- ***注意：NSFW 检测仅用于评分，不会阻止图片生成或上传。所有生成的图片都会被上传到 S3。***

### 2. S3 上传模块

- **~~S3 客户端**: 使用 AWS SDK (如 `boto3`) 进行 S3 上传。~~
- 给到签名过的 s3地址（http），直接 往地址上面 put即可
- **文件命名**: 
  - 客户端提供的 URL 包含占位符文件名（如 `1.png`, `2.jpg` 等）
  - 系统自动将占位符替换为 `{image_hash}.jpg`，其中 `image_hash` 是生成图片的MD5哈希值
  - 这确保每个图片都有唯一的文件名，避免缓存问题
- **重试机制**: 实现上传失败时的自动重试机制（指数退避：1s, 2s, 4s）。
- **监控告警**: 增加上传失败率监控，及时发现并解决问题。

## ⚙️ API 接口变更

### 1. 接口定义

```
POST /generate
{
  "prompt": "positive prompt",
  "width": 512,
  "height": 1024,
  "num_inference_steps": 15,
  "response_format": "s3",
  "upscale": "true",
  "s3_prefix": "https://prod-data.sekai.chat/aiu-character/000/1.png?AWSAccessKeyId=AKIAQE43KJDN7ARTLAVM&Signature=%2ByiRa6eTIiuPtE3wGWzFzmS3snA%3D&Expires=1756921542",
  "enable_nsfw_check": "true"
}

Response:
{
  "data": {
    "s3_url": "https://prod-data.sekai.chat/aiu-character/000/d41d8cd98f00b204e9800998ecf8427e.jpg?AWSAccessKeyId=AKIAQE43KJDN7ARTLAVM&Signature=%2ByiRa6eTIiuPtE3wGWzFzmS3snA%3D&Expires=1756921542",
    "nsfw_score": 0.05,
    "image_hash": "d41d8cd98f00b204e9800998ecf8427e",
    "s3_upload_status": 200
  }
}
```

### 2. 参数说明

- `prompt`: (必选) 正向提示词。
- `width`: (必选) 图片宽度 (默认为 512)。
- `height`: (必选) 图片高度 (默认为 1024)。
- `num_inference_steps`: (必选) 推理步骤数 (默认为 15)。
- `response_format`: (必选) 响应格式 (默认为 "s3")。
- `upscale`: (必选) 是否进行放大 (默认为 "true")。
- `s3_prefix`: (必选) **完整的** 签名过的 **S3 Http URI，直接往地址put就行 (例如：https://prod-data.sekai.chat/aiu-character/000/1.png?AWSAccessKeyId=AKIAQE43KJDN7ARTLAVM&Signature=%2ByiRa6eTIiuPtE3wGWzFzmS3snA%3D&Expires=1756921542)。注意：URL中的文件名（如 `1.png`）是占位符，系统会自动替换为 `{image_hash}.jpg`。
- `enable_nsfw_check`: (必选) 是否启用 NSFW 检测 (true/false)。
- **负向提示词？？**

### 3. 响应说明

- `data`: (JSON 对象) 包含以下字段：
    - `s3_url`: (字符串) 上传到 S3 的图片 Http URL（文件名已替换为 `{image_hash}.jpg`）。
    - `nsfw_score`: (Float) NSFW 内容 的置信度
    - `image_hash`: (字符串) 生成图片内容的MD5哈希值（与S3 URL中的文件名对应）。
    - `s3_upload_status`: (Integer) HTTP status code from S3 upload (200/204 for success).

## 🧪 测试要求

### 1. 功能测试

- 确保 NSFW 检测能够正确识别 NSFW 内容。
- 确保 S3 文件上传能够成功完成，并且文件具有公开读取权限。
- 确保 API 接口能够正确接收参数，并返回正确的 S3 URL 和 NSFW 结果。
- 确保涵盖诸多可能的异常情况

### 2. 性能测试

- 测量集成 NSFW 检测和 S3 上传后的 API 响应时间，确保没有明显的性能下降。（正常情况下新增的这两个功能1-3s可以完成）

# **TODO:**

- [x]  nsfw代码示例

## NSFW Detection Implementation Guide

### Basic Usage
```python
from PIL import Image
from transformers import pipeline

# Load image
img = Image.open("<path_to_image_file>")

# Initialize classifier (do this once at startup)
classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

# Run detection
results = classifier(img)

# results will be like `[{'label': 'normal', 'score': 0.9998652935028076}, {'label': 'nsfw', 'score': 0.00013474702427629381}]`
```

### Expected Output Format
```python
# Classifier returns a list of dictionaries with scores for each class:
[
    {'label': 'nsfw', 'score': 0.954},  # NSFW probability
    {'label': 'normal', 'score': 0.046}  # Safe content probability
]
```

### CUDA-Optimized Setup
```python
import torch
from transformers import pipeline

# Initialize with CUDA for faster inference
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "image-classification", 
    model="Falconsai/nsfw_image_detection",
    device=device  # Use GPU if available
)
```

### Integration Considerations
- **Model Loading**: Load once at service startup to avoid repeated initialization overhead
- **Timeout Handling**: Implement 5-second timeout using asyncio or threading
- **Error Handling**: On failure/timeout, return nsfw=true (fail-safe approach)
- **Memory Management**: Model requires ~500MB GPU memory
- **Batch Processing**: Consider batching multiple images for better throughput
- **Response Format**: Extract nsfw score from classifier output: `nsfw_score = next(r['score'] for r in results if r['label'] == 'nsfw')`

- [ ]  s3上传代码示例—需要
  
    ```python
    import requests
    
    # 预签名的 URL
    url = "https://prod-data.sekai.chat/aiu-character/000/1.png?AWSAccessKeyId=AKIAQE43KJDN7ARTLAVM&Signature=%2ByiRa6eTIiuPtE3wGWzFzmS3snA%3D&Expires=1756921542"
    
    # 要上传的文件路径
    file_path = "local.png"
    
    with open(file_path, "rb") as f:
        response = requests.put(url, data=f)
    
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        print("上传成功")
    else:
        print("上传失败:", response.text)
    ```
    
- [ ]  图片大小问题， 图片jpeg格式
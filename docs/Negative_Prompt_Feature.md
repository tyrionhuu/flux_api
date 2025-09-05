# Negative Prompt Support in Sekai API

## Overview
The Sekai API now supports negative prompts, allowing users to specify elements they want to exclude from generated images. This feature leverages the built-in negative prompt support in the FLUX model pipeline.

## Implementation Details

### API Changes

#### 1. Request Models
Added `negative_prompt` field to:
- `GenerateRequest` - Main generation endpoint
- `ImageUploadGenerateRequest` - Image-to-image generation

Field specification:
```python
negative_prompt: Optional[str] = Field(
    None, 
    description="Negative prompt to exclude unwanted elements from image generation"
)
```

#### 2. Updated Endpoints

##### `/generate` endpoint
```json
{
    "prompt": "A beautiful sunset over mountains",
    "negative_prompt": "dark, cloudy, rainy, stormy",
    "width": 512,
    "height": 512,
    "num_inference_steps": 15
}
```

##### `/submit-request` endpoint (Queue)
```json
{
    "prompt": "A serene lake",
    "negative_prompt": "people, buildings, pollution",
    "width": 512,
    "height": 512
}
```

##### `/upload-image-generate` endpoint
Form data now includes:
- `negative_prompt` (optional): Text describing what to exclude

### Queue Manager Support
The queue system fully supports negative prompts:
- Stores negative_prompt in `QueueRequest` dataclass
- Returns negative_prompt in status queries
- Passes negative_prompt through to model generation

### Model Integration
The FLUX model (`fp4_flux_model.py`) already had negative prompt support built-in. The API layer now properly passes this parameter through:
- `generate_image()` method accepts `negative_prompt` parameter
- Pipeline uses negative_prompt during inference if provided

## Usage Examples

### Basic Usage
```bash
curl -X POST "http://localhost:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "A sunny beach with palm trees",
        "negative_prompt": "rain, storm, dark clouds, people",
        "width": 512,
        "height": 512
    }'
```

### Common Negative Prompts
- Quality issues: `"blurry, low quality, pixelated, distorted"`
- Unwanted elements: `"text, watermark, signature, logo"`
- Style modifiers: `"dark, gloomy, horror, scary"`
- Content exclusion: `"people, animals, buildings, vehicles"`

### Multi-GPU Deployment
The negative prompt feature works seamlessly with the multi-GPU deployment:
```bash
./start_multi_gpu.sh -m fp4_sekai
```

All 8 GPU instances support negative prompts through the nginx load balancer.

## Backward Compatibility
- Fully backward compatible - negative_prompt is optional
- Defaults to `None` if not specified
- Existing API calls continue to work without modification

## Testing
Use the provided test script:
```bash
python test_negative_prompt.py
```

This tests:
1. Generation without negative prompt
2. Generation with negative prompt
3. Queue submission with negative prompt

## Performance Impact
- Minimal performance impact
- No additional GPU memory required
- Processing time remains the same

## Best Practices
1. Keep negative prompts concise and focused
2. Use comma-separated keywords for clarity
3. Test different negative prompts to find optimal results
4. Combine with positive prompt refinement for best results

## Technical Details
- Negative prompts are processed by the FLUX transformer's text encoder
- Applied during the diffusion denoising process
- Influences the latent space to avoid specified concepts

## Future Enhancements
- Weighted negative prompts (e.g., `"ugly:1.5, blurry:0.8"`)
- Per-LoRA negative prompt templates
- Automatic negative prompt suggestions based on prompt analysis
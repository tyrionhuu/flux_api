# Internal S3 Upload Feature

## Overview
The Sekai API now automatically uploads both input requests and generated images to an internal S3 bucket for archival and tracking purposes. This happens asynchronously in the background without affecting API response times.

## What Gets Uploaded

For each image generation request, two files are uploaded to S3:

1. **Input Request JSON** (`input-{image_hash}.json`)
   - Contains all request parameters
   - Includes prompt, dimensions, seed, LoRA settings, etc.
   - Includes generation timestamp and time taken

2. **Output Image** (`output-{image_hash}.jpg`)
   - The generated image in JPEG format
   - Uses the same hash as the input for correlation

## S3 Location

Files are uploaded to:
- **Bucket**: `customers-sekai-background-image-gen`
- **Path**: `customers/sekai/background-image-gen/`
- **ACL**: Private (files are not publicly accessible)

## Configuration

The feature can be enabled/disabled in `config/s3_internal_config.py`:
```python
ENABLE_INTERNAL_S3_UPLOAD = True  # Set to False to disable
```

## How It Works

1. After image generation, a unique hash is calculated from the image content
2. The upload task is triggered asynchronously using a thread pool
3. Both input JSON and output image are uploaded with matching hashes
4. Upload failures are logged but don't affect the API response
5. The feature works with all response formats (binary, json, s3)

## Testing

Several test scripts are provided:

1. **Test internal S3 upload directly**:
   ```bash
   python tests/test_internal_s3_upload.py
   ```

2. **Test API with S3 upload**:
   ```bash
   python tests/test_api_with_s3.py
   ```

3. **Verify uploaded files**:
   ```bash
   python tests/verify_s3_uploads.py <image_hash>
   ```

## Monitoring

Check the API logs for S3 upload status:
- Look for `"Triggered async internal S3 upload for hash: ..."`
- Success: `"Internal S3: Successfully uploaded ..."`
- Failure: `"Internal S3: Failed to upload ..."`

## Implementation Files

- `config/s3_internal_config.py` - S3 configuration and presigned POST generation
- `utils/internal_s3_uploader.py` - Upload utilities for JSON and images
- `utils/async_tasks.py` - Background task executor
- `api/sekai_routes.py` - Modified to trigger async uploads

## Notes

- Uploads are fire-and-forget (non-blocking)
- Uses presigned POST for secure uploads without SDK
- Files are stored with private ACL for security
- The same S3 credentials from the test example are used
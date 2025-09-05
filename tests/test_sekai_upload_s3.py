import os
import requests

# === Presigned POST data from your output ===
url = "https://customers-sekai-background-image-gen.s3.amazonaws.com/"
fields = {
    "acl": "private",
    "key": "customers/sekai/background-image-gen/${filename}",
    "AWSAccessKeyId": "AKIAUBX3L5WZRFJCJA5G",
    "policy": "eyJleHBpcmF0aW9uIjogIjIwMjUtMDktMTJUMDU6Mjg6MzRaIiwgImNvbmRpdGlvbnMiOiBbWyJzdGFydHMtd2l0aCIsICIka2V5IiwgImN1c3RvbWVycy9zZWthaS9iYWNrZ3JvdW5kLWltYWdlLWdlbi8iXSwgWyJjb250ZW50LWxlbmd0aC1yYW5nZSIsIDAsIDUwMDAwMDAwXSwgeyJhY2wiOiAicHJpdmF0ZSJ9LCB7ImJ1Y2tldCI6ICJjdXN0b21lcnMtc2VrYWktYmFja2dyb3VuZC1pbWFnZS1nZW4ifSwgWyJzdGFydHMtd2l0aCIsICIka2V5IiwgImN1c3RvbWVycy9zZWthaS9iYWNrZ3JvdW5kLWltYWdlLWdlbi8iXV19",
    "signature": "uMapoYNvLIrn2BWudLNuJ4d9xv8="
}
# ============================================

# Local file to upload
file_path = "example.jpg"

with open(file_path, "rb") as f:
    files = {"file": (os.path.basename(file_path), f)}
    response = requests.post(url, data=fields, files=files, timeout=120)

print("Status Code:", response.status_code)
if response.status_code == 204:
    key = fields["key"].replace("${filename}", os.path.basename(file_path))
    print("Upload successful ✅")
    print("File stored at:", f"s3://customers-sekai-background-image-gen/{key}")
else:
    print("Upload failed ❌")
    print(response.text[:500])

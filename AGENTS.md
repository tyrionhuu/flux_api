# Repository Guidelines

## Global Style Rule
- Always code in Linux Torvalds style. For C/C++ or shell, follow Linux kernel style (tabs, K&R braces, 80 cols). For Python, use PEP8/Black and Torvalds principles: small functions, early returns, simple over clever.

## Project Structure & Modules
- `api/` — FastAPI routes and Pydantic models (`sekai_routes.py`, `models.py`).
- `models/` — FLUX model logic (FP4 + upscaler), GPU-heavy code.
- `utils/` — helpers: queueing, GPU/NSFW checks, S3 upload, cleanup.
- `config/` — runtime settings (ports, queue limits, S3). Avoid committing secrets.
- `frontend/` — optional static UI. `generated_images/`, `uploads/`, and `logs/` are runtime artifacts.

## Build, Test, and Development
- Create env: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- Run API (single instance): `python main_fp4_sekai.py` (defaults to `FP4_API_PORT=8000`).
- Alt dev server: `uvicorn main_fp4_sekai:app --reload --host 0.0.0.0 --port 8000`.
- Multi‑GPU: `./start_multi_gpu.sh -m fp4_sekai` (ports start at 23333). Set `CUDA_VISIBLE_DEVICES` to restrict GPUs.
- Quick smoke test (API running on 8000): `python tests/test_upscaler_api.py`
- Logs: `tail -f logs/flux_api_fp4.log` or `./scripts/manage_logs.sh follow`.

## Coding Style & Naming
- Python 3.12, 4‑space indentation, PEP8 with type hints for new/changed code (apply Torvalds principles noted above).
- Format: `black .`  | Lint: `ruff check .` (use `--fix` when safe).
- Names: modules/files `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`; constants `UPPER_SNAKE`.
- API paths: hyphenated, nouns + verbs (e.g., `/apply-lora`, `/generate`).

## Testing Guidelines
- Tests are executable scripts in `tests/` that hit a running API.
- Default target is `http://localhost:8000` (some scripts use `8080` behind nginx). Adjust constants if needed.
- Recommended: run `tests/test_upscaler_api.py` and `tests/test_sekai_api.py` after changes. Coverage is not enforced yet.

## Commit & PR Guidelines
- Commits: concise, imperative subject (e.g., "Add queue backoff logic"). Group related changes only.
- Before PR: run `black . && ruff check .` and verify key tests above.
- PR description: purpose, user‑visible changes (routes/params), test evidence (logs or sample responses), and any config/env keys touched.
- Do not include large binaries or runtime artifacts; keep `generated_images/` and `uploads/` out of Git.

## Security & Config Tips
- Never commit real secrets. Prefer env vars: `FP4_API_PORT`, `MAX_CONCURRENT_REQUESTS`, `ENABLE_INTERNAL_S3_UPLOAD`, AWS creds, etc.
- `config/s3_internal_config.py` contains placeholders — override via environment or secret manager in production.

## Agent Notes
- Add new endpoints in `api/sekai_routes.py`; extend request/response models in `api/models.py`.
- Keep changes minimal and localized; update docs/tests when behavior or routes change.

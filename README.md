# Multi-Modal Document Intelligence - Module 0

This repository contains the foundational backend structure for a Multi-Modal Document Intelligence system.

## Backend

The backend uses FastAPI with centralized configuration and logging.

### Run the server

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and adjust values if needed.

3. Start the server:

```bash
uvicorn app.main:app --reload
```

### API Endpoints

- `GET /`
  - Returns `{ "message": "Multi-Modal Document Intelligence API Running" }`
- `GET /health`
  - Returns `{ "status": "healthy" }`
- `POST /upload`
  - Accepts `multipart/form-data` and uploads files with extensions `.pdf`, `.docx`, `.png`, `.jpg`, `.jpeg`
  - Returns upload metadata including `file_id`, `original_name`, `stored_name`, `file_size`, `file_type`, and `upload_time`

## Notes

- Configuration values are loaded from `.env` using `python-dotenv`.
- Logs are written to `logs/app.log`.
- This module contains the project foundation only and does not implement AI, OCR, vector databases, authentication, upload processing, or frontend functionality.

# Advanced Medical AI

## Requirements
- Python 3.10+
- pip

## Install dependencies
```bash
pip install -r requirements.txt
```

## Run the project
### Windows
```bash
start.bat
```

### Manual run
Backend:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Frontend:
```bash
streamlit run frontend/app.py
```

## Notes
- The backend API runs on port 8000.
- The frontend runs on port 8501.
- Set `API_BASE_URL` if you need to point the frontend to a different backend URL.

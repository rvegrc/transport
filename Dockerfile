FROM python:3.10-slim

WORKDIR /app

COPY best_model.joblib .

COPY hellodapi.py .

COPY requirements.txt .

RUN pip install -r requirements.txt

ENTRYPOINT [ "uvicorn", "hellodapi:app", "--host", "0.0.0.0", "--port", "6000"]
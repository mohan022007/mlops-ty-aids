FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir fastapi uvicorn numpy joblib scikit-learn

EXPOSE 8000

CMD ["uvicorn", "exp4:app", "--host", "0.0.0.0", "--port", "8000"]
# Setup Guide

## Prerequisites

- **Docker Desktop** ([download](https://www.docker.com/products/docker-desktop))
- **Python 3.11+** (if running locally without Docker)

---

## Quick Start (Docker - Recommended)

```bash
# 1. Train the model
docker run --rm -v "$(pwd)":/app -w /app python:3.11-slim bash -c "\
  pip install -q pandas scikit-learn xgboost && \
  python scripts/create_model_v5.py"

# 2. Start the API
docker-compose up --build

# 3. Test the API
curl http://localhost:8000/health
```

API available at: **http://localhost:8000**

---

## Alternative: Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python scripts/create_model_v5.py

# 3. Start the API
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

---

## Making Requests

### Full prediction (17 fields required)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3, "bathrooms": 2.5, "sqft_living": 2000,
    "sqft_lot": 8000, "floors": 1.5, "sqft_above": 2000,
    "sqft_basement": 0, "waterfront": 0, "view": 0,
    "condition": 3, "grade": 7, "yr_built": 1990,
    "yr_renovated": 0, "lat": 47.5112, "long": -122.257,
    "zipcode": "98103", "sqft_living15": 1840, "sqft_lot15": 7620
  }'
```

### Minimal prediction (only zipcode required)
```bash
curl -X POST http://localhost:8000/predict-minimal \
  -H "Content-Type: application/json" \
  -d '{"zipcode": "98103"}'
```

---

## Running Tests

```bash
# Ensure API is running first
docker-compose up -d

# Install test dependencies
pip install pytest requests

# Run tests
pytest tests/test_housing_endpoints.py -v
```

---

## Error Handling

The API handles errors at multiple levels:

- **422 Unprocessable Entity**: Invalid data types or missing required fields
- **400 Bad Request**: Invalid zipcode (not in demographics data)
- **500 Internal Server Error**: Unexpected errors during prediction

All errors are logged with details. View logs:
```bash
docker-compose logs -f
```

---

## Troubleshooting

**Model not found?**
```bash
python scripts/create_model_v5.py
```

**Port 8000 in use?**
```bash
# Change port in docker-compose.yml
ports:
  - "8001:8000"
```

**Check API health:**
```bash
curl http://localhost:8000/health
```

---

## Documentation

Interactive API docs: **http://localhost:8000/docs**
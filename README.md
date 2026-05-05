# Customer Churn MLOps on GCP

End-to-end MLOps pipeline demonstrating training, tracking, deployment, and CI/CD on Google Cloud Platform.

## Architecture

```
Local/VS Code -> GitHub -> GitHub Actions
                              |
                              v
GCS (data + models) <- Compute Engine VM (training + MLflow) -> Artifact Registry -> Cloud Run (inference)
```

## Stack

- **Storage**: GCS
- **Compute**: Compute Engine (e2-medium VM)
- **Tracking**: MLflow on VM
- **CI/CD**: GitHub Actions
- **Registry**: Artifact Registry
- **Serving**: Cloud Run
- **Project**: `upgrad-101`

## Quickstart (local dev)

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Run training on VM

```bash
python src/prepare_data.py --input-csv data/telco_churn.csv --bucket upgrad-101-churn-mlops
python src/train.py \
  --bucket upgrad-101-churn-mlops \
  --mlflow-uri http://VM_EXTERNAL_IP:5000 \
  --experiment-name churn-prediction
```

## API

- `GET  /health` - service status
- `POST /predict` - returns predictions + probabilities

See `IMPLEMENTATION_GUIDE.docx` for the full step-by-step.

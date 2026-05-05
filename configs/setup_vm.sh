#!/bin/bash
# Run this ONCE on the Compute Engine VM after first SSH login.
# Sets up Python, installs deps, starts MLflow tracking server, configures Docker.

set -e

echo "==> Updating system..."
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git docker.io tmux

echo "==> Adding user to docker group..."
sudo usermod -aG docker $USER

echo "==> Creating Python venv..."
python3 -m venv ~/mlops-env
source ~/mlops-env/bin/activate

echo "==> Installing Python deps..."
pip install --upgrade pip
pip install mlflow==2.14.1 google-cloud-storage==2.17.0 \
            pandas==2.2.2 scikit-learn==1.5.0 joblib==1.4.2 \
            numpy==1.26.4

echo "==> Starting MLflow tracking server in tmux..."
# Backend store: SQLite on local disk; artifacts: GCS
tmux new-session -d -s mlflow \
  "source ~/mlops-env/bin/activate && \
   mlflow server \
     --host 0.0.0.0 \
     --port 5000 \
     --backend-store-uri sqlite:///$HOME/mlflow.db \
     --default-artifact-root gs://upgrad-101-churn-mlops/mlflow-artifacts"

echo "==> Done. MLflow UI: http://VM_EXTERNAL_IP:5000"
echo "    Reattach: tmux attach -t mlflow"

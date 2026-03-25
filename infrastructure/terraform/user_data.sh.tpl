#!/bin/bash
set -euo pipefail

# Log all output
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting PowerInfer EC2 user data script..."

# Install NVIDIA driver (simplified - in production use Deep Learning AMI)
# For G5 instances: driver already present in Deep Learning AMI
# This script assumes GPU drivers are available

# Install Docker
echo "Installing Docker..."
apt-get update -y
apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start Docker
systemctl start docker
systemctl enable docker

# Add Ubuntu user to docker group
usermod -aG docker ubuntu

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update -y
apt-get install -y nvidia-docker2
systemctl restart docker

# Verify GPU is visible
echo "Verifying GPU access..."
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi || echo "WARNING: nvidia-smi failed in container"

# Download model from S3
echo "Downloading model from ${model_s3_uri}..."
S3_BUCKET="${s3_bucket}"
S3_KEY="${s3_key}"
MODEL_PATH="/models/$(basename $S3_KEY)"

mkdir -p /models
aws s3 cp "s3://${S3_BUCKET}/${S3_KEY}" "$MODEL_PATH"

# Run PowerInfer container
echo "Starting PowerInfer_x64 server..."
docker run -d \
  --name powerinfer \
  --gpus all \
  -p 8080:8080 \
  -v /models:/models:ro \
  -e POWERINFER_MODEL_PATH="/models/$(basename $S3_KEY)" \
  -e POWERINFER_PORT=8080 \
  -e POWERINFER_CONCURRENCY=4 \
  -e POWERINFER_LOG_LEVEL=info \
  --restart unless-stopped \
  powerinfer:latest

echo "User data script completed. PowerInfer should be running on port 8080."

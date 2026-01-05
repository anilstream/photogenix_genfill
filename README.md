# Photogenix Genfill

# Test locally
docker build -t 'genfill_api:latest' .
docker run -d -p 5007:5007 --gpus all genfill_api
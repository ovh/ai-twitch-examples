# OVHcloud - Live Twitch 15/12 - AI Deploy Beta

Launch the AI Notebook:

```bash
ovhai notebook run conda jupyterlab --framework-version conda-py38-cuda11.3-v22-4 --gpu 1
```

Launch the AI Deploy app:

```bash
ovhai app run --cpu 4 --default-http-port 8000 priv-registry.gra.training.ai.cloud.ovh.net/ai-deploy-portfolio/fastapi-sentiment-classification
```

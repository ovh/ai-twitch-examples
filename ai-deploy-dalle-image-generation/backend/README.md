# OVHcloud dockerfile for deploying dalle-playground over AI-App

## Compile the docker

First you need to clone the dalle-playground repository : https://github.com/saharmor/dalle-playground

```
git clone https://github.com/saharmor/dalle-playground.git
```

```
docker build dalle-playground/backend/ -f ovhcloud.Dockerfile -t <your-registry>/dalle-playground:latest
```

## Push you container into your registry

```
docker push <your-registry>/dalle-playground:latest
```

## Launch you backend app

```
ovhai app run \
    --probe-path / \
    --gpu 1 \
    -p 8080 \
    <your-registry>/dalle-playground:latest \
    -- python3 app.py --port 8080 --model_version mega_full
```
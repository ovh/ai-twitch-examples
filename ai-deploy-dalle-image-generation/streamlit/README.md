# Streamlit frontend for dalle-playground backend

## Compile the docker

```
docker build . -t <chosen-image-name>
```

## Run the docker container locally

```
docker run -p 8501:8501 -e BACKEND_URL=<dalle-playground-backend-url> <chosen-image-name>
```

Your frontend should now be accessible on http://localhost:8501
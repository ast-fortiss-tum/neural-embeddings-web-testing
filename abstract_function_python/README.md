# Flask BERT App - Docker

This is a simple Flask app that uses fine-tuned BERT models to classify pairs of states. The app is containerized using Docker.

### Building the image

Example:

```
docker build -t flask-classifier-app .
```

### Running the app

Example:

```
docker run -e FEATURE=content -e HF_MODEL_NAME=lgk03/NDD-claroline_test-content flask-classifier-app
```

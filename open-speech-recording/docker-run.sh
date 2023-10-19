#!/bin/bash

IMAGE_NAME="fooby-speech-recording"
DOCKERFILE_PATH="."

# Check if the image exists
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
  echo "Image $IMAGE_NAME does not exist. Building..."
  docker build -t $IMAGE_NAME $DOCKERFILE_PATH
fi

# Run the container with environment variables
docker run -p 8080:5000 --env-file .env.docker -e GOOGLE_APPLICATION_CREDENTIALS=/app/.secrets/fooby-research-ba-2b78fe420798.json $IMAGE_NAME



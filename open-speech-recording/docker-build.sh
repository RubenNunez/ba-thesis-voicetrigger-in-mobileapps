#!/bin/bash

IMAGE_NAME="fooby-speech-recording"
DOCKERFILE_PATH="."

# Stop containers using the image
containers=$(docker ps -a -q --filter ancestor=$IMAGE_NAME)
if [ ! -z "$containers" ]; then
    echo "Stopping and removing containers associated with image $IMAGE_NAME..."
    docker stop $containers
    docker rm $containers
fi

# Remove the image
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" != "" ]]; then
    echo "Removing image $IMAGE_NAME..."
    docker rmi $IMAGE_NAME
fi

# Rebuild the image
echo "Rebuilding image $IMAGE_NAME..."
docker build -t $IMAGE_NAME $DOCKERFILE_PATH

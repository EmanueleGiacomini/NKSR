#!/bin/bash

echo "Building docker"

docker build -f Dockerfile -t nksr:localbuild .

echo "Done."

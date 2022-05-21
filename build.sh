#!/bin/sh

docker build -t eidos-service.di.unito.it/bragagnolo/zero-grad:latest . -f Dockerfile.python
docker push eidos-service.di.unito.it/bragagnolo/zero-grad:latest

docker build -t eidos-service.di.unito.it/bragagnolo/zero-grad:sweep . -f Dockerfile.sweep
docker push eidos-service.di.unito.it/bragagnolo/zero-grad:sweep

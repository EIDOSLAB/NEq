#!/bin/sh

docker build -t eidos-service.di.unito.it/bragagnolo/neq:latest . -f Dockerfile.python
docker push eidos-service.di.unito.it/bragagnolo/neq:latest

docker build -t eidos-service.di.unito.it/bragagnolo/neq:sweep . -f Dockerfile.sweep
docker push eidos-service.di.unito.it/bragagnolo/neq:sweep

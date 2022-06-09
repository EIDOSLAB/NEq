#!/bin/sh

docker build -t eidos-service.di.unito.it/bragagnolo/neq:base . -f Dockerfile
docker push eidos-service.di.unito.it/bragagnolo/neq:base

docker build -t eidos-service.di.unito.it/bragagnolo/neq:python . -f Dockerfile.python
docker push eidos-service.di.unito.it/bragagnolo/neq:python

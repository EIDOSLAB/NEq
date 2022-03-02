FROM eidos-service.di.unito.it/eidos-base-pytorch:1.10.0

COPY src /src
WORKDIR /src

ENTRYPOINT ["python3"]
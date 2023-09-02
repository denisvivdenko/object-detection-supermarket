FROM python:3.10-slim-buster

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

ARG GIT_TOKEN
RUN git clone https://$GIT_TOKEN@github.com/denisvivdenko/object-detection-supermarket.git

WORKDIR /object-detection-supermarket

COPY requirements.txt requirements.txt
RUN pip install -e .

COPY gcp-credentials.json .
ENV GOOGLE_APPLICATION_CREDENTIALS=gcp-credentials.json

RUN dvc pull
RUN dvc repro

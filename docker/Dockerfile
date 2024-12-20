FROM alpine AS helper

RUN apk add --no-cache git git-lfs
RUN git clone https://huggingface.co/IvanKuchin/pancreas_cancer_classification /app

FROM python:3.12 AS app

ARG APP_FOLDER=/app

COPY . ${APP_FOLDER}

WORKDIR ${APP_FOLDER}

RUN pip install tensorflow
RUN pip install -r requirements.txt

COPY --from=helper /app/weights.keras ${APP_FOLDER}/checkpoints/weights.keras

CMD ["python", "./src/pancreas_ai/bin/predict.py"]

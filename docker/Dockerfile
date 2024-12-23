FROM alpine AS helper

RUN apk add --no-cache git git-lfs
RUN git clone https://huggingface.co/IvanKuchin/pancreas_cancer_classification /app

FROM python:3.12 AS app

ARG APP_FOLDER=/app

RUN pip install tensorflow totalsegmentator

RUN python `python -c "import os; import totalsegmentator; print(os.path.dirname(totalsegmentator.__file__))"`/download_pretrained_weights.py

# taken from TotalSegmentator Dockerfile 
# https://github.com/wasserth/TotalSegmentator/blob/master/Dockerfile
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 xvfb -y
RUN pip install fury

WORKDIR ${APP_FOLDER}

COPY . ${APP_FOLDER}

RUN pip install .

COPY --from=helper /app/weights.keras ${APP_FOLDER}/checkpoints/weights.keras

CMD ["predict"]

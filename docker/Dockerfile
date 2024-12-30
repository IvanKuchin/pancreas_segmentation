FROM alpine AS helper

RUN apk add --no-cache git git-lfs
RUN git clone https://huggingface.co/IvanKuchin/pancreas_cancer_classification /app

FROM python:3.12-slim AS app

ARG APP_FOLDER=/app
ENV PATH_BACKUP="$PATH"


############################
# Install TotalSegmentator #
############################

# taken from TotalSegmentator Dockerfile
# https://github.com/wasserth/TotalSegmentator/blob/master/Dockerfile
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 xvfb -y

ENV VIRTUAL_ENV=/opt/venv_totalsegmentator/

RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH_BACKUP"

RUN pip install fury
RUN pip install totalsegmentator

RUN python `python -c "import os; import totalsegmentator; print(os.path.dirname(totalsegmentator.__file__))"`/download_pretrained_weights.py



######################
# Install Tensorflow #
######################

ENV VIRTUAL_ENV=/opt/venv_tensorflow/

RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH_BACKUP"

RUN pip install tensorflow




###############
# Install app #
###############

WORKDIR ${APP_FOLDER}

COPY . ${APP_FOLDER}

RUN pip install .

COPY --from=helper /app/weights.keras ${APP_FOLDER}/checkpoints/weights.keras

CMD ["docker/app"]
